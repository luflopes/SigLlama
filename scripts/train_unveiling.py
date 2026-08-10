#!/usr/bin/env python
"""Treino do Xception_Net completo (Unveiling_Deepfake) no FF++ (c23).

Adapta [Unveiling_Deepfake/train.py] para:
  - 1 GPU (sem DataParallel; a fixação de 4 GPUs do original é removida);
  - dataset baseado no nosso ``ff_classification`` (JSONL com ``image``/``label``),
    usando faces LIMPAS (sem a concatenação de máscara do pipeline ``process/``,
    que vaza o rótulo) — protocolo alinhado ao texto do artigo (RetinaFace crop
    299x299 + normalização), justo para in- e cross-dataset;
  - batch pequeno + acúmulo de gradiente (o ramo DCT faz upsample x8, pesado
    em VRAM em 1 GPU);
  - amostragem balanceada real/fake e seleção do melhor checkpoint por AUC de
    validação.

Convenção de rótulo: 0=real, 1=fake (bate com o jsonl e com softmax[:,1]).

A rede/loss são importadas do repo clonado (``--repo``). Os 3 backbones Xception
são inicializados a partir do peso ImageNet ``xception-b5690688.pth`` (via
``--xception-pretrained``); sem ele, o treino parte do zero.

Exemplo::

    python scripts/train_unveiling.py \
      --repo Unveiling_Deepfake \
      --data-root /datasets/deepfake/ff_classification \
      --xception-pretrained unvealing_deepfake_models/xception-b5690688.pth \
      --batch-size 8 --grad-accum 2 --epochs 15 \
      --output-dir outputs/unveiling_train
"""
import argparse
import datetime
import glob
import json
import logging
import os
import random
import sys


def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--repo", default="Unveiling_Deepfake",
                    help="caminho do repo Unveiling_Deepfake clonado")
    # dados: --data-root (com train.jsonl/val.jsonl/frames) OU chaves separadas
    ap.add_argument("--data-root", default=None,
                    help="dir com train.jsonl, val.jsonl e frames/ (ff_classification)")
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--val-jsonl", default=None)
    ap.add_argument("--frames-root", default=None,
                    help="dir das imagens (default: <data-root>/frames)")
    ap.add_argument("--image-key", default="image")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--video-key", default="video_id")
    # backbone ImageNet
    ap.add_argument("--xception-pretrained",
                    default="unvealing_deepfake_models/xception-b5690688.pth",
                    help="peso ImageNet do Xception para inicializar os backbones")
    # otimização (defaults do artigo)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="passos de acúmulo de gradiente (batch efetivo = bs*grad_accum)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--step-size", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--bml-method", choices=["mi", "auto", "hyper"], default="mi",
                    help="balanceamento das perdas: 'mi'=soma (código do repo); "
                         "'auto'=AutomaticWeightedLoss (o que o ARTIGO descreve); "
                         "'hyper'=pesos fixos de --scales")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu", default="0", help="GPU visível (CUDA_VISIBLE_DEVICES)")
    # subamostragem (para caber no tempo de 1 GPU)
    ap.add_argument("--frames-per-video", type=int, default=0,
                    help="mantém no máx. N frames por vídeo no TREINO (0=todos)")
    ap.add_argument("--max-train", type=int, default=0,
                    help="limita o total de amostras de treino (0=sem limite)")
    ap.add_argument("--max-val", type=int, default=0,
                    help="limita o total de amostras de validação (0=sem limite)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--output-dir", default="outputs/unveiling_train")
    ap.add_argument("--keep-epoch-ckpts", action="store_true",
                    help="salva um checkpoint por época (ocupa muito disco); "
                         "por padrão salva APENAS o melhor (best.pkl)")
    return ap.parse_args()


def setup_logging(output_path):
    log_file = os.path.join(output_path, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("train_unveiling")


def set_random_seed(seed, torch):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def subsample_frames_per_video(rows, n, video_key, image_key):
    """Mantém no máx. n frames por vídeo (ordenados por nome de imagem)."""
    from collections import defaultdict
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[str(r.get(video_key))].append(r)
    out = []
    for vid, group in by_vid.items():
        group = sorted(group, key=lambda r: r.get(image_key, ""))
        out.extend(group[:n] if n > 0 else group)
    return out


def build_dataset_class(Dataset, cv2, transforms):
    class FFJsonlDataset(Dataset):
        def __init__(self, rows, frames_root, image_key, label_key, train):
            self.rows = rows
            self.frames_root = frames_root
            self.image_key = image_key
            self.label_key = label_key
            self.transform = transforms["train"] if train else transforms["val"]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            import numpy as np
            row = self.rows[idx]
            path = os.path.join(self.frames_root, row[self.image_key])
            image = cv2.imread(path)
            if image is None:
                image = np.zeros((299, 299, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)["image"]
            label = int(row[self.label_key])
            return image, label

    return FFJsonlDataset


def main():
    args = parse_args()

    # 1 GPU: fixa a visibilidade ANTES de importar torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from sklearn.metrics import roc_auc_score, f1_score

    # importa o repo
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)

    # aponta o peso ImageNet do Xception (usado ao instanciar os 3 backbones).
    import network.xception as xmod
    if args.xception_pretrained and os.path.exists(args.xception_pretrained):
        xmod.PRETAINED_WEIGHT_PATH = os.path.abspath(args.xception_pretrained)
    from network.mymodel_bdct_dfcs_triplet_mi_loss import Xception_Net
    from losses.mi_loss import loss_functions
    from dataset.transform import xception_default_data_transforms

    # resolve caminhos de dados
    train_jsonl = args.train_jsonl
    val_jsonl = args.val_jsonl
    frames_root = args.frames_root
    if args.data_root:
        train_jsonl = train_jsonl or os.path.join(args.data_root, "train.jsonl")
        val_jsonl = val_jsonl or os.path.join(args.data_root, "val.jsonl")
        frames_root = frames_root or os.path.join(args.data_root, "frames")
    if not (train_jsonl and val_jsonl and frames_root):
        raise SystemExit("Forneça --data-root OU (--train-jsonl e --val-jsonl e --frames-root)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logging(output_path)
    set_random_seed(args.seed, torch)

    if not torch.cuda.is_available():
        logger.error("CUDA indisponível. Xception_Net.features() usa .to('cuda').")
        raise SystemExit(1)

    if args.xception_pretrained and os.path.exists(args.xception_pretrained):
        logger.info("Backbones Xception inicializados de: %s", xmod.PRETAINED_WEIGHT_PATH)
    else:
        logger.warning("Peso ImageNet do Xception não encontrado (%s): treino do zero.",
                       args.xception_pretrained)

    # carrega dados
    train_rows = read_jsonl(train_jsonl)
    val_rows = read_jsonl(val_jsonl)
    if args.frames_per_video > 0:
        before = len(train_rows)
        train_rows = subsample_frames_per_video(
            train_rows, args.frames_per_video, args.video_key, args.image_key)
        logger.info("Subamostragem por vídeo: %d -> %d frames de treino",
                    before, len(train_rows))
    if args.max_train > 0 and args.max_train < len(train_rows):
        random.shuffle(train_rows)
        train_rows = train_rows[:args.max_train]
        logger.info("Treino limitado a %d amostras", len(train_rows))
    if args.max_val > 0 and args.max_val < len(val_rows):
        # embaralha antes de cortar: o val.jsonl vem ordenado por método
        # (reais primeiro), então um corte sequencial deixaria só uma classe.
        random.shuffle(val_rows)
        val_rows = val_rows[:args.max_val]
        logger.info("Validação limitada a %d amostras (amostragem aleatória)", len(val_rows))

    n_real = sum(1 for r in train_rows if int(r[args.label_key]) == 0)
    n_fake = len(train_rows) - n_real
    logger.info("Treino: %d (real=%d, fake=%d) | Val: %d",
                len(train_rows), n_real, n_fake, len(val_rows))

    DsCls = build_dataset_class(Dataset, cv2, xception_default_data_transforms)
    train_ds = DsCls(train_rows, frames_root, args.image_key, args.label_key, train=True)
    val_ds = DsCls(val_rows, frames_root, args.image_key, args.label_key, train=False)

    # amostragem balanceada
    if n_real > 0 and n_fake > 0:
        wr, wf = 1.0 / n_real, 1.0 / n_fake
        weights = [wr if int(r[args.label_key]) == 0 else wf for r in train_rows]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=train_shuffle, sampler=sampler,
        drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    model = Xception_Net().to(device)

    # loss e otimização (default idêntico ao train.py original: bml_method='mi'=soma)
    loss_function = loss_functions(
        method="mi", mi_calculator="kl", temperature=1.5, bml_method=args.bml_method,
        scales=[1, 2, 10], dec_loss=True, gia_loss=True, device="cuda:0")
    logger.info("Balanceamento de perdas (bml_method): %s", args.bml_method)
    # com 'auto' (AutomaticWeightedLoss) os pesos são parâmetros treináveis e
    # precisam entrar no otimizador.
    params = list(model.parameters())
    if args.bml_method == "auto":
        loss_function.balance_loss = loss_function.balance_loss.to(device)
        params += list(loss_function.balance_loss.parameters())
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_metric = -1.0   # métrica de seleção (AUC; cai p/ acurácia se AUC indefinido)
    best_auc = float("nan")
    iteration = 0
    for epoch in range(args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)
        logger.info("-" * 10)
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        optimizer.zero_grad()
        for step, (image, labels) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(image)
            _, preds = torch.max(outputs["out"].data, 1)
            losses = loss_function.criterion(outputs, labels)
            loss = loss_function.balance_mult_loss(losses)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % args.log_every):
                logger.info("iter %d loss: %.6f acc: %.6f", iteration,
                            iter_loss / args.batch_size,
                            iter_corrects.item() / args.batch_size)
        # flush de gradiente residual
        if len(train_loader) % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        n_train = max(len(train_ds), 1)
        logger.info("epoch train loss: %.6f acc: %.6f",
                    train_loss / n_train, train_corrects.item() / n_train)

        # validação
        model.eval()
        all_labels, all_probs, all_preds = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for image, labels in val_loader:
                image = image.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = model(image)
                probs = F.softmax(outputs["out"], dim=1)[:, 1]
                _, preds = torch.max(outputs["out"].data, 1)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
                losses = loss_function.criterion(outputs, labels)
                val_loss += loss_function.balance_mult_loss(losses).data.item()
        n_val = max(len(val_ds), 1)
        val_acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = float("nan")
        val_f1 = f1_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0
        logger.info("epoch val loss: %.6f acc: %.6f auc: %.6f f1: %.6f",
                    val_loss / n_val, val_acc, val_auc, val_f1)

        scheduler.step()

        # salva o melhor por AUC (fallback p/ acurácia se AUC indefinido).
        # Por padrão salva APENAS best.pkl; --keep-epoch-ckpts guarda por época.
        sel = val_acc if np.isnan(val_auc) else val_auc
        if sel > best_metric:
            best_metric = sel
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))
            logger.info("  -> novo melhor (seleção=%.6f, auc=%.6f) salvo best.pkl",
                        sel, val_auc)
        if args.keep_epoch_ckpts:
            torch.save(model.state_dict(),
                       os.path.join(output_path, f"epoch_{epoch + 1}.pkl"))

    logger.info("Treino concluído. Melhor métrica de seleção: %.6f (AUC=%.6f)",
                best_metric, best_auc)
    logger.info("Checkpoints em: %s", output_path)


if __name__ == "__main__":
    main()
