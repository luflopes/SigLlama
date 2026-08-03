#!/usr/bin/env python
"""Avaliação cross-dataset do modelo Unveiling_Deepfake (backbone Xception).

Os pesos disponibilizados pelos autores (ex.: ffpp_c23.pth) são de um Xception
simples (TransferModel), não da rede tripla completa. Este script:
  - constrói TransferModel('xception', num_out_classes=2) e carrega o checkpoint;
  - reproduz EXATAMENTE o pré-processamento de teste do repositório
    (cv2 BGR->RGB + albumentations 'val': Resize 299, Normalize 0.5/0.5, ToTensor);
  - roda inferência frame-level e reporta AUC, Acc, F1, Precision, Recall,
    Especificidade e Balanced Accuracy (no limiar 0.5 e no melhor limiar/Youden);
  - opcionalmente agrega em nível de vídeo (média de P(fake)).

Convenção de rótulo: 0 = real, 1 = fake; P(fake) = softmax(logits)[:, 1].

NOTA METODOLÓGICA: o pré-processamento de TREINO dos autores concatena o rosto
com a máscara de manipulação ([rosto | máscara | máscara]). Máscaras de
manipulação não existem (de forma justa) em cross-dataset, então aqui usamos
apenas os frames de rosto alinhados de cada dataset — protocolo padrão de
avaliação cross-dataset.

Exemplos:
  # DD-VQA (nosso jsonl: campo 'image' + 'is_real')
  python scripts/eval_unveiling.py \
    --checkpoint unvealing_deepfake_models/ffpp_c23.pth \
    --repo Unveiling_Deepfake \
    --metadata /datasets/deepfake/ddvqa_prepared/test.jsonl \
    --image-key image --images-root /datasets/deepfake/ddvqa_prepared/frames \
    --is-real-key is_real --video-key video_id \
    --output outputs/unveiling/ddvqa_test_scores.jsonl --dataset-name DD-VQA

  # Lista estilo-repo (label,dir por linha; glob de frames no diretório)
  python scripts/eval_unveiling.py \
    --checkpoint unvealing_deepfake_models/ffpp_c23.pth --repo Unveiling_Deepfake \
    --list-file /datasets/celebdf/test_list.txt \
    --output outputs/unveiling/celebdf_scores.jsonl --dataset-name Celeb-DF
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_state_dict_flexible(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    # remove prefixo 'module.' (DataParallel), se houver
    if any(k.startswith("module.") for k in obj):
        obj = {k[len("module."):] if k.startswith("module.") else k: v for k, v in obj.items()}
    return obj


def truthy(v):
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "real", "t", "y"}
    return bool(v)


def read_samples(args):
    """Retorna lista de (image_path, label, video_id). label: 0=real, 1=fake."""
    samples = []
    if args.frames_dir:
        import re
        exts = [e.strip() for e in args.glob_ext.split(",")]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(args.frames_dir, f"*.{ext}")))
        vid_re = re.compile(args.video_regex) if args.video_regex else None
        for f in sorted(files):
            base = os.path.basename(f)
            has_fake = args.fake_substr in base
            has_real = args.real_substr in base
            if has_fake == has_real:  # ambos ou nenhum -> ambíguo
                raise ValueError(
                    f"rótulo ambíguo pelo nome em '{base}' "
                    f"(fake_substr='{args.fake_substr}', real_substr='{args.real_substr}')")
            label = 1 if has_fake else 0
            vid = None
            if vid_re is not None:
                m = vid_re.search(base)
                if m:
                    vid = m.group("vid") if "vid" in m.groupdict() else m.group(1)
            samples.append((f, label, vid))
        return samples
    if args.list_file:
        exts = [e.strip() for e in args.glob_ext.split(",")]
        with open(args.list_file) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split(",")
                label = int(parts[0])
                d = ",".join(parts[1:]).strip()
                files = []
                for ext in exts:
                    files.extend(glob.glob(os.path.join(d, f"*.{ext}")))
                for f in sorted(files):
                    samples.append((f, label, os.path.basename(d.rstrip("/"))))
        return samples

    # jsonl
    with open(args.metadata) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = obj[args.image_key]
            if args.images_root:
                img = img if os.path.isabs(img) else os.path.join(args.images_root, img)
            if args.is_real_key is not None:
                label = 0 if truthy(obj.get(args.is_real_key)) else 1
            else:
                val = obj[args.label_key]
                fake_vals = {v.strip() for v in str(args.fake_values).split(",")}
                label = 1 if str(val).strip() in fake_vals else 0
            vid = str(obj.get(args.video_key)) if args.video_key else None
            samples.append((img, label, vid))
    if args.dedup_by_image:
        seen, dedup = set(), []
        for s in samples:
            if s[0] in seen:
                continue
            seen.add(s[0])
            dedup.append(s)
        samples = dedup
    return samples


class FrameDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            # imagem inválida: retorna tensor preto (será marcado por índice)
            image = np.zeros((299, 299, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image, label, idx


def compute_metrics(labels, probs, preds):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = np.asarray(preds)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auc"] = float("nan")
    out["accuracy"] = float(accuracy_score(labels, preds))
    out["f1"] = float(f1_score(labels, preds, zero_division=0))
    out["precision"] = float(precision_score(labels, preds, zero_division=0))
    out["recall"] = float(recall_score(labels, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    out["specificity"] = float(spec)
    out["balanced_accuracy"] = float((out["recall"] + spec) / 2)
    return out


def best_threshold_youden(labels, probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k])


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", default="Unveiling_Deepfake", help="caminho do repo clonado")
    ap.add_argument("--dataset-name", default="dataset")
    # entrada: jsonl OU list-file
    ap.add_argument("--metadata", help="jsonl com uma imagem por linha")
    ap.add_argument("--image-key", default="image")
    ap.add_argument("--images-root", default="")
    ap.add_argument("--is-real-key", default=None,
                    help="chave booleana onde verdadeiro=real (label 0)")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--fake-values", default="1,fake",
                    help="valores de --label-key que representam fake")
    ap.add_argument("--video-key", default=None)
    ap.add_argument("--dedup-by-image", action="store_true",
                    help="mantém 1 entrada por imagem (útil p/ DD-VQA multi-QA)")
    ap.add_argument("--list-file", help="lista estilo-repo: 'label,dir' por linha")
    # modo diretório de frames com rótulo no nome do arquivo (ex.: WildDeepfake)
    ap.add_argument("--frames-dir", help="diretório com frames; rótulo vem do nome")
    ap.add_argument("--fake-substr", default="fake")
    ap.add_argument("--real-substr", default="real")
    ap.add_argument("--video-regex", default=None,
                    help=r"regex p/ extrair video_id do nome; use grupo (?P<vid>...) "
                         r"ex.: '^\d+_(?:real|fake)_(?P<vid>\d+)_'")
    ap.add_argument("--glob-ext", default="png,jpg,jpeg")
    # execução
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="amostra N frames (debug)")
    ap.add_argument("--output", default=None, help="jsonl de scores por frame")
    args = ap.parse_args()

    if not args.metadata and not args.list_file and not args.frames_dir:
        ap.error("forneça --metadata (jsonl), --list-file ou --frames-dir")

    sys.path.insert(0, os.path.abspath(args.repo))
    from network.xception import TransferModel
    from dataset.transform import xception_default_data_transforms

    if not torch.cuda.is_available():
        print("[ERRO] CUDA indisponível. O modelo exige GPU (features usa .to('cuda')).")
        sys.exit(1)

    samples = read_samples(args)
    if args.limit and args.limit < len(samples):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(samples), size=args.limit, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    n_real = sum(1 for _, l, _ in samples if l == 0)
    n_fake = sum(1 for _, l, _ in samples if l == 1)
    print(f"[{args.dataset_name}] frames: {len(samples)} | real(0)={n_real} fake(1)={n_fake}")
    if n_real == 0 or n_fake == 0:
        print("[AVISO] apenas uma classe presente; AUC ficará indefinido.")

    model = TransferModel("xception", num_out_classes=2)
    model.load_state_dict(load_state_dict_flexible(args.checkpoint), strict=True)
    model = model.cuda().eval()

    ds = FrameDataset(samples, xception_default_data_transforms["val"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=args.num_workers, pin_memory=True)

    all_probs = np.zeros(len(samples), dtype=np.float64)
    all_preds = np.zeros(len(samples), dtype=np.int64)
    done = 0
    with torch.no_grad():
        for images, labels, idxs in loader:
            images = images.cuda(non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            idxs = idxs.numpy()
            all_probs[idxs] = probs.detach().cpu().numpy()
            all_preds[idxs] = preds.detach().cpu().numpy()
            done += len(idxs)
            if done % (args.batch_size * 20) < args.batch_size:
                print(f"  {done}/{len(samples)}")

    labels = np.array([l for _, l, _ in samples], dtype=np.int64)

    m05 = compute_metrics(labels, all_probs, all_preds)
    print(f"\n=== {args.dataset_name} — frame-level (limiar 0.5) ===")
    for k in ["auc", "accuracy", "f1", "precision", "recall", "specificity", "balanced_accuracy"]:
        print(f"  {k:>18}: {m05[k]:.4f}")
    if not np.isnan(m05["auc"]) and m05["auc"] < 0.5:
        print("  [AVISO] AUC < 0.5 — verifique a convenção de rótulo (real/fake).")

    # melhor limiar (Youden) apenas como referência
    if n_real > 0 and n_fake > 0:
        thr = best_threshold_youden(labels, all_probs)
        preds_thr = (all_probs >= thr).astype(np.int64)
        m_best = compute_metrics(labels, all_probs, preds_thr)
        print(f"\n=== {args.dataset_name} — frame-level (limiar Youden = {thr:.4f}) ===")
        for k in ["accuracy", "f1", "precision", "recall", "specificity", "balanced_accuracy"]:
            print(f"  {k:>18}: {m_best[k]:.4f}")

    # nível de vídeo: AUC pela média de P(fake); acc/F1/recall por majority voting
    # (mesmo protocolo dos nossos experimentos cross-dataset).
    has_vid = any(v is not None for _, _, v in samples)
    if has_vid:
        from collections import defaultdict
        vid_scores, vid_preds, vid_labels = defaultdict(list), defaultdict(list), {}
        for (path, lab, vid), p, pr in zip(samples, all_probs, all_preds):
            key = vid if vid is not None else path
            vid_scores[key].append(p)
            vid_preds[key].append(pr)
            vid_labels[key] = lab
        keys = list(vid_scores)
        v_lab = np.array([vid_labels[k] for k in keys])
        v_prob = np.array([float(np.mean(vid_scores[k])) for k in keys])
        v_pred = np.array([1 if np.mean(vid_preds[k]) >= 0.5 else 0 for k in keys])
        if len(set(v_lab.tolist())) == 2:
            mv = compute_metrics(v_lab, v_prob, v_pred)
            print(f"\n=== {args.dataset_name} — video-level (AUC=média P(fake); "
                  f"pred=majority voting), n={len(v_lab)} ===")
            for k in ["auc", "accuracy", "f1", "precision", "recall", "specificity", "balanced_accuracy"]:
                print(f"  {k:>18}: {mv[k]:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            for (path, lab, vid), p, pr in zip(samples, all_probs, all_preds):
                fh.write(json.dumps({"image": path, "label": int(lab),
                                     "video_id": vid, "prob_fake": float(p),
                                     "pred": int(pr)}) + "\n")
        print(f"\nScores por frame salvos em: {args.output}")


if __name__ == "__main__":
    main()
