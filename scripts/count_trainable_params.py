#!/usr/bin/env python
"""Conta (não estima) os parâmetros totais e treináveis do FaceGroundVLM.

Reproduz exatamente a instanciação usada no treinamento final:

  * Ramo VLM (Estágio 3 / A4 / A6): ``build_model`` a partir de
    ``configs/tinyllava_stage3_cls_loc.yaml`` com ``use_lora=True`` e
    ``train_connector=True``. Fornece: SigLIP (congelado), Adaptador DINOv2,
    Conector MLP (treinável), TinyLLaMA + LoRA (treinável) e o DINOv2 base
    (congelado, sem LoRA neste ramo).

  * Ramo classificador (Estágio 1): ``DINOv2LoRAClassifier`` a partir de
    ``configs/dino_lora_classifier.yaml``. Fornece o LoRA do DINOv2
    (treinável no Estágio 1) e as duas cabeças (binária + tipo de
    manipulação).

Os contadores usam ``p.numel()`` e ``p.requires_grad`` diretamente sobre os
módulos reais — nenhum número é estimado.

NÃO carrega checkpoints treinados (as contagens dependem apenas da
arquitetura). Ainda assim, ``from_pretrained`` baixa/lê os pesos-base do
SigLIP, DINOv2 e TinyLLaMA; execute onde houver o cache do HuggingFace
(ex.: no servidor). Use HF_HUB_OFFLINE=1 se o cache já existir.

Uso:
    python scripts/count_trainable_params.py \
        --stage3-config configs/tinyllava_stage3_cls_loc.yaml \
        --dino-config   configs/dino_lora_classifier.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Permite rodar a partir da raiz do repositório.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def count_params(params) -> tuple[int, int]:
    """Retorna (total, treináveis) para um iterável de parâmetros."""
    total = 0
    trainable = 0
    for p in params:
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def m(n: int) -> str:
    return f"{n / 1e6:.3f}M"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage3-config", default="configs/tinyllava_stage3_cls_loc.yaml")
    ap.add_argument("--dino-config", default="configs/dino_lora_classifier.yaml")
    args = ap.parse_args()

    stage3_cfg = yaml.safe_load(open(REPO_ROOT / args.stage3_config))
    dino_cfg = yaml.safe_load(open(REPO_ROOT / args.dino_config))

    # Evita depender de arquivos de checkpoint: as contagens independem dos
    # valores dos pesos. A estrutura de LoRA/cabeças é idêntica à treinada.
    stage3_cfg["load_tinyllava_weights"] = False
    stage3_cfg["dino_lora_checkpoint"] = None

    print("=" * 78)
    print("CONFIGURAÇÃO EFETIVA")
    print("=" * 78)
    print(f"  VLM  (Estágio 3): lora_rank={stage3_cfg.get('lora_rank')} "
          f"lora_alpha={stage3_cfg.get('lora_alpha')} "
          f"target={stage3_cfg.get('lora_target_modules')} "
          f"train_connector={stage3_cfg.get('train_connector')} "
          f"use_dino={stage3_cfg.get('use_dino')}")
    print(f"  DINO (Estágio 1): lora_rank={dino_cfg.get('lora_rank')} "
          f"lora_alpha={dino_cfg.get('lora_alpha')} "
          f"target={dino_cfg.get('lora_target_modules')} "
          f"head_hidden_dim={dino_cfg.get('head_hidden_dim')}")

    # ------------------------------------------------------------------
    # Ramo classificador DINOv2 (Estágio 1): LoRA do DINOv2 + duas cabeças
    # ------------------------------------------------------------------
    from models.dino_lora_classifier import DINOv2LoRAClassifier

    clf = DINOv2LoRAClassifier(
        dino_model=dino_cfg.get("dinov2_model", "facebook/dinov2-large"),
        lora_rank=int(dino_cfg.get("lora_rank", 16)),
        lora_alpha=int(dino_cfg.get("lora_alpha", 32)),
        lora_dropout=float(dino_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=dino_cfg.get("lora_target_modules"),
        head_hidden_dim=int(dino_cfg.get("head_hidden_dim", 256)),
        head_dropout=float(dino_cfg.get("head_dropout", 0.3)),
        use_moe=False,
    )
    dino_total, dino_train = count_params(clf.dinov2.parameters())  # base+LoRA; treinável=LoRA
    dino_base = dino_total - dino_train                              # backbone congelado
    binhead_total, binhead_train = count_params(clf.binary_head.parameters())
    forghead_total, forghead_train = count_params(clf.forgery_head.parameters())

    del clf
    import gc
    gc.collect()

    # ------------------------------------------------------------------
    # Ramo VLM (Estágio 3): SigLIP + Adaptador + Conector + TinyLLaMA(+LoRA)
    # ------------------------------------------------------------------
    from models import build_model

    vlm = build_model(stage3_cfg, use_lora=True)
    siglip_total, siglip_train = count_params(vlm.siglip.parameters())
    adapter_total, adapter_train = count_params(vlm.dino_adapter.parameters())
    connector_total, connector_train = count_params(vlm.connector.parameters())
    llm_total, llm_train = count_params(vlm.llm.parameters())  # base+LoRA; treinável=LoRA

    # ------------------------------------------------------------------
    # Tabela por componente (espelha a Tabela 4.1 da dissertação)
    # ------------------------------------------------------------------
    rows = [
        ("SigLIP-So400m",            "Encoder semântico",                 siglip_total,  siglip_train),
        ("DINOv2-Large",             "Encoder deepfake-aware (base)",     dino_base,     0),
        ("  + LoRA DINOv2",          "Adaptação r={}".format(dino_cfg.get("lora_rank")), dino_train, dino_train),
        ("Adaptador (proj. DINOv2)", "DINOv2 -> LLM",                     adapter_total, adapter_train),
        ("Conector MLP",             "Projeção visual -> LLM",            connector_total, connector_train),
        ("TinyLLaMA (+LoRA)",        "Modelo de linguagem",               llm_total,     llm_train),
        ("Cabeça Binária",           "Veredito (Real/Fake)",              binhead_total, binhead_train),
        ("Cabeça de Manipulação",    "Tipo de forjamento (6 classes)",    forghead_total, forghead_train),
    ]

    print("\n" + "=" * 78)
    print("PARÂMETROS POR COMPONENTE")
    print("=" * 78)
    print(f"{'Componente':<26}{'Função':<32}{'Total':>10}{'Treináveis':>12}")
    print("-" * 80)
    for name, role, tot, tr in rows:
        print(f"{name:<26}{role:<32}{m(tot):>10}{m(tr):>12}")

    total_params = siglip_total + dino_total + adapter_total + connector_total + llm_total + binhead_total + forghead_total
    total_train = siglip_train + dino_train + adapter_train + connector_train + llm_train + binhead_train + forghead_train

    print("-" * 80)
    print(f"{'TOTAL':<26}{'':<32}{m(total_params):>10}{m(total_train):>12}")
    print(f"{'% treináveis do total':<58}{100 * total_train / max(total_params, 1):>12.2f}%")

    # Subtotais úteis para o texto.
    gen_path = adapter_train + connector_train + llm_train
    print("\n" + "=" * 78)
    print("SUBTOTAIS PARA O TEXTO")
    print("=" * 78)
    print(f"  Caminho de geração VLM (Adaptador + Conector + LoRA LLM): {m(gen_path)}")
    print(f"  + LoRA DINOv2 + cabeças (Estágio 1):                      {m(dino_train + binhead_train + forghead_train)}")
    print(f"  Treináveis totais (todos os estágios):                    {m(total_train)}")
    print(f"  Parâmetros totais:                                        {total_params/1e9:.3f}B")

    print("\n" + "=" * 78)
    print("LINHAS LaTeX (para a Tabela 4.1)")
    print("=" * 78)
    def latex_num(n):
        # formato brasileiro: milhão com uma casa decimal e vírgula
        return f"{n/1e6:.1f}M".replace(".", ",")
    print(f"SigLIP-So400m  & \\textit{{Encoder}} semântico     & {siglip_total/1e6:.0f}M  & 0,0 \\\\")
    print(f"DINOv2-Large   & \\textit{{Encoder}} deepfake-aware & {dino_base/1e6:.0f}M & {latex_num(dino_train)} \\\\")
    print(f"Adaptador MLP     & Projeção DINOv2 $\\rightarrow$ LLM & {latex_num(adapter_total)} & {latex_num(adapter_train)} \\\\")
    print(f"Conector MLP   & Projeção visual $\\rightarrow$ LLM & {latex_num(connector_total)} & {latex_num(connector_train)} \\\\")
    print(f"TinyLLaMA      & Modelo de linguagem    & 1,1B  & {latex_num(llm_train)} \\\\")
    print(f"Cabeça Binária   & Veredito (Real/Fake) & {latex_num(binhead_total)} & {latex_num(binhead_train)} \\\\")
    print(f"Cabeça de Manipulação & Tipo de forjamento & {latex_num(forghead_total)} & {latex_num(forghead_train)} \\\\")
    print(f"\\textbf{{Total}} & & \\textbf{{{total_params/1e9:.2f}B}} & \\textbf{{{latex_num(total_train)}}} \\\\")


if __name__ == "__main__":
    main()
