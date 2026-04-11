# Stage 2 — Plano de Implementação: Fine-tuning para Detecção Explicável de Deepfakes

> Documento de planejamento técnico para o Stage 2 do SigLlama.
> Última atualização: 28/03/2026

---

## 1. Definição do Problema

Deepfakes estão cada vez mais realistas e acessíveis, tornando a detecção automática
essencial. Modelos existentes tipicamente fornecem apenas classificação binária
(real vs. falso), sem explicar *por que* uma imagem é considerada manipulada.

**Objetivo do SigLlama:** Produzir um modelo **pequeno (~1.2B params)**, **explicável**
e capaz de **localizar artefatos**, que receba uma imagem e um prompt textual e
gere uma resposta descrevendo se a imagem é real ou fake, apontando os artefatos
observados e suas localizações.

**Diferencial em relação ao estado da arte:**

| Aspecto             | TruthLens (Google)     | ExDDV (Bucharest)      | SigLlama (nosso)               |
|---------------------|------------------------|------------------------|---------------------------------|
| LLM                 | Gemma2-3B              | BLIP-2/LLaVA/Phi-3    | **TinyLlama 1.1B**              |
| Params totais       | ~3B+                   | 2.7B–7B                | **~1.2B**                       |
| Info localizada     | DINOv2 (implícito)     | Clicks + masking       | Soft tokens (YOLO + landmarks)  |
| Especialização      | MoF (mistura features) | —                      | **MoE (experts por domínio)**   |
| Localização output  | Não                    | Não (externo)          | **Tokens `<loc>` na saída**     |

---

## 2. Estado Atual (completado)

- [x] Estrutura do projeto
- [x] Pipeline de extração de soft tokens (YOLO + MediaPipe)
- [x] Stage 1a — Adapter pre-training no LCS-558K (alignment)
  - Loss final: ~2.38 | Val loss: ~2.43 (step 4000)
  - Checkpoint: `outputs/pretraining/best_adapter.pt`
- [x] Stage 1b — Config para instruction tuning (preparado, não executado)
- [x] Acesso ao FaceForensics++ obtido
- [x] Script de download do FF++ em `./faceforensics/download.py`

---

## 3. Datasets para o Stage 2

### 3.1 DD-VQA (Deepfake Detection Visual Question Answering)

- **Paper:** Zhang et al. *Common Sense Reasoning for DeepFake Detection.* ECCV 2024.
- **Repositório:** https://github.com/Reality-Defender/Research-DD-VQA
- **Base de imagens:** Frames cropados do FaceForensics++ (requer acesso ao FF++)
- **Formato das anotações:**

```json
{
  "manipulateid_videoid": {
    "question_id": {
      "question": "Does the image/the person's eyes/nose/mouth look fake?",
      "answer": ["answer1", "answer2"]
    }
  }
}
```

- **Tipos de manipulação:** Deepfakes (0), Face2Face (1), FaceShift (2),
  FaceSwap (3), Original (5), NeuralTextures (6)
- **Tipos de pergunta:**
  - Geral: "Does the person in the image look fake?"
  - Fine-grained: "Do the person's eyes/nose/mouth/eyebrows/skin look fake?"
- **Volume:** ~9000 HITs anotados via Amazon Mechanical Turk
- **Uso planejado:** Dataset principal para Stage 2a (fine-tuning sem MoE)

### 3.2 ExDDV (Explainable Deepfake Detection in Video)

- **Paper:** Hondru et al. *ExDDV: A New Dataset for Explainable Deepfake Detection
  in Video.* WACV 2026.
- **Repositório:** https://github.com/vladhondru25/ExDDV
- **Licença:** CC BY-NC-SA 4.0
- **Conteúdo:** ~5.4K vídeos (1K reais, 4.4K falsos) de FF++, DeeperForensics,
  DFDC e BioDeepAV
- **Anotações:**
  - Descrições textuais dos artefatos (por vídeo)
  - Coordenadas de clicks (x, y, frame) indicando localização dos artefatos
  - Nível de dificuldade (easy / medium / hard)
- **Volume:** 21.282 clicks totais, ~4.9 clicks por vídeo
- **Uso planejado:**
  - Stage 2c — treino de localização (tokens `<loc>`)
  - Avaliação complementar

### 3.3 FaceForensics++ (base de imagens)

- **Paper:** Rössler et al. *FaceForensics++: Learning to Detect Manipulated
  Facial Images.* ICCV 2019.
- **Acesso:** Formulário aprovado ✓
- **Download:** `python faceforensics/download.py ./ -d all -c c23 -t videos --server EU2`
- **Conteúdo:** 1000 vídeos originais + 5 métodos de manipulação
  (Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures)
- **Compressões:** raw, c23, c40
- **Uso:** Base de imagens para DD-VQA e ExDDV

### 3.4 DVF (Diffusion Video Forensics) — opcional

- **Paper:** Song et al. *On Learning Multi-Modal Forgery Representation for
  Diffusion Generated Video Detection.* NeurIPS 2024.
- **Repositório:** https://github.com/SparkleXFantasy/MM-Det
- **Download:** Google Drive / BaiduNetDisk
- **Uso potencial:** Expandir o treinamento para conteúdo totalmente sintético
  (gerado por difusão). Não prioritário na primeira iteração.

---

## 4. Plano de Implementação por Fases

### Fase 2a — Fine-tuning supervisionado (sem MoE, sem soft tokens)

**Objetivo:** Ensinar o modelo a detectar e explicar deepfakes usando VQA.

#### 4a.1 Preparação dos dados (no servidor)

```
Ações:
  1. Baixar FF++ (c23, videos) no servidor
  2. Extrair frames dos vídeos (1 frame a cada N frames, ou frames-chave)
  3. Croppar faces dos frames (usando YOLO ou alinhamento do FF++)
  4. Clonar repositório DD-VQA e mapear anotações para os frames extraídos
  5. Gerar metadata JSON unificado no formato:
     {"image_path": "...", "question": "...", "answer": "..."}
```

**Script a criar:** `scripts/prepare_ddvqa.py`

#### 4a.2 Dataset PyTorch

**Arquivo:** `data/deepfake_vqa_dataset.py`

```
Formato de entrada:
  [visual_patches]  <s> {question}\n{answer} </s> [pad...]
                    |--- label=-100 ---|--- predicted ---|

Campos retornados pelo __getitem__:
  - pixel_values: imagem processada pelo SigLIP processor
  - input_ids: tokens do texto (question + answer)
  - attention_mask: máscara de atenção
  - labels: input_ids com masking em question + BOS + padding
```

#### 4a.3 Modelo Stage 2a

**Arquivo:** `models/sigllama_finetune.py`

```
Componentes:
  - SigLIP (frozen)
  - Adapter (trainable, inicializado de best_adapter.pt)
  - TinyLlama + LoRA (trainable nos módulos q_proj, v_proj)

Diferenças em relação ao Stage 1:
  - LoRA aplicado ao TinyLlama (via peft)
  - Adapter continua treinando (ablação do TruthLens mostra que é essencial)
  - Label masking: loss apenas nos tokens da resposta
```

**Hiperparâmetros iniciais:**

| Parâmetro                   | Valor          | Referência          |
|-----------------------------|----------------|---------------------|
| Learning rate               | 2e-4           | ExDDV (LoRA)        |
| LoRA rank                   | 16             | Config existente    |
| LoRA alpha                  | 32             | 2× rank (padrão)   |
| LoRA target modules         | q_proj, v_proj | Config existente    |
| Batch size                  | 16             | ExDDV               |
| Gradient accumulation       | 2              | Efetivo: bs=32      |
| Epochs                      | 5–10           | TruthLens           |
| Max text length             | 256            | Respostas maiores   |
| Optimizer                   | AdamW          | Padrão              |
| Scheduler                   | Cosine warmup  | Padrão              |
| Mixed precision             | bf16           | Como Stage 1        |
| Gradient checkpointing      | true           | Como Stage 1        |

#### 4a.4 Script de treinamento

**Arquivo:** `training/finetune.py`

```
Estrutura (similar ao pretrain.py):
  1. Carregar config YAML
  2. Inicializar tokenizer, processor, dataset
  3. Construir modelo com LoRA via peft
  4. Carregar adapter do Stage 1
  5. Loop de treino com accelerate
  6. Validação periódica
  7. Geração de amostras (pergunta → resposta)
  8. Salvar checkpoints (latest + best)
```

#### 4a.5 Config

**Arquivo:** `configs/finetuning.yaml` (atualizar o existente)

#### 4a.6 Avaliação

**Métricas:**
- **Detecção:** Accuracy (real vs fake) via LLM-as-judge ou parsing da resposta
- **Qualidade da explicação:** Sentence-BERT, BERTScore, BLEU, ROUGE-L, CIDEr
- **Métricas específicas:** Comparar com baselines do DD-VQA

**Arquivo:** `evaluation/evaluate.py` (implementar)

---

### Fase 2b — Integração de Soft Tokens

**Objetivo:** Adicionar informação estruturada (detecções YOLO + landmarks faciais)
como tokens extras na sequência de entrada.

#### 4b.1 Extração de soft tokens do FF++

```
Ações:
  1. Executar scripts/extract_soft_tokens.py nos frames do FF++
  2. Gerar NDJSON com detecções + landmarks por frame
  3. Alinhar com o metadata do DD-VQA
```

**Script a criar:** `scripts/extract_ff_soft_tokens.py`

#### 4b.2 Atualização do modelo

**Mudanças em `models/sigllama_finetune.py`:**

```
Sequência de entrada:
  [soft_tokens] [visual_patches] <s> {question}\n{answer} </s>

Novos componentes ativados:
  - SoftTokenEmbedder (já existe como stub em models/soft_token_embedder.py)
  - Concatenação de soft_embeds + visual_embeds + text_embeds
  - Labels mascarados para soft tokens e visual tokens
```

#### 4b.3 Avaliação comparativa

Comparar métricas de 2b vs 2a para medir o impacto dos soft tokens.

---

### Fase 2c — Ativação do MoE

**Objetivo:** Especializar o modelo por domínio de deepfake via Mixture of Experts.

#### 4c.1 Preparação dos dados com labels de domínio

```
Labels de domínio do FF++:
  - Deepfakes → "GAN"
  - Face2Face → "Reenactment"
  - FaceSwap → "Swap"
  - FaceShifter → "Swap"
  - NeuralTextures → "Neural"
  - Original → "Real"
```

#### 4c.2 Implementação do MoE

**Arquivos existentes a completar:**
- `models/moe/router.py` — Router top-k com load balancing
- `models/moe/experts.py` — Expert modules (LoRA por domínio)
- `models/moe/moe_layer.py` — Composição: recebe hidden states do TinyLlama,
  aplica routing, soma das saídas ponderadas dos experts

**Posição do MoE:** Aplicado nos hidden states do TinyLlama antes do LM head.

**Loss:** LM loss + λ × auxiliary load-balancing loss (λ = 0.01)

#### 4c.3 Avaliação

- Comparar 2c vs 2b para medir o impacto do MoE
- Analisar distribuição de routing por tipo de manipulação
- Verificar se experts se especializam nos domínios esperados

---

### Fase 2d — Localização de Artefatos (tokens `<loc>`)

**Objetivo:** Gerar coordenadas de artefatos diretamente no texto de saída.

#### 4d.1 Dados

Usar anotações de clicks do **ExDDV** para construir pares (imagem, explicação
com coordenadas):

```
Formato de saída esperado:
  "The mouth shows distortion <loc:0.45,0.72>. The eyebrows flicker
  <loc:0.50,0.35>. This is a deepfake."
```

#### 4d.2 Tokens especiais

Adicionar tokens `<loc>` e `</loc>` ao vocabulário do TinyLlama.
As coordenadas são normalizadas [0,1] e discretizadas (e.g., 2 casas decimais).

#### 4d.3 Treinamento

Fine-tune adicional usando ExDDV, com loss nos tokens de localização incluído.

---

## 5. Ambiente de Execução

| Ambiente  | Uso                                               |
|-----------|---------------------------------------------------|
| **Local** | Desenvolvimento de código, testes com amostras pequenas, debug, versionamento |
| **Servidor (tarkin)** | Download de datasets, extração de frames, extração de soft tokens, treinamento, avaliação |

**Fluxo de trabalho:**

```
Local: editar código → git push
Servidor: git pull → preparar dados → treinar → salvar resultados
Local: rsync resultados ← servidor → analisar → iterar
```

**Servidor:**
- GPU: NVIDIA RTX A6000 (48GB VRAM)
- CUDA: 12.4
- PyTorch: cu124
- Caminho do repo: `~/repos/SigLlama/`
- Datasets: `/datasets/deepfake/` (ou a definir para FF++)

---

## 6. Cronograma Sugerido

| Semana | Atividade                                              |
|--------|--------------------------------------------------------|
| 1      | Download FF++ no servidor, extrair frames, preparar DD-VQA |
| 1–2    | Implementar `deepfake_vqa_dataset.py` + `sigllama_finetune.py` |
| 2      | Implementar `training/finetune.py` + config             |
| 2–3    | **Treinar Fase 2a** no servidor, avaliar resultados     |
| 3–4    | Integrar soft tokens (Fase 2b), retreinar, comparar     |
| 4–5    | Implementar e ativar MoE (Fase 2c), retreinar, comparar |
| 5–6    | Localização com ExDDV (Fase 2d)                         |
| 6–7    | Ablation study completo, avaliação final                |
| 7–8    | Redação dos resultados                                  |

---

## 7. Métricas de Avaliação

### Detecção

| Métrica    | Descrição                                    |
|------------|----------------------------------------------|
| Accuracy   | Classificação real/fake (parsing ou LLM-judge) |
| F1-Score   | Equilíbrio entre precision e recall          |

### Qualidade da explicação

| Métrica        | Descrição                                         |
|----------------|---------------------------------------------------|
| Sentence-BERT  | Similaridade semântica (cosine) entre gerado e GT |
| BERTScore      | F1 token-level usando BERT embeddings             |
| BLEU (1–4)     | Sobreposição de n-gramas                          |
| ROUGE-L        | Longest common subsequence                        |
| METEOR         | Combinação de precision, recall, sinônimos        |
| CIDEr          | Consenso entre referências (TF-IDF ponderado)     |

### Localização (Fase 2d)

| Métrica    | Descrição                                              |
|------------|--------------------------------------------------------|
| MAE        | Erro médio absoluto das coordenadas preditas vs GT     |
| Recall@r   | % de predições dentro de raio r pixels do GT           |

### MoE (Fase 2c)

| Métrica              | Descrição                                      |
|----------------------|------------------------------------------------|
| Expert utilization   | Distribuição de routing por expert              |
| Domain alignment     | Se experts se especializam no domínio esperado  |
| Load balance loss    | Valor da auxiliary loss durante treino          |

---

## 8. Referências

1. Zhang, Y. et al. *Common Sense Reasoning for DeepFake Detection.* ECCV 2024.
   — Dataset DD-VQA. https://github.com/Reality-Defender/Research-DD-VQA

2. Hondru, V. et al. *ExDDV: A New Dataset for Explainable Deepfake Detection
   in Video.* WACV 2026. — Dataset com explicações + clicks.
   https://github.com/vladhondru25/ExDDV

3. Kundu, R. et al. *TruthLens: Explainable DeepFake Detection for Face
   Manipulated and Fully Synthetic Data.* arXiv:2503.15867, 2025.
   — Arquitetura PaliGemma2 + DINOv2 com Mixture of Features.

4. Rössler, A. et al. *FaceForensics++: Learning to Detect Manipulated Facial
   Images.* ICCV 2019. — Dataset base com 5 métodos de manipulação.

5. Beyer, L. et al. *PaLiGemma 2: A Family of Versatile VLMs for Transfer.*
   arXiv:2412.03555, 2024. — Inspiração arquitetural (soft tokens + VLM).

6. Zhang, P. et al. *TinyLlama: An Open-Source Small Language Model.*
   arXiv:2401.02385, 2024. — Decoder autoregressivo do SigLlama.

7. Zhai, X. et al. *Sigmoid Loss for Language Image Pre-Training (SigLIP).*
   ICCV 2023. — Encoder visual do SigLlama.

8. Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.*
   ICLR 2022. — Fine-tuning eficiente do LLM.

9. Shazeer, N. et al. *Outrageously Large Neural Networks: The Sparsely-Gated
   Mixture-of-Experts Layer.* ICLR 2017. — Base teórica do MoE.

10. Liu, H. et al. *Visual Instruction Tuning (LLaVA).* NeurIPS 2023.
    — Referência para treino multimodal em dois estágios.

11. Song, X. et al. *On Learning Multi-Modal Forgery Representation for Diffusion
    Generated Video Detection (DVF/MM-Det).* NeurIPS 2024.
    — Dataset de vídeos sintéticos por difusão.

---

## 9. Estrutura de Arquivos a Criar/Modificar

```
Novos arquivos:
  scripts/prepare_ddvqa.py          ← Preparação do DD-VQA (frames + metadata)
  scripts/extract_ff_soft_tokens.py ← Soft tokens para frames do FF++
  data/deepfake_vqa_dataset.py      ← Dataset PyTorch para DD-VQA
  models/sigllama_finetune.py       ← Modelo Stage 2 (SigLIP + Adapter + LoRA)
  configs/finetuning.yaml           ← Atualizar com paths e params reais

Arquivos existentes a completar:
  training/finetune.py              ← Loop de treino completo
  evaluation/evaluate.py            ← Script de avaliação com métricas
  evaluation/metrics.py             ← Implementar métricas (BLEU, ROUGE, etc.)
  models/moe/router.py              ← Router top-k (Fase 2c)
  models/moe/experts.py             ← Expert modules (Fase 2c)
  models/moe/moe_layer.py           ← MoE layer (Fase 2c)
  models/soft_token_embedder.py     ← Embedder de soft tokens (Fase 2b)
  models/sigllama.py                ← Forward pass completo (Fase 2c)
```

---

*Este documento serve como referência viva e será atualizado conforme
o projeto avança.*
