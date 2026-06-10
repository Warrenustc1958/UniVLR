<div align="center">

# 					UniVLR 🧠👁️

###   Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs

​				[![arXiv](https://img.shields.io/badge/arXiv-2605.11856-b31b1b.svg)](https://arxiv.org/abs/2605.11856)[![ModelScope Stage1](https://img.shields.io/badge/ModelScope-UniVLR--Stage--1--7B-orange.svg)](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage1-7B)[![ModelScope Stage2](https://img.shields.io/badge/ModelScope-UniVLR--Stage--2--7B-blue.svg)](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage2-7B)[![Backbone](https://img.shields.io/badge/Backbone-Qwen2.5--VL--7B-yellow.svg)](https://github.com/QwenLM/Qwen2.5-VL)[![Evaluation](https://img.shields.io/badge/Eval-VLMEvalKit-pink.svg)](https://github.com/open-compass/VLMEvalKit)

**Official implementation of [UniVLR: Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs](https://arxiv.org/abs/2605.11856).**

UniVLR turns multimodal reasoning into a compact **visual latent workspace**: text reasoning traces and auxiliary visual evidence are rendered onto a shared canvas, compressed by the frozen vision encoder, and used as latent supervision for MLLM inference.

</div>

<p align="center">
  <a href="#-news">News</a> •
  <a href="#-model-zoo">Model Zoo</a> •
  <a href="#-method-overview">Method</a> •
  <a href="#-results">Results</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 🔥 News

- **2026.06.10 Released checkpoints:** [UniVLR-Stage1-7B](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage1-7B) and [UniVLR-Stage2-7B](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage2-7B) are available on ModelScope.
- **2026.05.12 Paper online:** the UniVLR paper is available on [arXiv](https://arxiv.org/abs/2605.11856).

## ✨ Highlights

- **Unified visual workspace.** Textual reasoning steps and auxiliary visual evidence are rendered into the same canvas and encoded by the base MLLM vision encoder.
- **Two-stage latent alignment.** Stage I grounds visual latent reasoning with auxiliary visual targets; Stage II aligns the latent channel to unified text-vision canvas targets.
- **Compact inference.** UniVLR reasons through a small latent token budget and decodes only the final answer, avoiding verbose intermediate text CoT at evaluation time.
- **Qwen2.5-VL backbone.** The released implementation builds on Qwen2.5-VL and freezes the vision tower and patch merger by default.
- **VLMEvalKit-ready.** The repository includes a customized VLMEvalKit wrapper for UniVLR decoding and benchmark evaluation.

## 📦 Model Zoo

| Checkpoint | Stage | Recommended Use | Link |
| --- | --- | --- | --- |
| **UniVLR-Stage1-7B** | Visual latent grounding | Warm-up checkpoint, ablations, continued alignment | [ModelScope](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage1-7B) |
| **UniVLR-Stage2-7B** | Text-vision unified alignment | Main checkpoint for evaluation and downstream use | [ModelScope](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage2-7B) |

Download with the ModelScope SDK:

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

snapshot_download(
    "Warrenustc1958/UniVLR-Stage2-7B",
    cache_dir="checkpoints",
)
```

Then point evaluation to the downloaded checkpoint:

```bash
export MODEL_PATH=/path/to/checkpoints/UniVLR-Stage2-7B
```

## 🧩 Method Overview

UniVLR uses special tokens to control the latent reasoning span:

```text
<|univlr_start|> <|univlr|> ... <|univlr|> <|univlr_end|> <|univlr_latent_end|>
```

The training sequence is:

```text
Input multimodal prompt
<|univlr_start|>
K visual latent tokens
<|univlr_end|>
<|univlr_latent_end|>
Final answer
```

At training time, latent targets are extracted from rendered visual canvases with the frozen vision encoder of the base MLLM. UniVLR trains a lightweight projection head to align decoder hidden states with these visual targets using a normalized regression objective together with the standard language modeling loss.

The paper uses `K_train=24` latent targets during training and `K_infer=12` latent steps for the main inference setting. The scripts expose these values through `IMAGE_LATENT_TOKENS` and `UNIVLR_STEPS`.

## 📊 Results

UniVLR is evaluated on perception-centric and visual reasoning benchmarks, including V*, HRBench4K, HRBench8K, and MME-RealWorld-Lite. With Qwen2.5-VL-7B-Instruct as the backbone, UniVLR improves average accuracy over representative visual latent reasoning baselines while using substantially fewer generated reasoning tokens.

| Model | V* | HRBench4K | HRBench8K | MME-RealWorld-Lite |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 77.4 | 69.0 | 66.0 | 46.2 |
| Monet | 79.1 | 71.9 | 63.5 | 46.9 |
| SkiLa | 80.1 | 70.3 | 62.9 | 45.6 |
| CoVT | 78.0 | 71.9 | 69.7 | 48.2 |
| **UniVLR** | **82.7** | **73.3** | **68.8** | **50.7** |

During inference, UniVLR uses 12 latent reasoning tokens and does not generate intermediate text CoT, while interleaved visual latent reasoning baselines typically generate hundreds of reasoning tokens per instance.

## 🗂️ Repository Structure

```text
UniVLR/
+-- canvas/                         # Canvas rendering and dataset construction utilities
+-- scripts/                        # Stage-I and Stage-II SFT entry scripts
+-- src/
|   +-- dataset/                    # UniVLR training data pipeline
|   +-- model/                      # Qwen UniVLR model wrapper and latent heads
|   +-- train/                      # Training entry and forward monkey patches
|   +-- trainer/                    # UniVLR trainer
+-- VLMEvalKit/
|   +-- config/                     # Evaluation config templates
|   +-- univlr_eval/                # UniVLR evaluation scripts
|   +-- vlmeval/                    # Customized VLMEvalKit package
+-- requirements.txt
```

Key files:

- `src/train/train_univlr_stage1.py`: main SFT training entry.
- `src/model/qwen_univlr_model.py`: Qwen2.5-VL UniVLR model implementation.
- `src/model/univlr_heads.py`: latent reasoning projection heads.
- `src/constants.py`: UniVLR special tokens.
- `scripts/univlr_stage1_sft.sh`: Stage-I visual latent grounding.
- `scripts/univlr_stage2_sft.sh`: Stage-II text-vision unified alignment.
- `VLMEvalKit/univlr_eval/eval_univlr.sh`: UniVLR evaluation entry.

## ⚙️ Installation

Create a Python environment and install dependencies:

```bash
git clone https://github.com/Warrenustc1958/UniVLR.git
cd UniVLR

conda create -n univlr python=3.12 -y
conda activate univlr
pip install -r requirements.txt
```

The environment used in our experiments includes PyTorch 2.6, Transformers 4.54, DeepSpeed 0.16, Flash Attention 2, and Qwen-VL utilities. The pinned `requirements.txt` contains the full environment snapshot. If your CUDA or PyTorch version differs, install the matching Flash Attention wheel manually.

For VLMEvalKit evaluation:

```bash
cd UniVLR/VLMEvalKit
pip install -r requirements.txt
```

## 🧱 Data Preparation

UniVLR expects Monet/Zebra-style JSON or JSONL manifests with image paths and latent target paths. The release does not hard-code local data paths. Set paths from the shell:

```bash
export UNIVLR_DATA_ROOT=/path/to/data
export MONET_ROOT=/path/to/Monet-SFT-125K
```

Canvas and dataset construction utilities are under `canvas/`:

```bash
python canvas/build_zebra_cot_step1_dataset.py \
  --output-root "$UNIVLR_DATA_ROOT/Zebra_CoT_step1"

python canvas/build_zebra_cot_step1_vertical_ablation_dataset.py \
  --input-manifest "$UNIVLR_DATA_ROOT/Zebra_CoT_step1/qwen2_5_vl_latent_targets_24token_2dpool/train_offline_k24.json" \
  --output-root "$UNIVLR_DATA_ROOT/Zebra_CoT_step1_vertical_ablation"
```

The paper uses a two-stage curriculum:

- **Stage I:** full Visual-CoT subset as the latent warm-up corpus.
- **Stage II:** filtered Zebra-CoT subsets mixed with sampled Visual-CoT data at a 7:3 ratio.

Offline latent target manifests are expected in folders such as:

```text
<subset>/qwen2_5_vl_latent_targets_24token_2dpool/train_offline_k24.json
```

## 🚀 Training

### Stage I: Visual Latent Grounding

```bash
cd UniVLR

export MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
export MONET_ROOT=/path/to/Monet-SFT-125K
export SUBSETS=Visual_CoT
export IMAGE_LATENT_TOKENS=24
export UNIVLR_ALIGN_LAYER=14
export UNIVLR_HEAD=True
export UNIVLR_HEAD_TYPE=simple
export GLOBAL_BATCH_SIZE=64
export BATCH_PER_DEVICE=1
export OUTPUT_DIR=checkpoints/univlr_stage1

bash scripts/univlr_stage1_sft.sh
```

Useful variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | Base MLLM checkpoint |
| `MONET_ROOT` | `data/Monet-SFT-125K` | Training data root |
| `IMAGE_LATENT_TOKENS` | `24` | Training latent target budget |
| `UNIVLR_TARGET_RESAMPLE_MODE` | `pool_avg` | Target feature compression method |
| `UNIVLR_ALIGN_LAYER` | `14` | Decoder hidden layer used for latent alignment |
| `UNIVLR_HEAD_TYPE` | `simple` | Projection head type |
| `LAMBDA_UNIVLR` | `0.1` | Latent alignment loss weight |

### Stage II: Text-Vision Unified Alignment

```bash
cd UniVLR

export MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
export MONET_ROOT=/path/to/Monet-SFT-125K
export A1_BASE_CHECKPOINT=/path/to/stage1/checkpoint
export ZEBRA_STEP1_DIR="$MONET_ROOT/Zebra_CoT_step1_vertical_ablation"
export VISUAL_COT_DIR="$MONET_ROOT/Visual_CoT"
export ZEBRA_RATIO_NUM=7
export VISUAL_RATIO_NUM=3
export OUTPUT_DIR=checkpoints/univlr_stage2

bash scripts/univlr_stage2_sft.sh
```

The Stage-II script builds a mixed manifest if `DATA_PATH` is not provided. To only build the mixed dataset without launching training:

```bash
BUILD_DATASET_ONLY=True bash scripts/univlr_stage2_sft.sh
```

## 🔍 Evaluation

Edit `VLMEvalKit/config/univlr_stage1_config.json` or provide `MODEL_PATH` directly from the shell. For the released Stage-II checkpoint:

```bash
cd UniVLR/VLMEvalKit

export LMUData=/path/to/VLMEvalKit/data
export MODEL_PATH=/path/to/UniVLR-Stage2-7B
export MODEL_ALIAS=UniVLR-Stage2-7B
export DECODING_STRATEGY=univlr
export UNIVLR_STEPS=12

bash univlr_eval/eval_univlr.sh
```

The default evaluation config includes V*, HRBench4K, HRBench8K, and MME-RealWorld-Lite:

```text
VLMEvalKit/config/univlr_stage1_config.json
```

For a dry run that only prints the generated command and effective config:

```bash
DRY_RUN=1 bash univlr_eval/eval_univlr.sh
```

For API-based judging in VLMEvalKit, configure your own credentials before running evaluation. This repository does not include private API keys.

## 🧠 Notes on Inference

During UniVLR inference, the model enters latent mode after `<|univlr_start|>`, recursively feeds predicted continuous latent embeddings for `UNIVLR_STEPS`, then exits latent mode and decodes the final answer. The evaluation wrapper can strip internal latent markers with:

```bash
export CLEAN_UNIVLR_OUTPUT=true
```

This keeps benchmark outputs focused on the natural-language answer.

## 🙏 Acknowledgements

We thank the open-source projects and communities that made this work possible, including [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [LVR](https://github.com/VincentLeebang/lvr/), and [Monet](https://github.com/NOVAglow646/Monet).

Please follow the licenses and terms of the corresponding upstream models, datasets, and evaluation tools.

## 📚 Citation

If you find UniVLR useful, please consider citing our paper:

```bibtex
@article{jiang2026univlr,
  title={UniVLR: Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs},
  author={Jiang, Houcheng and Fu, Jiajun and Fang, Junfeng and Gao, Chen and Wang, Xiang and He, Xiangnan and Li, Yong},
  journal={arXiv preprint arXiv:2605.11856},
  year={2026}
}
```
