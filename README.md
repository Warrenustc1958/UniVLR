# UniVLR

Official anonymous implementation of **UniVLR: Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs**.

UniVLR is a visual latent reasoning framework for multimodal large language models. Instead of interleaving explicit text chain-of-thought with visual latent tokens, UniVLR renders textual reasoning traces and auxiliary visual evidence into a shared visual canvas, compresses that canvas into compact visual latent tokens, and performs inference through the latent channel before decoding only the final answer.

## Highlights

- **Unified visual workspace.** Textual reasoning steps and auxiliary visual evidence are represented on the same rendered canvas, then encoded by the base MLLM vision encoder.
- **Two-stage latent alignment.** Stage I grounds the model in visual latent reasoning with auxiliary visual targets. Stage II aligns the latent channel to unified text-vision canvas targets.
- **Compact inference.** During evaluation, UniVLR uses a small latent token budget and does not generate verbose intermediate text reasoning.
- **Qwen2.5-VL backbone.** The released code instantiates UniVLR on top of Qwen2.5-VL and freezes the vision tower and patch merger by default.
- **VLMEvalKit support.** The repository includes a customized VLMEvalKit wrapper for UniVLR decoding and benchmark evaluation.

## Method Overview

UniVLR uses four special tokens to control latent reasoning:

```text
<|univlr_start|> <|univlr|> ... <|univlr|> <|univlr_end|> <|univlr_latent_end|>
```

The training sequence follows:

```text
Input multimodal prompt
<|univlr_start|>
K visual latent tokens
<|univlr_end|>
<|univlr_latent_end|>
Final answer
```

The latent targets are extracted from rendered visual canvases using the frozen vision encoder of the base MLLM. UniVLR then trains a lightweight projection head to align decoder hidden states with these visual targets using a normalized regression objective combined with the standard language modeling loss.

The paper uses `K_train=24` latent targets during training and `K_infer=12` latent steps for the main inference setting. The scripts expose these values through `IMAGE_LATENT_TOKENS` and `UNIVLR_STEPS`.

## Results From The Paper

The paper evaluates UniVLR on perception-centric and visual reasoning benchmarks including V*, HRBench4K, HRBench8K, and MME-RealWorld-Lite. With Qwen2.5-VL-7B-Instruct as the backbone, UniVLR improves average accuracy over representative visual latent reasoning baselines while using substantially fewer generated reasoning tokens.

Reported main comparison:

| Model | V* | HRBench4K | HRBench8K | MME-RealWorld-Lite |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 77.4 | 69.0 | 66.0 | 46.2 |
| Monet | 79.1 | 71.9 | 63.5 | 46.9 |
| SkiLa | 80.1 | 70.3 | 62.9 | 45.6 |
| CoVT | 78.0 | 71.9 | 69.7 | 48.2 |
| UniVLR | 82.7 | 73.3 | 68.8 | 50.7 |

In the paper, UniVLR performs inference with 12 latent reasoning tokens and no generated intermediate text CoT, while interleaved visual latent reasoning baselines typically generate hundreds of reasoning tokens per instance.

## Repository Structure

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

## Installation

Create a Python environment and install dependencies:

```bash
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

## Data Preparation

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

## Training

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

## Evaluation

Edit `VLMEvalKit/config/univlr_stage1_config.json` or provide `MODEL_PATH` directly from the shell:

```bash
cd UniVLR/VLMEvalKit

export LMUData=/path/to/VLMEvalKit/data
export MODEL_PATH=/path/to/univlr/checkpoint
export MODEL_ALIAS=UniVLR
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

For API-based judging in VLMEvalKit, configure your own credentials before running evaluation. The anonymous release does not include private API keys.

## Notes On Inference

During UniVLR inference, the model enters latent mode after `<|univlr_start|>`, recursively feeds predicted continuous latent embeddings for `UNIVLR_STEPS`, then exits latent mode and decodes the final answer. The evaluation wrapper can strip internal latent markers with:

```bash
export CLEAN_UNIVLR_OUTPUT=true
```

This keeps benchmark outputs focused on the natural-language answer.


## Acknowledgements

We thank the open-source projects and communities that made this work possible, including [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [LVR](https://github.com/VincentLeebang/lvr/), and [Monet](https://github.com/NOVAglow646/Monet).

Please follow the licenses and terms of the corresponding upstream models, datasets, and evaluation tools.
