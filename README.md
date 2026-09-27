<div align="center">

# 					UniVLR 🧠👁️

###   Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs

​				[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-purple.svg)](https://neurips.cc/)[![arXiv](https://img.shields.io/badge/arXiv-2605.11856-b31b1b.svg)](https://arxiv.org/abs/2605.11856)[![ModelScope Stage1](https://img.shields.io/badge/ModelScope-UniVLR--Stage--1--7B-orange.svg)](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage1-7B)[![ModelScope Stage2](https://img.shields.io/badge/ModelScope-UniVLR--Stage--2--7B-blue.svg)](https://www.modelscope.cn/models/Warrenustc1958/UniVLR-Stage2-7B)[![Backbone](https://img.shields.io/badge/Backbone-Qwen2.5--VL--7B-yellow.svg)](https://github.com/QwenLM/Qwen2.5-VL)[![Evaluation](https://img.shields.io/badge/Eval-VLMEvalKit-pink.svg)](https://github.com/open-compass/VLMEvalKit)

**Official implementation of [UniVLR: Unifying Text and Vision in Visual Latent Reasoning for Multimodal LLMs](https://arxiv.org/abs/2605.11856) (NeurIPS 2026 Poster).**

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

- **2026.09.26 🎉 Exciting News: Our paper has been accepted to NeurIPS 2026 as a Poster!**
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
