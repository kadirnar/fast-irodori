# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)
[![VoiceDesign](https://img.shields.io/badge/VoiceDesign-HuggingFace-orange)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Flow Matching-based Japanese Text-to-Speech model using a Rectified Flow Diffusion Transformer (RF-DiT) over [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents. Based on [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/).

## Performance

| GPU | TTFT | Generation Time | TTFA | Audio Length | RTF |
|---|---|---|---|---|---|
| NVIDIA H100 PCIe | 0.5 ms | 1,648.8 ms | 1,726 ms | ~5.6 s | 0.31x |

## Installation

```bash
git clone https://github.com/kadirnar/fast-irodori.git
cd fast-irodori
uv sync
```

## Quick Start

### Inference

```bash
# With reference audio (voice cloning)
uv run irodori-infer \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav output.wav

# Without reference audio
uv run irodori-infer \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav output.wav

# VoiceDesign (caption-conditioned)
uv run irodori-infer \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "今日はいい天気ですね。" \
  --caption "落ち着いた女性の声で、やわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav output.wav
```

### Web UI

```bash
uv run irodori-app                # Base model (port 7860)
uv run irodori-app-voicedesign    # VoiceDesign (port 7861)
```

## Training

### 1. Prepare Data

```bash
uv run irodori-prepare-manifest \
  --dataset myorg/my_dataset \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

### 2. Train

```bash
# Single GPU
uv run irodori-train \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl

# Multi-GPU DDP
uv run torchrun --nproc_per_node 4 -m irodori_tts.training.train \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl

# LoRA fine-tuning
uv run irodori-train \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --init-checkpoint path/to/model.safetensors
```

### 3. Convert Checkpoint

```bash
uv run irodori-convert-checkpoint outputs/checkpoint_final.pt
```

## Project Structure

```
irodori_tts/
  config.py                    # Model / Train / Sampling configs
  model/                       # DiT architecture, RF sampling, LoRA
  text/                        # Tokenizer, text normalization
  audio/                       # DACVAE codec
  training/                    # Train loop, dataset, optimizer
  inference/                   # CLI inference, runtime engine
  app/                         # Gradio web UIs
  tools/                       # Data prep, checkpoint conversion
configs/                       # YAML training presets
```

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: See [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) and [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)

## Acknowledgments

- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) — Original codebase
- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE

## Citation

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Aratako/Irodori-TTS}}
}
```
