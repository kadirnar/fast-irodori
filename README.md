# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)
[![VoiceDesign](https://img.shields.io/badge/VoiceDesign-HuggingFace-orange)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Flow Matching-based Japanese Text-to-Speech model using a Rectified Flow Diffusion Transformer (RF-DiT) over [DACVAE](https://github.com/kadirnar/fast-dacvae) continuous latents. Based on [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/).

## Performance

Benchmarked on NVIDIA H100 PCIe, 40 steps, ~18s audio, `--no-ref` mode:

| # | Optimization | TTFT | TTFA | End-to-End | Generation Time | Speedup |
|---|---|---|---|---|---|---|
| 0 | **Baseline** (fp32, 40 steps) | 0.7 ms | 1,076 ms | 1,076 ms | 1,074 ms | x1.0 |
| 1 | + fast-dacvae (conv2d, poly snake, compile decode) | 0.7 ms | 1,056 ms | 1,056 ms | 1,054 ms | x1.0 |
| 2 | + block cache F1 (cache-dit, skip 11/12 blocks) | 0.6 ms | 224 ms | 224 ms | 223 ms | x4.8 |
| 3 | + bf16 model precision | 0.5 ms | 153 ms | 153 ms | 152 ms | x7.0 |
| 4 | + torch.compile (per-block) | 0.4 ms | 141 ms | 141 ms | 140 ms | x7.6 |
| 5 | + velocity cache (skip=2, 1 forward per 3 steps) | 0.4 ms | 102 ms | 102 ms | 101 ms | x10.5 |
| 6 | + precomputed cond embeddings + in-place Euler | 0.4 ms | 100 ms | 100 ms | 99 ms | x10.8 |
| 7 | + bf16 codec decode | 0.4 ms | **86 ms** | **86 ms** | **85 ms** | **x12.5** |

> **TTFT** = Time To First Token (tokenize + prepare reference)
> **TTFA** = Time To First Audio (total until audio is decoded)
> **Generation Time** = Sampling + Decode (excludes tokenization overhead)
> Attention uses FlashAttention-3 (cuDNN backend) automatically via PyTorch SDPA.

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

### Optimized Inference (x12.5 faster)

```bash
uv run irodori-infer \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --model-precision bf16 \
  --codec-precision bf16 \
  --block-cache \
  --optimize-codec \
  --output-wav output.wav
```

### SGLang-Compatible Server

```python
from irodori_tts.serving.sglang_adapter import launch_server

launch_server(
    model_path="Aratako/Irodori-TTS-500M-v2",
    port=8000,
    optimize=True,
)
```

```bash
# Generate audio via OpenAI-compatible API
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "今日はいい天気ですね。", "seconds": 20}' \
  --output output.wav

# Get JSON timings
curl -X POST http://localhost:8000/v1/audio/speech/json \
  -H "Content-Type: application/json" \
  -d '{"input": "今日はいい天気ですね。", "seconds": 20}'
```

### SGLang Python API (Offline)

```python
from irodori_tts.serving.sglang_adapter import IrodoriTTSGenerator

gen = IrodoriTTSGenerator.from_pretrained(
    model_path="Aratako/Irodori-TTS-500M-v2",
    optimize=True,
)

result = gen.generate(text="今日はいい天気ですね。", seconds=20.0)
result.save("output.wav")

print(result.timings)
# {'ttft_ms': 0.4, 'ttfa_ms': 86.0, 'end_to_end_ms': 86.1,
#  'generation_time_ms': 85.0, 'sampling_ms': 62.0, 'decode_ms': 23.0,
#  'audio_duration_s': 18.2}

gen.shutdown()
```

### Web UI

```bash
uv run irodori-app                # Base model (port 7860)
uv run irodori-app-voicedesign    # VoiceDesign (port 7861)
```

## Optimizations

| Optimization | Source | Effect |
|---|---|---|
| **Block Cache** (cache-dit) | [cache-dit](https://github.com/vipshop/cache-dit) | Skip 11/12 DiT blocks per step via residual caching |
| **Velocity Cache** | Custom | Reuse previous velocity, skip entire forward (1 per 3 steps) |
| **fast-dacvae** | [fast-dacvae](https://github.com/kadirnar/fast-dacvae) | Conv1d→Conv2d, polynomial Snake, weight-norm strip, torch.compile decode |
| **torch.compile** | PyTorch | Per-block compilation for kernel fusion |
| **bf16 precision** | PyTorch | Half-precision model + codec |
| **Precomputed cond embeddings** | Custom | Batch all timestep embeddings before Euler loop |
| **FlashAttention-3** | PyTorch SDPA (cuDNN) | Automatic via `F.scaled_dot_product_attention` on H100 |

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
  audio/                       # DACVAE codec (fast-dacvae optimized)
  training/                    # Train loop, dataset, optimizer
  inference/                   # CLI inference, runtime engine
  serving/                     # SGLang-compatible server + offline API
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
- [fast-dacvae](https://github.com/kadirnar/fast-dacvae) — Optimized DACVAE codec
- [cache-dit](https://github.com/vipshop/cache-dit) — Block caching algorithm
- [SGLang](https://github.com/sgl-project/sglang) — Serving framework

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
