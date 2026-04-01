# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Irodori-TTS is a Flow Matching-based Japanese Text-to-Speech system using a Rectified Flow Diffusion Transformer (RFDiT). It converts text to DACVAE latent representations via an Euler ODE sampler with classifier-free guidance (CFG). Two modes: speaker-conditioned (base model with reference audio) and caption-conditioned (VoiceDesign, text description of desired voice).

## Commands

### Setup
```bash
uv sync                    # Install all dependencies
```

### Training
```bash
# Single GPU
uv run irodori-train --config configs/train_500m_v2.yaml --manifest data/manifest.jsonl

# Multi-GPU DDP
uv run torchrun -m irodori_tts.training.train --config configs/train_500m_v2.yaml --manifest data/manifest.jsonl

# LoRA fine-tuning
uv run irodori-train --config configs/train_500m_v2_lora.yaml --manifest data/manifest.jsonl
```

### Inference
```bash
# CLI inference (local checkpoint)
uv run irodori-infer --checkpoint path/to/model.safetensors --text "テキスト"

# CLI inference (HuggingFace Hub)
uv run irodori-infer --hf-checkpoint repo/name --text "テキスト"
```

### Web UI
```bash
uv run irodori-app                # Base model UI (port 7860)
uv run irodori-app-voicedesign    # VoiceDesign UI
```

### Data Preparation
```bash
uv run irodori-prepare-manifest --dataset <name> --output-dir data/
```

### Checkpoint Conversion
```bash
uv run irodori-convert-checkpoint --checkpoint path/to/model.pt --output path/to/model.safetensors
```

### Linting
```bash
uv run ruff check .        # Lint (line-length=100, target py310)
```

## Architecture

### Core Pipeline
```
Text → TextEncoder (LLM tokenizer + RoPE blocks) ─┐
Reference Audio → DACVAE → ReferenceLatentEncoder ─┤→ DiffusionBlocks (JointAttention + AdaLN) → DACVAE decode → WAV
Caption → CaptionEncoder (optional) ───────────────┘      ↑ timestep embedding
```

### Key Modules (`irodori_tts/`)

#### Config
- **config.py** — `ModelConfig`, `TrainConfig`, `SamplingConfig` dataclasses. YAML configs merged with CLI overrides via `merge_dataclass_overrides()`.

#### Model (`model/`)
- **dit.py** — `TextToLatentRFDiT`: Main model with text/speaker/caption encoders feeding into diffusion transformer blocks via JointAttention (concatenated K/V from all conditioning sources). Uses LowRankAdaLN for timestep conditioning and SwiGLU MLPs.
- **rf.py** — Rectified Flow sampling: Euler integration, CFG with three guidance modes (`independent`/`joint`/`alternating`), logit-normal and stratified timestep sampling, truncation, temporal score rescaling.
- **lora.py** — PEFT LoRA with 11 target presets (e.g., `diffusion_attn_mlp`, `all_attn_mlp`). Saves adapter directories with trainer state for resume.

#### Text (`text/`)
- **tokenizer.py** — Byte-level and pretrained text tokenizers.
- **normalization.py** — Japanese text normalization utilities.

#### Audio (`audio/`)
- **codec.py** — `DACVAECodec`: Wraps DACVAE for encode/decode with latent patchify/unpatchify operations.

#### Training (`training/`)
- **train.py** — Training entry point with DDP support, checkpoint saving, validation loop.
- **dataset.py** — `LatentTextDataset` reads JSONL manifests with speaker-aware reference sampling. `TTSCollator` handles tokenization, padding, patching, and mask creation.
- **optim.py** — Muon optimizer for 2D+ weight matrices + AdamW for aux params. Cosine and WSD (warmup-stable-decay) LR schedulers.
- **progress.py** — Training progress tracking.

#### Inference (`inference/`)
- **infer.py** — CLI inference with local or HuggingFace Hub checkpoints.
- **runtime.py** — `InferenceRuntime`: Thread-safe cached inference server with lazy model loading and request queuing.

#### App (`app/`)
- **gradio_app.py** — Gradio web UI for base model speaker-conditioned inference.
- **gradio_app_voicedesign.py** — Gradio web UI for VoiceDesign caption-conditioned inference.

#### Tools (`tools/`)
- **prepare_manifest.py** — Data preparation from HuggingFace datasets to JSONL manifest.
- **convert_checkpoint_to_safetensors.py** — Convert training checkpoints to safetensors format.

### Configuration Cascade
YAML file → CLI args → dataclass defaults. Config files in `configs/` define training presets. CLI args override any YAML value.

### Training Specifics
- Loss: echo-style masked MSE normalized by valid-token ratio
- Condition dropout: independent text/speaker/caption dropout for CFG training
- Checkpoints: periodic (`checkpoint_NNNNNNN.pt`), best validation, and final
- LoRA checkpoints saved as directories with `adapter_config.json` + trainer state

### Manifest Format (JSONL)
```json
{"text": "...", "latent_path": "path/to/latent.pt", "speaker_id": "spk01", "caption": "..."}
```
