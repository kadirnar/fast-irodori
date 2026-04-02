#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference.runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
    save_wav,
)

FIXED_SECONDS = 30.0


def _print_timings(timings: list[tuple[str, float]], total_to_decode: float) -> None:
    print("[timing] ---- post-model-load to decode ----")
    for name, sec in timings:
        print(f"[timing] {name}: {sec * 1000.0:.1f} ms")
    print(f"[timing] total_to_decode: {total_to_decode:.3f} s")


def _resolve_checkpoint(
    checkpoint: str | None = None,
    hf_checkpoint: str | None = None,
) -> str:
    if checkpoint is not None:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] using local file: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    if hf_checkpoint is None:
        raise ValueError("Either checkpoint or hf_checkpoint must be provided.")

    repo_id = str(hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")

    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(
        f"[checkpoint] downloaded model.safetensors from hf://{repo_id} -> {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


def infer(
    text: str,
    checkpoint: str | None = None,
    hf_checkpoint: str | None = None,
    caption: str | None = None,
    output_wav: str = "output.wav",
    model_device: str | None = None,
    model_precision: str = "fp32",
    codec_device: str | None = None,
    codec_precision: str = "fp32",
    codec_deterministic_encode: bool = True,
    codec_deterministic_decode: bool = True,
    enable_watermark: bool = False,
    max_ref_seconds: float = 30.0,
    ref_normalize_db: float | None = -16.0,
    ref_ensure_max: bool = True,
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
    max_text_len: int | None = None,
    max_caption_len: int | None = None,
    num_steps: int = 40,
    num_candidates: int = 1,
    decode_mode: str = "sequential",
    compile_model: bool = False,
    compile_dynamic: bool = False,
    optimize_codec: bool = False,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_scale: float | None = None,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_min_t: float = 0.9,
    speaker_kv_max_layers: int | None = None,
    seed: int | None = None,
    trim_tail: bool = True,
    tail_window_size: int = 20,
    tail_std_threshold: float = 0.05,
    tail_mean_threshold: float = 0.1,
    block_cache: bool = False,
    block_cache_fn: int = 4,
    block_cache_bn: int = 0,
    block_cache_threshold: float = 0.08,
    block_cache_warmup: int = 2,
    ref_wav: str | None = None,
    ref_latent: str | None = None,
    no_ref: bool = False,
    show_timings: bool = True,
) -> None:
    """Run Irodori-TTS inference.

    Args:
        text: Input text to synthesize.
        checkpoint: Local model checkpoint path (.pt or .safetensors).
        hf_checkpoint: HuggingFace model repo id.
        caption: Optional caption for VoiceDesign checkpoints.
        output_wav: Output WAV file path.
        model_device: Model device (default: auto-detect).
        model_precision: Model precision (fp32, bf16).
        codec_device: Codec device (default: auto-detect).
        codec_precision: Codec precision (fp32, bf16).
        optimize_codec: Enable fast-dacvae optimizations.
        block_cache: Enable cache-dit block caching.
        ref_wav: Reference waveform path for voice cloning.
        ref_latent: Reference latent (.pt) path.
        no_ref: Run without speaker reference.
        show_timings: Print per-stage timings.
    """
    if model_device is None:
        model_device = default_runtime_device()
    if codec_device is None:
        codec_device = default_runtime_device()

    checkpoint_path = _resolve_checkpoint(checkpoint=checkpoint, hf_checkpoint=hf_checkpoint)

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=str(model_device),
            codec_repo=str(codec_repo),
            model_precision=str(model_precision),
            codec_device=str(codec_device),
            codec_precision=str(codec_precision),
            codec_deterministic_encode=bool(codec_deterministic_encode),
            codec_deterministic_decode=bool(codec_deterministic_decode),
            enable_watermark=bool(enable_watermark),
            compile_model=bool(compile_model),
            compile_dynamic=bool(compile_dynamic),
            optimize_codec=bool(optimize_codec),
        )
    )

    if runtime.model_cfg.use_speaker_condition and not (
        no_ref or ref_wav is not None or ref_latent is not None
    ):
        raise ValueError(
            "Speaker-conditioned checkpoints require ref_wav, ref_latent, or no_ref=True."
        )

    resolved_cfg_text, resolved_cfg_caption, resolved_cfg_speaker, scale_messages = (
        resolve_cfg_scales(
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=float(cfg_scale) if cfg_scale is not None else None,
            use_caption_condition=bool(
                runtime.model_cfg.use_caption_condition
                and caption is not None
                and str(caption).strip() != ""
            ),
            use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
        )
    )
    for msg in scale_messages:
        print(msg)

    result = runtime.synthesize(
        SamplingRequest(
            text=str(text),
            caption=None if caption is None else str(caption),
            ref_wav=ref_wav,
            ref_latent=ref_latent,
            no_ref=bool(no_ref),
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=bool(ref_ensure_max),
            num_candidates=int(num_candidates),
            decode_mode=str(decode_mode),
            seconds=FIXED_SECONDS,
            max_ref_seconds=float(max_ref_seconds) if max_ref_seconds is not None else None,
            max_text_len=None if max_text_len is None else int(max_text_len),
            max_caption_len=None if max_caption_len is None else int(max_caption_len),
            num_steps=int(num_steps),
            cfg_scale_text=resolved_cfg_text,
            cfg_scale_caption=resolved_cfg_caption,
            cfg_scale_speaker=resolved_cfg_speaker,
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale=None,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=None if truncation_factor is None else float(truncation_factor),
            rescale_k=None if rescale_k is None else float(rescale_k),
            rescale_sigma=None if rescale_sigma is None else float(rescale_sigma),
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=None if speaker_kv_scale is None else float(speaker_kv_scale),
            speaker_kv_min_t=None if speaker_kv_scale is None else float(speaker_kv_min_t),
            speaker_kv_max_layers=(
                None if speaker_kv_max_layers is None else int(speaker_kv_max_layers)
            ),
            seed=None if seed is None else int(seed),
            trim_tail=bool(trim_tail),
            tail_window_size=int(tail_window_size),
            tail_std_threshold=float(tail_std_threshold),
            tail_mean_threshold=float(tail_mean_threshold),
            block_cache_enabled=bool(block_cache),
            block_cache_fn=int(block_cache_fn),
            block_cache_bn=int(block_cache_bn),
            block_cache_threshold=float(block_cache_threshold),
            block_cache_warmup=int(block_cache_warmup),
        ),
        log_fn=None,
    )

    print(f"[seed] used_seed: {result.used_seed}")
    if int(num_candidates) == 1:
        out_path = save_wav(output_wav, result.audio, result.sample_rate)
        print(f"Saved: {out_path}")
    else:
        base_path = Path(str(output_wav))
        suffix = base_path.suffix if base_path.suffix else ".wav"
        for i, audio in enumerate(result.audios, start=1):
            out_path = base_path.with_name(f"{base_path.stem}_{i:03d}{suffix}")
            saved = save_wav(out_path, audio, result.sample_rate)
            print(f"Saved[{i}]: {saved}")
    if show_timings:
        _print_timings(result.stage_timings, result.total_to_decode)


def main() -> None:
    """CLI entry point — delegates to infer()."""
    import argparse

    parser = argparse.ArgumentParser(description="Inference for Irodori-TTS.")
    ckpt = parser.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--checkpoint", default=None)
    ckpt.add_argument("--hf-checkpoint", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--output-wav", default="output.wav")
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--optimize-codec", action="store_true", default=False)
    parser.add_argument("--block-cache", action="store_true", default=False)
    parser.add_argument("--block-cache-fn", type=int, default=4)
    parser.add_argument("--no-ref", action="store_true")
    parser.add_argument("--ref-wav", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=40)
    args = parser.parse_args()

    infer(
        text=args.text,
        checkpoint=args.checkpoint,
        hf_checkpoint=args.hf_checkpoint,
        caption=args.caption,
        output_wav=args.output_wav,
        model_precision=args.model_precision,
        codec_precision=args.codec_precision,
        optimize_codec=args.optimize_codec,
        block_cache=args.block_cache,
        block_cache_fn=args.block_cache_fn,
        no_ref=args.no_ref,
        ref_wav=args.ref_wav,
        seed=args.seed,
        num_steps=args.num_steps,
    )


if __name__ == "__main__":
    main()
