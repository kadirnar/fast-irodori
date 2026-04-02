#!/usr/bin/env python3
"""Baseline benchmark for Irodori-TTS inference: TTFT and TTFA measurements."""
from __future__ import annotations

import time

import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference.runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
)

HF_REPO = "Aratako/Irodori-TTS-500M-v2"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
TEST_TEXT = "こんにちは、これはテストです。音声合成の速度を測定しています。"
NUM_WARMUP = 1
NUM_RUNS = 3
SEED = 42


def run_benchmark(
    optimize_codec: bool = False,
    model_precision: str = "fp32",
    codec_precision: str = "fp32",
    compile_blocks: bool = False,
    block_cache: bool = False,
    block_cache_fn: int = 1,
    block_cache_velocity_skip: int = 0,
    num_steps: int = 40,
    seconds: float = 30.0,
    num_warmup: int = NUM_WARMUP,
    num_runs: int = NUM_RUNS,
) -> None:
    device = default_runtime_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_path = hf_hub_download(repo_id=HF_REPO, filename="model.safetensors")
    print(f"Checkpoint: {checkpoint_path}")

    print("\n=== Model Loading ===")
    t_load_start = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=device,
            codec_repo=CODEC_REPO,
            model_precision=model_precision,
            codec_device=device,
            codec_precision=codec_precision,
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            enable_watermark=False,
            compile_model=False,
            compile_dynamic=False,
            compile_blocks=compile_blocks,
            optimize_codec=optimize_codec,
        )
    )
    model_load_time = time.perf_counter() - t_load_start
    print(f"Model load time: {model_load_time:.3f} s")

    cfg_text, cfg_caption, cfg_speaker, _ = resolve_cfg_scales(
        cfg_guidance_mode="independent",
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0,
        cfg_scale_speaker=5.0,
        cfg_scale=None,
        use_caption_condition=False,
        use_speaker_condition=runtime.model_cfg.use_speaker_condition,
    )

    def make_request(seed: int) -> SamplingRequest:
        return SamplingRequest(
            text=TEST_TEXT,
            caption=None,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=seconds,
            max_ref_seconds=30.0,
            max_text_len=None,
            max_caption_len=None,
            num_steps=num_steps,
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_caption,
            cfg_scale_speaker=cfg_speaker,
            cfg_guidance_mode="independent",
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            seed=seed,
            trim_tail=True,
            tail_window_size=20,
            tail_std_threshold=0.05,
            tail_mean_threshold=0.1,
            block_cache_enabled=block_cache,
            block_cache_fn=block_cache_fn,
            block_cache_bn=0,
            block_cache_threshold=1.0,
            block_cache_warmup=1,
            block_cache_taylorseer=False,
            block_cache_schedule=None,
            block_cache_velocity_skip=block_cache_velocity_skip,
        )

    print(f"\n=== Warm-up ({num_warmup} run) ===")
    for i in range(num_warmup):
        result = runtime.synthesize(make_request(seed=SEED + i), log_fn=None)
        print(f"  warmup[{i}] total_to_decode: {result.total_to_decode:.3f} s")

    print(f"\n=== Benchmark ({num_runs} runs) ===")
    all_timings: list[dict[str, float]] = []
    all_totals: list[float] = []

    for i in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        result = runtime.synthesize(make_request(seed=SEED + 100 + i), log_fn=None)
        if device == "cuda":
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - t_start

        timings = dict(result.stage_timings)
        all_timings.append(timings)
        all_totals.append(result.total_to_decode)

        audio_duration = result.audio.shape[-1] / result.sample_rate
        rtf = result.total_to_decode / audio_duration if audio_duration > 0 else float("inf")

        print(f"\n  Run {i + 1}:")
        for name, sec in result.stage_timings:
            print(f"    {name}: {sec * 1000.0:.1f} ms")
        print(f"    total_to_decode: {result.total_to_decode:.3f} s")
        print(f"    wall_time: {wall_time:.3f} s")
        print(f"    audio_duration: {audio_duration:.2f} s")
        print(f"    RTF (real-time factor): {rtf:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY (averaged over benchmark runs)")
    print("=" * 60)
    print(f"  Model load (one-time): {model_load_time:.3f} s")

    stage_names = list(all_timings[0].keys())
    for name in stage_names:
        vals = [t[name] for t in all_timings if name in t]
        avg_ms = sum(vals) / len(vals) * 1000.0
        print(f"  {name}: {avg_ms:.1f} ms (avg)")

    avg_total = sum(all_totals) / len(all_totals)
    print(f"  total_to_decode (avg): {avg_total:.3f} s")

    ttft_vals = [t.get("tokenize_text", 0) + t.get("prepare_reference", 0) for t in all_timings]
    avg_ttft = sum(ttft_vals) / len(ttft_vals) * 1000.0
    sample_vals = [t.get("sample_rf", 0) for t in all_timings]
    avg_sample = sum(sample_vals) / len(sample_vals) * 1000.0
    decode_vals = [t.get("decode_latent", 0) for t in all_timings]
    avg_decode = sum(decode_vals) / len(decode_vals) * 1000.0
    avg_ttfa = avg_total * 1000.0

    print(f"\n  TTFT (Time To First Token / sampling start): {avg_ttft:.1f} ms")
    print(f"  Sampling (RF diffusion, {num_steps} steps): {avg_sample:.1f} ms")
    print(f"  Decode (DACVAE latent → audio): {avg_decode:.1f} ms")
    print(f"  TTFA (Time To First Audio): {avg_ttfa:.1f} ms")
    print(f"  Generation Time (sampling + decode): {avg_sample + avg_decode:.1f} ms")


if __name__ == "__main__":
    run_benchmark()
