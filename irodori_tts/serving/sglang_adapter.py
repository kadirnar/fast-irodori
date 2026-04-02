"""SGLang-compatible TTS generator using Irodori-TTS optimized runtime.

Implements the same interface pattern as SGLang's DiffGenerator for
offline/programmatic usage, and provides an OpenAI-compatible HTTP
server for online serving.

Usage (offline):
    from irodori_tts.serving.sglang_adapter import IrodoriTTSGenerator

    gen = IrodoriTTSGenerator.from_pretrained(
        model_path="Aratako/Irodori-TTS-500M-v2",
        optimize=True,
    )
    result = gen.generate(text="こんにちは", seconds=20.0)
    result.save("output.wav")
    gen.shutdown()

Usage (server):
    uv run python -m irodori_tts.serving.sglang_adapter \
        --model-path Aratako/Irodori-TTS-500M-v2 --optimize --port 8000
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference.runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
)


@dataclass
class TTSSamplingParams:
    """SGLang-style sampling params for TTS."""

    num_steps: int = 40
    seconds: float = 20.0
    caption: str | None = None
    seed: int | None = None
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    # Optimization params
    block_cache: bool = True
    block_cache_fn: int = 1
    block_cache_velocity_skip: int = 2


@dataclass
class TTSGenerationResult:
    """SGLang-style generation result."""

    audio: torch.Tensor
    sample_rate: int
    seed: int
    timings: dict[str, float]

    def save(self, path: str) -> str:
        import soundfile as sf

        audio_np = self.audio.squeeze().cpu().numpy()
        sf.write(path, audio_np, self.sample_rate, format="WAV")
        return path

    def to_wav_bytes(self) -> bytes:
        import soundfile as sf

        buf = BytesIO()
        audio_np = self.audio.squeeze().cpu().numpy()
        sf.write(buf, audio_np, self.sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()


class IrodoriTTSGenerator:
    """SGLang-compatible offline generator for Irodori-TTS.

    Mirrors SGLang's DiffGenerator API pattern while using the
    fully optimized Irodori-TTS inference pipeline (block-cache,
    velocity-cache, torch.compile, fast-dacvae, bf16).
    """

    def __init__(
        self,
        runtime: InferenceRuntime,
        cfg_scales: tuple[float, float, float],
        optimize: bool = True,
    ):
        self.runtime = runtime
        self.cfg_scales = cfg_scales
        self.optimize = optimize

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "Aratako/Irodori-TTS-500M-v2",
        device: str | None = None,
        model_precision: str = "bf16",
        codec_precision: str = "bf16",
        optimize: bool = True,
        codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        warmup: bool = True,
        **kwargs: Any,
    ) -> IrodoriTTSGenerator:
        """Create generator from a pretrained model (SGLang DiffGenerator pattern).

        Args:
            model_path: HuggingFace repo ID or local checkpoint path.
            device: Inference device (default: auto-detect).
            model_precision: Model precision (fp32, bf16).
            codec_precision: Codec precision (fp32, bf16).
            optimize: Enable all optimizations.
            codec_repo: DACVAE codec repo.
            warmup: Run warmup inference on init.
        """
        if device is None:
            device = default_runtime_device()

        # Resolve checkpoint
        ckpt_path = model_path
        if not model_path.endswith((".pt", ".safetensors")):
            ckpt_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")

        runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=ckpt_path,
                model_device=device,
                codec_repo=codec_repo,
                model_precision=model_precision,
                codec_device=device,
                codec_precision=codec_precision,
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
                enable_watermark=False,
                compile_model=False,
                compile_dynamic=False,
                compile_blocks=optimize,
                optimize_codec=optimize,
            )
        )

        cfg_scales = resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
            use_caption_condition=False,
            use_speaker_condition=runtime.model_cfg.use_speaker_condition,
        )[:3]

        gen = cls(runtime=runtime, cfg_scales=cfg_scales, optimize=optimize)

        if warmup:
            gen._warmup()

        return gen

    def _warmup(self, seconds: float = 20.0) -> None:
        """Warm up torch.compile caches."""
        for i in range(3 if self.optimize else 1):
            self.generate(text="テスト", seconds=seconds, seed=i)

    def generate(
        self,
        text: str,
        seconds: float = 20.0,
        caption: str | None = None,
        seed: int | None = None,
        sampling_params: TTSSamplingParams | None = None,
        **kwargs: Any,
    ) -> TTSGenerationResult:
        """Generate speech from text (SGLang DiffGenerator.generate pattern).

        Args:
            text: Input text to synthesize.
            seconds: Target audio duration.
            caption: Optional voice description (VoiceDesign models).
            seed: Random seed (auto-generated if None).
            sampling_params: Full sampling configuration.

        Returns:
            TTSGenerationResult with audio tensor, timings, and save methods.
        """
        if seed is None:
            seed = secrets.randbelow(2**31)

        sp = sampling_params or TTSSamplingParams()
        cfg_t, cfg_c, cfg_s = self.cfg_scales

        req = SamplingRequest(
            text=text,
            caption=caption or sp.caption,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=seconds or sp.seconds,
            max_ref_seconds=30.0,
            max_text_len=None,
            max_caption_len=None,
            num_steps=sp.num_steps,
            cfg_scale_text=sp.cfg_scale_text or cfg_t,
            cfg_scale_caption=cfg_c,
            cfg_scale_speaker=sp.cfg_scale_speaker or cfg_s,
            cfg_guidance_mode=sp.cfg_guidance_mode,
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
            block_cache_enabled=sp.block_cache if self.optimize else False,
            block_cache_fn=sp.block_cache_fn,
            block_cache_bn=0,
            block_cache_threshold=1.0,
            block_cache_warmup=1,
            block_cache_taylorseer=False,
            block_cache_schedule=None,
            block_cache_velocity_skip=sp.block_cache_velocity_skip if self.optimize else 0,
        )

        import time

        t0 = time.perf_counter()
        result = self.runtime.synthesize(req, log_fn=None)
        end_to_end = (time.perf_counter() - t0) * 1000
        t = dict(result.stage_timings)

        sampling_ms = t.get("sample_rf", 0) * 1000
        decode_ms = t.get("decode_latent", 0) * 1000

        return TTSGenerationResult(
            audio=result.audio,
            sample_rate=result.sample_rate,
            seed=result.used_seed,
            timings={
                "ttft_ms": round(
                    (t.get("tokenize_text", 0) + t.get("prepare_reference", 0)) * 1000, 2
                ),
                "ttfa_ms": round(result.total_to_decode * 1000, 2),
                "end_to_end_ms": round(end_to_end, 2),
                "generation_time_ms": round(sampling_ms + decode_ms, 2),
                "sampling_ms": round(sampling_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "audio_duration_s": round(result.audio.shape[-1] / result.sample_rate, 2),
            },
        )

    def shutdown(self) -> None:
        """Release resources (SGLang DiffGenerator.shutdown pattern)."""
        del self.runtime
        torch.cuda.empty_cache()


# ============ HTTP Server (SGLang-compatible OpenAI API) ============

def create_app(gen: IrodoriTTSGenerator) -> Any:
    """Create FastAPI app with SGLang-compatible endpoints."""
    from fastapi import FastAPI
    from fastapi.responses import Response
    from pydantic import BaseModel

    app = FastAPI(title="Irodori-TTS (SGLang)")

    class SpeechRequest(BaseModel):
        model: str = "irodori-tts"
        input: str
        voice: str = "default"
        response_format: str = "wav"
        speed: float = 1.0
        num_steps: int = 40
        caption: str | None = None
        seed: int | None = None
        seconds: float = 20.0

    @app.post("/v1/audio/speech")
    async def create_speech(request: SpeechRequest):
        result = gen.generate(
            text=request.input,
            seconds=request.seconds,
            caption=request.caption,
            seed=request.seed,
            sampling_params=TTSSamplingParams(num_steps=request.num_steps),
        )
        return Response(
            content=result.to_wav_bytes(),
            media_type="audio/wav",
            headers={
                "X-Request-Id": f"tts-{result.seed}",
                "X-TTFT-Ms": f"{result.timings['ttft_ms']:.1f}",
                "X-TTFA-Ms": f"{result.timings['ttfa_ms']:.1f}",
                "X-End-To-End-Ms": f"{result.timings['end_to_end_ms']:.1f}",
                "X-Generation-Time-Ms": f"{result.timings['generation_time_ms']:.1f}",
            },
        )

    @app.post("/v1/audio/speech/json")
    async def create_speech_json(request: SpeechRequest):
        result = gen.generate(
            text=request.input,
            seconds=request.seconds,
            caption=request.caption,
            seed=request.seed,
            sampling_params=TTSSamplingParams(num_steps=request.num_steps),
        )
        return {"id": f"tts-{result.seed}", "model": "irodori-tts", "timings": result.timings}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [
                {"id": "irodori-tts", "object": "model", "owned_by": "irodori"},
            ]
        }

    return app


def launch_server(
    model_path: str = "Aratako/Irodori-TTS-500M-v2",
    host: str = "0.0.0.0",
    port: int = 8000,
    model_precision: str = "bf16",
    codec_precision: str = "bf16",
    optimize: bool = True,
) -> None:
    """Launch the SGLang-compatible TTS server.

    Args:
        model_path: HuggingFace repo ID or local checkpoint path.
        host: Server bind address.
        port: Server port.
        model_precision: Model precision (fp32, bf16).
        codec_precision: Codec precision (fp32, bf16, fp16).
        optimize: Enable all optimizations.
    """
    import uvicorn

    gen = IrodoriTTSGenerator.from_pretrained(
        model_path=model_path,
        model_precision=model_precision,
        codec_precision=codec_precision,
        optimize=optimize,
    )
    app = create_app(gen)
    print(f"[sglang] Server ready at http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
