from __future__ import annotations

from dataclasses import dataclass, field

import torch

from irodori_tts.model.dit import TextToLatentRFDiT, get_timestep_embedding


def _build_steps_mask_ultra(total_steps: int) -> list[int]:
    """Generate an 'ultra' schedule mask: 1=compute, 0=cache."""
    if total_steps <= 8:
        compute_bins, cache_bins = [2, 1, 1], [2, 2]
    elif total_steps <= 16:
        compute_bins, cache_bins = [2, 1, 1, 1, 1], [1, 2, 3, 3]
    elif total_steps <= 28:
        compute_bins, cache_bins = [4, 1, 1, 1, 1], [2, 5, 6, 7]
    else:
        # Scale up for >28 steps
        ratio = total_steps / 28.0
        compute_bins = [max(1, int(x * ratio)) for x in [4, 1, 1, 1, 1]]
        cache_bins = [max(1, int(x * ratio)) for x in [2, 5, 6, 7]]
    mask: list[int] = []
    cb, ab = list(reversed(compute_bins)), list(reversed(cache_bins))
    while len(mask) < total_steps:
        if cb:
            mask.extend([1] * cb.pop())
        if ab:
            mask.extend([0] * ab.pop())
    return mask[:total_steps]


class _TaylorSeer:
    """Order-1 Taylor extrapolation for cached residuals."""

    def __init__(self) -> None:
        self.prev: torch.Tensor | None = None
        self.current: torch.Tensor | None = None
        self.deriv: torch.Tensor | None = None
        self.last_update_step: int = -1
        self.current_step: int = -1

    def update(self, y: torch.Tensor, step: int) -> None:
        if self.current is not None and self.current.shape == y.shape:
            window = max(1, step - self.last_update_step)
            self.deriv = (y - self.current) / window
        self.prev = self.current
        self.current = y
        self.last_update_step = step

    def approximate(self, step: int) -> torch.Tensor | None:
        if self.current is None:
            return None
        if self.deriv is None:
            return self.current
        elapsed = step - self.last_update_step
        return self.current + self.deriv * elapsed

    def reset(self) -> None:
        self.prev = self.current = self.deriv = None
        self.last_update_step = self.current_step = -1


@dataclass
class BlockCacheState:
    """Runtime state for cache-dit style block caching during Euler sampling."""

    fn_blocks: int = 4
    bn_blocks: int = 0
    threshold: float = 0.08
    warmup_steps: int = 2
    use_taylorseer: bool = False
    steps_mask: list[int] | None = None
    velocity_cache_max_skip: int = 0  # 0=disabled, N=skip up to N steps between forwards
    # runtime (reset per inference)
    step: int = 0
    prev_fn_residual: torch.Tensor | None = None
    cached_mn_residual: torch.Tensor | None = None
    cached_steps: list[int] = field(default_factory=list)
    _seer: _TaylorSeer = field(default_factory=_TaylorSeer)

    def reset(self) -> None:
        self.step = 0
        self.prev_fn_residual = None
        self.cached_mn_residual = None
        self.cached_steps = []
        self._seer.reset()


def _make_rng(seed: int, device: torch.device) -> tuple[torch.Generator, torch.device]:
    # MPS generators are not available on some PyTorch builds; use CPU generator as fallback.
    try:
        return torch.Generator(device=device).manual_seed(seed), device
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(seed), torch.device("cpu")


def sample_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(min=t_min, max=t_max)


def sample_stratified_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    """
    Stratified sampling for logit-normal timesteps.

    u ~ stratified U(0, 1), z = mean + std * Phi^{-1}(u), t = sigmoid(z)
    """
    if batch_size <= 0:
        return torch.empty((0,), device=device)
    u = (
        torch.arange(batch_size, device=device, dtype=torch.float32)
        + torch.rand(batch_size, device=device)
    ) / float(batch_size)
    u = u.clamp(1e-6, 1.0 - 1e-6)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0**0.5)
    z = z * std + mean
    t = torch.sigmoid(z)
    # Randomize assignment order so dataset ordering does not correlate with t bins.
    t = t[torch.randperm(batch_size, device=device)]
    return t.clamp(min=t_min, max=t_max)


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Straight line interpolation: x_t = (1-t) x0 + t z.
    return (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise


def rf_velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # For x_t = (1-t) x0 + t z, velocity is d/dt x_t = z - x0.
    return noise - x0


def rf_predict_x0(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x_t = x0 + t * v  =>  x0 = x_t - t * v
    return x_t - t[:, None, None] * v_pred


def temporal_score_rescale(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: float | torch.Tensor,
    rescale_k: float,
    rescale_sigma: float,
) -> torch.Tensor:
    """
    Temporal score rescaling from https://arxiv.org/pdf/2510.01184.
    """
    t_value = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
    if t_value >= 1.0:
        return v_pred
    one_minus_t = 1.0 - t_value
    snr = (one_minus_t * one_minus_t) / (t_value * t_value)
    sigma_sq = float(rescale_sigma) * float(rescale_sigma)
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / float(rescale_k) + 1.0)
    return (ratio * (one_minus_t * v_pred + x_t) - x_t) / one_minus_t


def scale_speaker_kv_cache(
    context_kv_cache: list[tuple[torch.Tensor, ...]],
    scale: float,
    max_layers: int | None = None,
) -> None:
    """
    In-place scaling of speaker K/V tensors in precomputed context cache.
    """
    if max_layers is None:
        n_layers = len(context_kv_cache)
    else:
        n_layers = max(0, min(int(max_layers), len(context_kv_cache)))
    for i in range(n_layers):
        layer_kv = context_kv_cache[i]
        if len(layer_kv) < 4:
            raise ValueError(
                f"Expected at least 4 tensors in context KV cache entry, got {len(layer_kv)}"
            )
        k_speaker = layer_kv[2]
        v_speaker = layer_kv[3]
        k_speaker.mul_(scale)
        v_speaker.mul_(scale)


@torch.inference_mode()
def sample_euler_rf_cfg(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    block_cache: BlockCacheState | None = None,
) -> torch.Tensor:
    """
    Euler sampling over RF ODE with text/reference/caption conditioning CFG.

    Returns:
      latent sequence in patched space, shape (B, sequence_length, patched_latent_dim)
    """
    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    latent_dim = model.cfg.patched_latent_dim

    rng, rng_device = _make_rng(seed=seed, device=device)
    x_t = torch.randn(
        (batch_size, sequence_length, latent_dim), device=rng_device, dtype=dtype, generator=rng
    )
    if rng_device != device:
        x_t = x_t.to(device=device)
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    if cfg_scale is not None:
        # Backward compatibility for old single-scale caller.
        cfg_scale_text = float(cfg_scale)
        cfg_scale_caption = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)
    if not model.cfg.use_speaker_condition:
        cfg_scale_speaker = 0.0
        speaker_kv_scale = None

    cfg_guidance_mode = str(cfg_guidance_mode).strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, alternating."
        )

    init_scale = 0.999
    t_schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device) * init_scale
    use_independent_cfg = cfg_guidance_mode == "independent"
    use_joint_cfg = cfg_guidance_mode == "joint"
    use_alternating_cfg = cfg_guidance_mode == "alternating"

    (
        text_state_cond,
        text_mask_cond,
        speaker_state_cond,
        speaker_mask_cond,
        caption_state_cond,
        caption_mask_cond,
    ) = model.encode_conditions(
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        ref_latent=ref_latent,
        ref_mask=ref_mask,
        caption_input_ids=caption_input_ids,
        caption_mask=caption_mask,
    )
    text_state_uncond = torch.zeros_like(text_state_cond)
    text_mask_uncond = torch.zeros_like(text_mask_cond)
    speaker_state_uncond = None
    speaker_mask_uncond = None
    if model.cfg.use_speaker_condition:
        if speaker_state_cond is None or speaker_mask_cond is None:
            raise RuntimeError(
                "Speaker conditioning is enabled but encoded speaker state is missing."
            )
        speaker_state_uncond = torch.zeros_like(speaker_state_cond)
        speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)
    caption_state_uncond = None
    caption_mask_uncond = None
    if model.cfg.use_caption_condition:
        if caption_state_cond is None or caption_mask_cond is None:
            raise RuntimeError(
                "Caption conditioning is enabled but encoded caption state is missing."
            )
        caption_state_uncond = torch.zeros_like(caption_state_cond)
        caption_mask_uncond = torch.zeros_like(caption_mask_cond)

    has_text_cfg = cfg_scale_text > 0
    has_caption_cfg = (
        model.cfg.use_caption_condition
        and cfg_scale_caption > 0
        and caption_mask_cond is not None
        and bool(caption_mask_cond.any().item())
    )
    has_speaker_cfg = cfg_scale_speaker > 0

    def _bundle(
        *,
        text_state: torch.Tensor,
        text_mask_val: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask_val: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask_val: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return (
            text_state,
            text_mask_val,
            speaker_state,
            speaker_mask_val,
            caption_state,
            caption_mask_val,
        )

    cond_bundle = _bundle(
        text_state=text_state_cond,
        text_mask_val=text_mask_cond,
        speaker_state=speaker_state_cond,
        speaker_mask_val=speaker_mask_cond,
        caption_state=caption_state_cond,
        caption_mask_val=caption_mask_cond,
    )
    enabled_cfg_names: list[str] = []
    cfg_scales: dict[str, float] = {}
    if has_text_cfg:
        enabled_cfg_names.append("text")
        cfg_scales["text"] = float(cfg_scale_text)
    if has_speaker_cfg:
        enabled_cfg_names.append("speaker")
        cfg_scales["speaker"] = float(cfg_scale_speaker)
    if has_caption_cfg:
        enabled_cfg_names.append("caption")
        cfg_scales["caption"] = float(cfg_scale_caption)

    independent_bundles = [cond_bundle]
    independent_names = ["cond"]
    if use_independent_cfg:
        for name in enabled_cfg_names:
            independent_names.append(name)
            independent_bundles.append(
                _bundle(
                    text_state=text_state_uncond if name == "text" else text_state_cond,
                    text_mask_val=text_mask_uncond if name == "text" else text_mask_cond,
                    speaker_state=(
                        speaker_state_uncond if name == "speaker" else speaker_state_cond
                    ),
                    speaker_mask_val=(
                        speaker_mask_uncond if name == "speaker" else speaker_mask_cond
                    ),
                    caption_state=(
                        caption_state_uncond if name == "caption" else caption_state_cond
                    ),
                    caption_mask_val=(
                        caption_mask_uncond if name == "caption" else caption_mask_cond
                    ),
                )
            )
    cfg_batch_mult = len(independent_bundles)

    def _cat_optional_tensors(values: list[torch.Tensor | None]) -> torch.Tensor | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        if len(present) != len(values):
            raise ValueError("Cannot concatenate optional condition tensors with mixed presence.")
        return torch.cat(present, dim=0)

    independent_text_state = torch.cat([bundle[0] for bundle in independent_bundles], dim=0)
    independent_text_mask = torch.cat([bundle[1] for bundle in independent_bundles], dim=0)
    independent_speaker_state = _cat_optional_tensors([bundle[2] for bundle in independent_bundles])
    independent_speaker_mask = _cat_optional_tensors([bundle[3] for bundle in independent_bundles])
    independent_caption_state = _cat_optional_tensors([bundle[4] for bundle in independent_bundles])
    independent_caption_mask = _cat_optional_tensors([bundle[5] for bundle in independent_bundles])

    joint_uncond_bundle = _bundle(
        text_state=text_state_uncond,
        text_mask_val=text_mask_uncond,
        speaker_state=speaker_state_uncond,
        speaker_mask_val=speaker_mask_uncond,
        caption_state=caption_state_uncond,
        caption_mask_val=caption_mask_uncond,
    )

    alternating_bundles: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ] = {
        "text": _bundle(
            text_state=text_state_uncond,
            text_mask_val=text_mask_uncond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        ),
        "caption": _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_uncond,
            caption_mask_val=caption_mask_uncond,
        ),
    }
    if has_speaker_cfg:
        alternating_bundles["speaker"] = _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_uncond,
            speaker_mask_val=speaker_mask_uncond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        )

    # Force-speaker scaling operates on projected speaker K/V, so it requires context KV caches.
    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))

    context_kv_cond = None
    context_kv_cfg = None
    context_kv_joint_uncond = None
    context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    if effective_use_context_kv_cache:
        context_kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
            caption_state=caption_state_cond,
        )
        if use_independent_cfg and cfg_batch_mult > 1:
            context_kv_cfg = model.build_context_kv_cache(
                text_state=independent_text_state,
                speaker_state=independent_speaker_state,
                caption_state=independent_caption_state,
            )
        elif use_joint_cfg:
            if enabled_cfg_names:
                context_kv_joint_uncond = model.build_context_kv_cache(
                    text_state=joint_uncond_bundle[0],
                    speaker_state=joint_uncond_bundle[2],
                    caption_state=joint_uncond_bundle[4],
                )
        elif use_alternating_cfg:
            for name in enabled_cfg_names:
                bundle = alternating_bundles[name]
                context_kv_alternating[name] = model.build_context_kv_cache(
                    text_state=bundle[0],
                    speaker_state=bundle[2],
                    caption_state=bundle[4],
                )
    if speaker_kv_scale is not None:
        scale_speaker_kv_cache(
            context_kv_cache=context_kv_cond,
            scale=float(speaker_kv_scale),
            max_layers=speaker_kv_max_layers,
        )
        if context_kv_cfg is not None:
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cfg,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
        for cache in context_kv_alternating.values():
            scale_speaker_kv_cache(
                context_kv_cache=cache,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
    speaker_kv_active = speaker_kv_scale is not None

    # --- Precompute all timestep condition embeddings (batch of num_steps) ---
    _precomputed_conds: list[torch.Tensor] | None = None
    if block_cache is not None:
        all_t_vals = t_schedule[:-1].to(dtype=dtype)  # (num_steps,)
        all_t_embed = get_timestep_embedding(all_t_vals, model.cfg.timestep_embed_dim).to(
            dtype=dtype
        )
        all_cond = model.cond_module(all_t_embed)  # (num_steps, model_dim*3)
        _precomputed_conds = [all_cond[j : j + 1, None, :] for j in range(num_steps)]
        del all_t_embed, all_cond

    _vcache_prev_v: torch.Tensor | None = None
    _vcache_skips: int = 0
    _use_vcache = (
        block_cache is not None and block_cache.velocity_cache_max_skip > 0
    )
    _vcache_max = block_cache.velocity_cache_max_skip if _use_vcache else 0
    _vcache_warmup = block_cache.warmup_steps if block_cache is not None else 0

    # Precompute per-step scalars to avoid .item() and Python branching in hot loop
    _t_floats = [float(t_schedule[j].item()) for j in range(num_steps)]
    _dt_floats = [float(t_schedule[j + 1].item()) - _t_floats[j] for j in range(num_steps)]
    _has_cfg = bool(enabled_cfg_names)
    _use_cfg_flags = [
        _has_cfg and (cfg_min_t <= _t_floats[j] <= cfg_max_t) for j in range(num_steps)
    ]
    # Precompute tt tensors for steps that will actually run forwards
    _tt_tensors = [
        torch.full((batch_size,), _t_floats[j], device=device, dtype=dtype)
        for j in range(num_steps)
    ]

    for i in range(num_steps):
        # --- Velocity caching: reuse previous v entirely, skip all forwards ---
        if (
            _use_vcache
            and _vcache_prev_v is not None
            and _vcache_skips < _vcache_max
            and i >= _vcache_warmup
        ):
            v = _vcache_prev_v
            _vcache_skips += 1
            block_cache.step += 1
        else:
            _vcache_skips = 0
            tt = _tt_tensors[i]

            # Precomputed cond_embed for this step (avoids per-step MLP call)
            _pc = _precomputed_conds[i] if _precomputed_conds is not None else None

            use_cfg = _use_cfg_flags[i]
            if use_cfg:
                if use_independent_cfg:
                    x_t_cfg = torch.cat([x_t] * cfg_batch_mult, dim=0).to(dtype)
                    tt_cfg = tt.repeat(cfg_batch_mult)
                    _pc_cfg = _pc.expand(cfg_batch_mult, -1, -1) if _pc is not None else None
                    v_out = model.forward_with_encoded_conditions(
                        x_t=x_t_cfg,
                        t=tt_cfg,
                        text_state=independent_text_state,
                        text_mask=independent_text_mask,
                        speaker_state=independent_speaker_state,
                        speaker_mask=independent_speaker_mask,
                        caption_state=independent_caption_state,
                        caption_mask=independent_caption_mask,
                        context_kv_cache=context_kv_cfg,
                        cache_state=block_cache,
                        precomputed_cond_embed=_pc_cfg,
                    )
                    chunks = v_out.chunk(cfg_batch_mult, dim=0)
                    v = chunks[0]
                    for name, chunk in zip(independent_names[1:], chunks[1:], strict=True):
                        v = v + cfg_scales[name] * (chunks[0] - chunk)
                else:
                    v_cond = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=text_state_cond,
                        text_mask=text_mask_cond,
                        speaker_state=speaker_state_cond,
                        speaker_mask=speaker_mask_cond,
                        caption_state=caption_state_cond,
                        caption_mask=caption_mask_cond,
                        context_kv_cache=context_kv_cond,
                        cache_state=block_cache,
                        precomputed_cond_embed=_pc,
                    )
                    if use_joint_cfg:
                        if len(enabled_cfg_names) > 1:
                            joint_scales = [cfg_scales[name] for name in enabled_cfg_names]
                            if max(joint_scales) - min(joint_scales) > 1e-6:
                                raise ValueError(
                                    "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                                    "set matching text/speaker/caption scales or use --cfg-scale."
                                )
                        joint_scale = cfg_scales[enabled_cfg_names[0]]
                        v_uncond_joint = model.forward_with_encoded_conditions(
                            x_t=x_t.to(dtype),
                            t=tt,
                            text_state=joint_uncond_bundle[0],
                            text_mask=joint_uncond_bundle[1],
                            speaker_state=joint_uncond_bundle[2],
                            speaker_mask=joint_uncond_bundle[3],
                            caption_state=joint_uncond_bundle[4],
                            caption_mask=joint_uncond_bundle[5],
                            context_kv_cache=context_kv_joint_uncond,
                            cache_state=block_cache,
                            precomputed_cond_embed=_pc,
                        )
                        v = v_cond + joint_scale * (v_cond - v_uncond_joint)
                    elif use_alternating_cfg:
                        alt_name = enabled_cfg_names[i % len(enabled_cfg_names)]
                        alt_bundle = alternating_bundles[alt_name]
                        v_uncond_alt = model.forward_with_encoded_conditions(
                            x_t=x_t.to(dtype),
                            t=tt,
                            text_state=alt_bundle[0],
                            text_mask=alt_bundle[1],
                            speaker_state=alt_bundle[2],
                            speaker_mask=alt_bundle[3],
                            caption_state=alt_bundle[4],
                            caption_mask=alt_bundle[5],
                            context_kv_cache=context_kv_alternating.get(alt_name),
                            cache_state=block_cache,
                            precomputed_cond_embed=_pc,
                        )
                        v = v_cond + cfg_scales[alt_name] * (v_cond - v_uncond_alt)
                    else:
                        raise RuntimeError(f"Unexpected cfg_guidance_mode: {cfg_guidance_mode}")
            else:
                v = model.forward_with_encoded_conditions(
                    x_t=x_t.to(dtype),
                    t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    caption_state=caption_state_cond,
                    caption_mask=caption_mask_cond,
                    context_kv_cache=context_kv_cond,
                    cache_state=block_cache,
                    precomputed_cond_embed=_pc,
                )

            _vcache_prev_v = v

        if rescale_k is not None and rescale_sigma is not None:
            v = temporal_score_rescale(
                v_pred=v,
                x_t=x_t,
                t=_t_floats[i],
                rescale_k=float(rescale_k),
                rescale_sigma=float(rescale_sigma),
            )

        if (
            speaker_kv_active
            and speaker_kv_min_t is not None
            and (_t_floats[i] + _dt_floats[i] < speaker_kv_min_t)
            and (_t_floats[i] >= speaker_kv_min_t)
        ):
            inv_scale = 1.0 / float(speaker_kv_scale)
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cond,
                scale=inv_scale,
                max_layers=speaker_kv_max_layers,
            )
            if context_kv_cfg is not None:
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv_cfg,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            for cache in context_kv_alternating.values():
                scale_speaker_kv_cache(
                    context_kv_cache=cache,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            speaker_kv_active = False

        x_t = x_t.add_(v, alpha=_dt_floats[i])

    return x_t
