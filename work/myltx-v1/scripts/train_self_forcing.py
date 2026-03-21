#!/usr/bin/env python3
"""Self-forcing training with SSM memory for MemoryForcing.

This script now supports two modes:
1. Preferred: real `.precomputed` ODE-regression triplets exported by
   `ode/convert_ode_pt_to_precomputed.py`.
2. Fallback: synthetic smoke batches when no real data is available.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except ImportError:
    Accelerator = None

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from ltx_core.model.transformer.ssm_integration import SSMAugmentedLTXModel
from ltx_core.model.transformer.ssm_memory import SSMConfig, SSMState
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape

from self_forcing_data import (
    SwitchEpisode,
    build_chunk_prompt_schedule,
    discover_precomputed_sample_ids,
    load_ode_precomputed_sample,
    load_switch_episodes,
    split_uniform_spans,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


@dataclass
class SelfForcingConfig:
    base_checkpoint: str = ""
    gemma_path: str = ""

    ssm_d_state: int = 64
    ssm_gate_bias: float = -2.0
    ssm_switch_state_decay: float = 1.0

    freeze_base: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    total_steps: int = 5000
    warmup_steps: int = 500
    batch_size: int = 1
    gradient_accumulation_steps: int = 4

    block_size: int = 6
    window_blocks: int = 4

    self_forcing: bool = False
    self_forcing_start_step: int = 2000

    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    memory_loss_weight: float = 0.1

    data_root: str = ""
    allow_synthetic_fallback: bool = True
    max_data_samples: int = 0
    switch_episode_manifest: str = ""
    keep_loss_weight: float = 1.0
    edit_loss_weight: float = 1.0
    switch_loss_weight: float = 1.0
    use_asymmetric_switching: bool = False
    num_workers: int = 2
    smoke_max_prompt_tokens: int = 0
    smoke_max_video_frames: int = 0
    smoke_max_video_height: int = 0
    smoke_max_video_width: int = 0
    smoke_max_audio_time_steps: int = 0

    log_interval: int = 10
    checkpoint_interval: int = 500
    output_dir: str = "outputs/self_forcing"

    seed: int = 42
    mixed_precision: str = "bf16"

    @classmethod
    def from_yaml(cls, path: str) -> "SelfForcingConfig":
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        valid = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in valid})


def apply_switch_state_decay(ssm_state: SSMState, decay: float) -> bool:
    if decay < 0:
        raise ValueError(f"ssm_switch_state_decay must be non-negative, got {decay}")
    if decay >= 1.0:
        return False

    scale_fn = getattr(ssm_state, "scale_", None)
    if callable(scale_fn):
        scale_fn(decay)
        return True

    if not ssm_state.states:
        return False

    for key, tensor in ssm_state.states.items():
        ssm_state.states[key] = tensor * decay
    return True


def chunked_forward_with_ssm(
    model: SSMAugmentedLTXModel,
    video_chunks: list[torch.Tensor],
    audio_chunks: list[torch.Tensor],
    video_target_chunks: list[torch.Tensor],
    audio_target_chunks: list[torch.Tensor],
    video_sigma: torch.Tensor,
    audio_sigma: torch.Tensor,
    video_positions_chunks: list[torch.Tensor],
    audio_positions_chunks: list[torch.Tensor],
    video_contexts: list[torch.Tensor] | torch.Tensor,
    audio_contexts: list[torch.Tensor] | torch.Tensor,
    context_masks: list[torch.Tensor] | torch.Tensor,
    prompt_switch_flags: list[bool] | None,
    config: SelfForcingConfig,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    from ltx_core.model.transformer.modality import Modality

    def _normalize_chunk_inputs(value: list[torch.Tensor] | torch.Tensor, num_chunks: int, name: str) -> list[torch.Tensor]:
        if isinstance(value, list):
            if len(value) != num_chunks:
                raise ValueError(f"{name} length mismatch: expected {num_chunks}, got {len(value)}")
            return value
        return [value for _ in range(num_chunks)]

    num_chunks = len(video_chunks)
    if not (
        len(audio_chunks) == len(video_target_chunks) == len(audio_target_chunks) == len(video_positions_chunks) == len(audio_positions_chunks) == num_chunks
    ):
        raise ValueError("Chunk lists must all have the same length")

    video_context_list = _normalize_chunk_inputs(video_contexts, num_chunks, "video_contexts")
    audio_context_list = _normalize_chunk_inputs(audio_contexts, num_chunks, "audio_contexts")
    context_mask_list = _normalize_chunk_inputs(context_masks, num_chunks, "context_masks")
    if prompt_switch_flags is None:
        prompt_switch_flags = [False for _ in range(num_chunks)]
    if len(prompt_switch_flags) != num_chunks:
        raise ValueError(f"prompt_switch_flags length mismatch: expected {num_chunks}, got {len(prompt_switch_flags)}")

    ssm_state = SSMState.empty()
    total_video_loss = torch.tensor(0.0, device=video_chunks[0].device)
    total_audio_loss = torch.tensor(0.0, device=video_chunks[0].device)
    total_weighted_loss = torch.tensor(0.0, device=video_chunks[0].device)
    keep_loss_sum = torch.tensor(0.0, device=video_chunks[0].device)
    edit_loss_sum = torch.tensor(0.0, device=video_chunks[0].device)
    switch_loss_sum = torch.tensor(0.0, device=video_chunks[0].device)
    keep_chunks = 0
    edit_chunks = 0
    switch_chunks = 0
    switch_state_decay_chunks = 0
    n_loss_chunks = 0
    use_self_forcing = config.self_forcing and step >= config.self_forcing_start_step

    past_video_latents: list[torch.Tensor] = []
    past_audio_latents: list[torch.Tensor] = []

    for chunk_idx in range(num_chunks):
        v_chunk = video_chunks[chunk_idx]
        a_chunk = audio_chunks[chunk_idx]
        v_targets = video_target_chunks[chunk_idx]
        a_targets = audio_target_chunks[chunk_idx]
        v_pos = video_positions_chunks[chunk_idx]
        a_pos = audio_positions_chunks[chunk_idx]
        v_context = video_context_list[chunk_idx]
        a_context = audio_context_list[chunk_idx]
        context_mask = context_mask_list[chunk_idx]
        is_switch_chunk = bool(prompt_switch_flags[chunk_idx])
        if is_switch_chunk and config.ssm_switch_state_decay < 1.0:
            if apply_switch_state_decay(ssm_state, config.ssm_switch_state_decay):
                switch_state_decay_chunks += 1

        batch_size = v_chunk.shape[0]
        v_seq = v_chunk.shape[1]
        a_seq = a_chunk.shape[1]
        v_timesteps = video_sigma.view(-1, 1).expand(batch_size, v_seq)
        a_timesteps = audio_sigma.view(-1, 1).expand(batch_size, a_seq)

        video_mod = Modality(
            enabled=True,
            sigma=video_sigma,
            latent=v_chunk,
            timesteps=v_timesteps,
            positions=v_pos,
            context=v_context,
            context_mask=context_mask,
        )
        audio_mod = Modality(
            enabled=True,
            sigma=audio_sigma,
            latent=a_chunk,
            timesteps=a_timesteps,
            positions=a_pos,
            context=a_context,
            context_mask=context_mask,
        )

        v_pred, a_pred, ssm_state = model(video=video_mod, audio=audio_mod, ssm_state=ssm_state)

        sigma_clamp = video_sigma.clamp_min(1e-6).view(-1, 1, 1)
        v_velocity_target = (v_chunk - v_targets) / sigma_clamp
        a_velocity_target = (a_chunk - a_targets) / sigma_clamp

        v_loss = F.mse_loss(v_pred, v_velocity_target)
        a_loss = F.mse_loss(a_pred, a_velocity_target) if a_pred is not None else torch.tensor(0.0, device=v_chunk.device)
        raw_chunk_loss = config.video_loss_weight * v_loss + config.audio_loss_weight * a_loss
        chunk_weight = config.keep_loss_weight
        if is_switch_chunk:
            chunk_weight = config.edit_loss_weight * config.switch_loss_weight
            edit_loss_sum = edit_loss_sum + raw_chunk_loss.detach()
            switch_loss_sum = switch_loss_sum + raw_chunk_loss.detach()
            edit_chunks += 1
            switch_chunks += 1
        else:
            keep_loss_sum = keep_loss_sum + raw_chunk_loss.detach()
            keep_chunks += 1

        total_video_loss = total_video_loss + v_loss
        total_audio_loss = total_audio_loss + a_loss
        total_weighted_loss = total_weighted_loss + raw_chunk_loss * chunk_weight
        n_loss_chunks += 1

        past_video_latents.append(v_chunk.detach())
        past_audio_latents.append(a_chunk.detach())
        if len(past_video_latents) > config.window_blocks:
            evicted_v = past_video_latents.pop(0)
            evicted_a = past_audio_latents.pop(0)
            ssm_state = model.compress_evicted_tokens(ssm_state, evicted_v, evicted_a)

    total_video_loss = total_video_loss / max(n_loss_chunks, 1)
    total_audio_loss = total_audio_loss / max(n_loss_chunks, 1)
    loss = total_weighted_loss / max(n_loss_chunks, 1)
    metrics = {
        "video_loss": total_video_loss.item(),
        "audio_loss": total_audio_loss.item(),
        "total_loss": loss.item(),
        "keep_loss": (keep_loss_sum / max(keep_chunks, 1)).item() if keep_chunks else 0.0,
        "edit_loss": (edit_loss_sum / max(edit_chunks, 1)).item() if edit_chunks else 0.0,
        "switch_loss": (switch_loss_sum / max(switch_chunks, 1)).item() if switch_chunks else 0.0,
        "num_chunks": float(num_chunks),
        "num_switch_chunks": float(switch_chunks),
        "switch_state_decay_chunks": float(switch_state_decay_chunks),
        "self_forcing": float(use_self_forcing),
    }
    return loss, metrics


def build_video_chunk_spans(num_frames: int, height: int, width: int, block_size: int) -> list[tuple[int, int]]:
    tokens_per_frame = height * width
    total_tokens = num_frames * tokens_per_frame
    if total_tokens <= 0:
        return []

    spans = [(0, min(tokens_per_frame, total_tokens))]
    offset = spans[0][1]
    tokens_per_block = block_size * tokens_per_frame
    while offset < total_tokens:
        end = min(offset + tokens_per_block, total_tokens)
        spans.append((offset, end))
        offset = end
    return spans


def chunk_by_spans(tensor: torch.Tensor, spans: list[tuple[int, int]], seq_dim: int) -> list[torch.Tensor]:
    return [tensor.narrow(seq_dim, start, end - start) for start, end in spans]


def prepare_context_mask(attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Keep prompt masks in the numeric format expected by the base transformer."""
    attention_mask = attention_mask.to(device=device)
    if attention_mask.dtype == torch.bool:
        attention_mask = attention_mask.to(dtype=torch.int64)
    return attention_mask.contiguous()


def crop_condition_payload(conditions: dict[str, torch.Tensor], config: SelfForcingConfig) -> dict[str, torch.Tensor]:
    video_prompt_embeds = conditions["video_prompt_embeds"]
    prompt_attention_mask = conditions["prompt_attention_mask"]
    audio_context = conditions.get("audio_prompt_embeds")
    if config.smoke_max_prompt_tokens > 0:
        start = max(prompt_attention_mask.shape[0] - config.smoke_max_prompt_tokens, 0)
        video_prompt_embeds = video_prompt_embeds[start:]
        prompt_attention_mask = prompt_attention_mask[start:]
        if audio_context is not None:
            audio_context = audio_context[start:]

    if audio_context is None:
        audio_context = torch.zeros(video_prompt_embeds.shape[0], 2048, dtype=video_prompt_embeds.dtype)

    return {
        "video_prompt_embeds": video_prompt_embeds.cpu().contiguous(),
        "audio_prompt_embeds": audio_context.cpu().contiguous(),
        "prompt_attention_mask": prompt_attention_mask.cpu().contiguous(),
    }


def condition_payload_to_tensors(
    condition_payload: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        condition_payload["video_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype),
        condition_payload["audio_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype),
        prepare_context_mask(condition_payload["prompt_attention_mask"].unsqueeze(0), device),
    )


def encode_prompt_conditions(prompt: str, text_encoder: Any, embeddings_processor: Any) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        hidden_states, prompt_attention_mask = text_encoder.encode(prompt, padding_side="left")
        video_prompt_embeds, audio_prompt_embeds = embeddings_processor.feature_extractor(
            hidden_states,
            prompt_attention_mask,
            "left",
        )

    payload: dict[str, torch.Tensor] = {
        "video_prompt_embeds": video_prompt_embeds[0].cpu().contiguous(),
        "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
    }
    if audio_prompt_embeds is not None:
        payload["audio_prompt_embeds"] = audio_prompt_embeds[0].cpu().contiguous()
    return payload


def load_switch_prompt_cache(
    config: SelfForcingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[SwitchEpisode], dict[str, dict[str, torch.Tensor]]]:
    if not config.switch_episode_manifest:
        return [], {}

    episodes = load_switch_episodes(config.switch_episode_manifest)
    if not episodes:
        logger.warning("Switch manifest is configured but no episodes were loaded: %s", config.switch_episode_manifest)
        return [], {}

    unique_prompts = sorted({segment.prompt for episode in episodes for segment in episode.segments})
    if not unique_prompts:
        logger.warning("Switch manifest contains no prompt text: %s", config.switch_episode_manifest)
        return episodes, {}

    from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder

    logger.info("Encoding %d unique switch prompts from %s", len(unique_prompts), config.switch_episode_manifest)
    text_encoder = load_text_encoder(config.gemma_path, device=device, dtype=dtype)
    embeddings_processor = load_embeddings_processor(config.base_checkpoint, device=device, dtype=dtype)

    prompt_cache: dict[str, dict[str, torch.Tensor]] = {}
    try:
        for idx, prompt in enumerate(unique_prompts, start=1):
            prompt_cache[prompt] = crop_condition_payload(
                encode_prompt_conditions(prompt, text_encoder, embeddings_processor),
                config,
            )
            if idx == len(unique_prompts) or idx % 8 == 0:
                logger.info("Encoded switch prompts: %d/%d", idx, len(unique_prompts))
    finally:
        del text_encoder
        del embeddings_processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Loaded %d switch episodes with %d cached prompts", len(episodes), len(prompt_cache))
    return episodes, prompt_cache


def build_chunk_condition_schedule(
    sample_id: str,
    num_chunks: int,
    base_conditions: dict[str, torch.Tensor],
    config: SelfForcingConfig,
    device: torch.device,
    dtype: torch.dtype,
    switch_episodes: list[SwitchEpisode] | None = None,
    prompt_cache: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    base_payload = crop_condition_payload(base_conditions, config)
    default_video_context, default_audio_context, default_context_mask = condition_payload_to_tensors(base_payload, device, dtype)
    fallback_contexts = [default_video_context for _ in range(num_chunks)]
    fallback_audio_contexts = [default_audio_context for _ in range(num_chunks)]
    fallback_masks = [default_context_mask for _ in range(num_chunks)]
    fallback_metadata = {
        "episode_id": None,
        "segment_indices": [0 for _ in range(num_chunks)],
        "chunk_prompts": [None for _ in range(num_chunks)],
        "prompt_switch_flags": [False for _ in range(num_chunks)],
    }
    if num_chunks <= 0:
        return fallback_contexts, fallback_audio_contexts, fallback_masks, fallback_metadata

    if not switch_episodes or not prompt_cache:
        return fallback_contexts, fallback_audio_contexts, fallback_masks, fallback_metadata

    schedule = build_chunk_prompt_schedule(sample_id, num_chunks, switch_episodes)
    if schedule is None:
        return fallback_contexts, fallback_audio_contexts, fallback_masks, fallback_metadata

    video_contexts: list[torch.Tensor] = []
    audio_contexts: list[torch.Tensor] = []
    context_masks: list[torch.Tensor] = []
    for prompt in schedule["prompts"]:
        payload = prompt_cache.get(prompt, base_payload)
        video_context, audio_context, context_mask = condition_payload_to_tensors(payload, device, dtype)
        video_contexts.append(video_context)
        audio_contexts.append(audio_context)
        context_masks.append(context_mask)

    return video_contexts, audio_contexts, context_masks, {
        "episode_id": schedule["episode_id"],
        "segment_indices": schedule["segment_indices"],
        "chunk_prompts": schedule["prompts"],
        "prompt_switch_flags": schedule["switch_flags"],
    }


def get_video_positions(
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
    fps: float,
    device: torch.device,
) -> torch.Tensor:
    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            frames=num_frames,
            height=height,
            width=width,
            batch=batch_size,
            channels=128,
        ),
        device=device,
    )
    pixel_coords = get_pixel_coords(latent_coords, VIDEO_SCALE_FACTORS, causal_fix=True).to(torch.float32)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
    return pixel_coords


def get_audio_positions(num_time_steps: int, batch_size: int, device: torch.device) -> torch.Tensor:
    patchifier = AudioPatchifier(patch_size=1)
    return patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(
            frames=num_time_steps,
            mel_bins=16,
            batch=batch_size,
            channels=8,
        ),
        device=device,
    ).to(torch.float32)


def build_synthetic_batch(
    base_model: Any,
    config: SelfForcingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    batch_size = config.batch_size
    num_frames = max(1, min(25, config.smoke_max_video_frames)) if config.smoke_max_video_frames > 0 else 25
    height = max(1, min(17, config.smoke_max_video_height)) if config.smoke_max_video_height > 0 else 17
    width = max(1, min(30, config.smoke_max_video_width)) if config.smoke_max_video_width > 0 else 30
    audio_tokens = max(1, min(125, config.smoke_max_audio_time_steps)) if config.smoke_max_audio_time_steps > 0 else 125
    context_tokens = max(1, min(384, config.smoke_max_prompt_tokens)) if config.smoke_max_prompt_tokens > 0 else 384
    tokens_per_frame = height * width
    total_video_tokens = num_frames * tokens_per_frame
    video_dim = base_model.inner_dim
    audio_dim = base_model.audio_inner_dim

    video_spans = build_video_chunk_spans(num_frames, height, width, config.block_size)
    audio_spans = split_uniform_spans(audio_tokens, len(video_spans))

    v_latent = torch.randn(batch_size, total_video_tokens, video_dim, device=device, dtype=dtype)
    v_target = torch.randn_like(v_latent)
    a_latent = torch.randn(batch_size, audio_tokens, audio_dim, device=device, dtype=dtype)
    a_target = torch.randn_like(a_latent)
    video_context = torch.randn(batch_size, context_tokens, 4096, device=device, dtype=dtype)
    audio_context = torch.randn(batch_size, context_tokens, 2048, device=device, dtype=dtype)
    context_mask = prepare_context_mask(torch.ones(batch_size, context_tokens, device=device, dtype=torch.int64), device)

    return {
        "video_chunks": chunk_by_spans(v_latent, video_spans, seq_dim=1),
        "video_target_chunks": chunk_by_spans(v_target, video_spans, seq_dim=1),
        "audio_chunks": chunk_by_spans(a_latent, audio_spans, seq_dim=1),
        "audio_target_chunks": chunk_by_spans(a_target, audio_spans, seq_dim=1),
        "video_positions_chunks": [
            torch.zeros(batch_size, 3, end - start, 2, device=device, dtype=torch.float32)
            for start, end in video_spans
        ],
        "audio_positions_chunks": [
            torch.zeros(batch_size, 1, end - start, 2, device=device, dtype=torch.float32)
            for start, end in audio_spans
        ],
        "video_contexts": [video_context for _ in range(len(video_spans))],
        "audio_contexts": [audio_context for _ in range(len(video_spans))],
        "context_masks": [context_mask for _ in range(len(video_spans))],
        "prompt_switch_flags": [False for _ in range(len(video_spans))],
        "sigma": torch.tensor([0.5] * batch_size, device=device, dtype=dtype),
        "sample_id": "synthetic",
        "switch_episode_id": None,
    }


def build_precomputed_batch(
    sample_id: str,
    sample: dict[str, Any],
    config: SelfForcingConfig,
    device: torch.device,
    dtype: torch.dtype,
    switch_episodes: list[SwitchEpisode] | None = None,
    prompt_cache: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    video_patchifier = VideoLatentPatchifier(patch_size=1)
    audio_patchifier = AudioPatchifier(patch_size=1)

    latents = sample["latents"]
    audio = sample["audio_latents"]
    conditions = sample["conditions"]

    num_frames = int(latents["num_frames"])
    height = int(latents["height"])
    width = int(latents["width"])
    fps = float(latents.get("fps", 24.0))
    num_time_steps = int(audio["num_time_steps"])

    max_frames = max(1, min(num_frames, config.smoke_max_video_frames)) if config.smoke_max_video_frames > 0 else num_frames
    max_height = max(1, min(height, config.smoke_max_video_height)) if config.smoke_max_video_height > 0 else height
    max_width = max(1, min(width, config.smoke_max_video_width)) if config.smoke_max_video_width > 0 else width
    max_audio_steps = max(1, min(num_time_steps, config.smoke_max_audio_time_steps)) if config.smoke_max_audio_time_steps > 0 else num_time_steps
    base_condition_payload = crop_condition_payload(conditions, config)

    video_latents = latents["latents"][:, :max_frames, :max_height, :max_width].unsqueeze(0).to(device=device, dtype=dtype)
    video_targets = latents["ode_target_latents"][:, :max_frames, :max_height, :max_width].unsqueeze(0).to(device=device, dtype=dtype)
    audio_latents = audio["latents"][:, :max_audio_steps, :].unsqueeze(0).to(device=device, dtype=dtype)
    audio_targets = audio["ode_target_latents"][:, :max_audio_steps, :].unsqueeze(0).to(device=device, dtype=dtype)

    num_frames = video_latents.shape[2]
    height = video_latents.shape[3]
    width = video_latents.shape[4]
    num_time_steps = audio_latents.shape[2]
    sigma = torch.tensor([float(latents["ode_sigma"])], device=device, dtype=dtype)

    if (
        num_frames != int(latents["num_frames"])
        or height != int(latents["height"])
        or width != int(latents["width"])
        or num_time_steps != int(audio["num_time_steps"])
        or base_condition_payload["prompt_attention_mask"].shape[0] != conditions["prompt_attention_mask"].shape[0]
    ):
        logger.info(
            "Smoke crop active: frames=%d height=%d width=%d audio_steps=%d prompt_tokens=%d",
            num_frames,
            height,
            width,
            num_time_steps,
            base_condition_payload["prompt_attention_mask"].shape[0],
        )

    video_tokens = video_patchifier.patchify(video_latents)
    video_target_tokens = video_patchifier.patchify(video_targets)
    audio_tokens = audio_patchifier.patchify(audio_latents)
    audio_target_tokens = audio_patchifier.patchify(audio_targets)

    video_spans = build_video_chunk_spans(num_frames, height, width, config.block_size)
    audio_spans = split_uniform_spans(audio_tokens.shape[1], len(video_spans))
    video_positions = get_video_positions(num_frames, height, width, batch_size=1, fps=fps, device=device)
    audio_positions = get_audio_positions(num_time_steps, batch_size=1, device=device)
    video_contexts, audio_contexts, context_masks, schedule_metadata = build_chunk_condition_schedule(
        sample_id=sample_id,
        num_chunks=len(video_spans),
        base_conditions=conditions,
        config=config,
        device=device,
        dtype=dtype,
        switch_episodes=switch_episodes,
        prompt_cache=prompt_cache,
    )

    return {
        "video_chunks": chunk_by_spans(video_tokens, video_spans, seq_dim=1),
        "video_target_chunks": chunk_by_spans(video_target_tokens, video_spans, seq_dim=1),
        "audio_chunks": chunk_by_spans(audio_tokens, audio_spans, seq_dim=1),
        "audio_target_chunks": chunk_by_spans(audio_target_tokens, audio_spans, seq_dim=1),
        "video_positions_chunks": chunk_by_spans(video_positions, video_spans, seq_dim=2),
        "audio_positions_chunks": chunk_by_spans(audio_positions, audio_spans, seq_dim=2),
        "video_contexts": video_contexts,
        "audio_contexts": audio_contexts,
        "context_masks": context_masks,
        "prompt_switch_flags": schedule_metadata["prompt_switch_flags"],
        "sigma": sigma,
        "switch_episode_id": schedule_metadata["episode_id"],
    }


def train(config: SelfForcingConfig) -> None:
    logger.info("Starting self-forcing training with SSM memory")
    logger.info("Config: freeze_base=%s lr=%s", config.freeze_base, config.learning_rate)
    logger.info(
        "SSM: d_state=%s gate_bias=%s switch_state_decay=%s",
        config.ssm_d_state,
        config.ssm_gate_bias,
        config.ssm_switch_state_decay,
    )
    if any([
        config.smoke_max_prompt_tokens > 0,
        config.smoke_max_video_frames > 0,
        config.smoke_max_video_height > 0,
        config.smoke_max_video_width > 0,
        config.smoke_max_audio_time_steps > 0,
    ]):
        logger.info(
            "Smoke caps: prompt_tokens=%s video=%sx%sx%s audio_steps=%s",
            config.smoke_max_prompt_tokens or "full",
            config.smoke_max_video_frames or "full",
            config.smoke_max_video_height or "full",
            config.smoke_max_video_width or "full",
            config.smoke_max_audio_time_steps or "full",
        )

    if config.ssm_switch_state_decay < 0:
        raise ValueError(f"ssm_switch_state_decay must be non-negative, got {config.ssm_switch_state_decay}")

    if Accelerator:
        set_seed(config.seed)
    else:
        torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32

    switch_episodes: list[SwitchEpisode] = []
    prompt_cache: dict[str, dict[str, torch.Tensor]] = {}
    if config.switch_episode_manifest:
        switch_episodes, prompt_cache = load_switch_prompt_cache(config, device, dtype)

    from ltx_trainer.model_loader import load_transformer

    base_model = load_transformer(config.base_checkpoint, device=device, dtype=dtype)
    base_model = base_model.to(device=device, dtype=dtype)

    ssm_config = SSMConfig(enabled=True, d_state=config.ssm_d_state, gate_bias=config.ssm_gate_bias)
    model = SSMAugmentedLTXModel.from_base(base_model, ssm_config).to(device=device, dtype=dtype)

    if config.freeze_base:
        model.freeze_base()
        logger.info("Base model frozen. SSM params: %s", f"{model.ssm_param_count():,}")
    else:
        logger.info(
            "Joint training. total_params=%s ssm_params=%s",
            f"{sum(p.numel() for p in model.parameters()):,}",
            f"{model.ssm_param_count():,}",
        )

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    def lr_lambda(step_idx: int) -> float:
        if step_idx < config.warmup_steps:
            return step_idx / max(config.warmup_steps, 1)
        progress = (step_idx - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = config.max_data_samples if config.max_data_samples > 0 else None
    sample_ids = discover_precomputed_sample_ids(config.data_root, limit=limit) if config.data_root else []
    if sample_ids:
        logger.info("Using %d precomputed samples from %s", len(sample_ids), config.data_root)
        sample_iter = cycle(sample_ids)
    else:
        if not config.allow_synthetic_fallback:
            raise ValueError("No precomputed samples found and allow_synthetic_fallback=false")
        logger.warning("Falling back to synthetic smoke batches")
        sample_iter = None

    model.train()
    running_loss = 0.0
    for step in range(config.total_steps):
        optimizer.zero_grad()
        if sample_iter is None:
            batch = build_synthetic_batch(base_model, config, device, dtype)
            sample_name = batch["sample_id"]
        else:
            sample_name = next(sample_iter)
            sample = load_ode_precomputed_sample(config.data_root, sample_name)
            batch = build_precomputed_batch(
                sample_id=sample_name,
                sample=sample,
                config=config,
                device=device,
                dtype=dtype,
                switch_episodes=switch_episodes,
                prompt_cache=prompt_cache,
            )

        loss, metrics = chunked_forward_with_ssm(
            model=model,
            video_chunks=batch["video_chunks"],
            audio_chunks=batch["audio_chunks"],
            video_target_chunks=batch["video_target_chunks"],
            audio_target_chunks=batch["audio_target_chunks"],
            video_sigma=batch["sigma"],
            audio_sigma=batch["sigma"],
            video_positions_chunks=batch["video_positions_chunks"],
            audio_positions_chunks=batch["audio_positions_chunks"],
            video_contexts=batch["video_contexts"],
            audio_contexts=batch["audio_contexts"],
            context_masks=batch["context_masks"],
            prompt_switch_flags=batch["prompt_switch_flags"],
            config=config,
            step=step,
        )

        loss.backward()
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        running_loss += metrics["total_loss"]
        if (step + 1) % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            logger.info(
                "Step %d/%d | loss=%.4f | v_loss=%.4f | a_loss=%.4f | keep=%.4f | edit=%.4f | switches=%d | sample=%s | chunks=%d | lr=%.2e",
                step + 1,
                config.total_steps,
                avg_loss,
                metrics["video_loss"],
                metrics["audio_loss"],
                metrics["keep_loss"],
                metrics["edit_loss"],
                int(metrics["num_switch_chunks"]),
                sample_name,
                int(metrics["num_chunks"]),
                scheduler.get_last_lr()[0],
            )
            running_loss = 0.0

        if (step + 1) % config.checkpoint_interval == 0:
            ckpt_path = output_dir / f"ssm_weights_step_{step + 1:05d}.pt"
            torch.save(model.ssm_layers.state_dict(), ckpt_path)
            logger.info("Saved SSM checkpoint: %s", ckpt_path)

    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-forcing training with SSM memory")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    args = parser.parse_args()

    config = SelfForcingConfig.from_yaml(args.config)
    if args.steps is not None:
        config.total_steps = args.steps
    if args.lr is not None:
        config.learning_rate = args.lr

    train(config)
