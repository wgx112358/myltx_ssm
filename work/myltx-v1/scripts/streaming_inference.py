#!/usr/bin/env python3
"""Streaming AV inference utilities."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from official_generation_defaults import get_official_2_stage_resolution
from self_forcing_data import (
    SwitchEpisode,
    load_switch_episodes,
    select_switch_episode_by_id,
    split_uniform_spans,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
OFFICIAL_SMALL_RESOLUTION = get_official_2_stage_resolution("small")


@dataclass
class StreamingConfig:
    mode: str = "replay"
    prompt: str = "A calm ocean scene with gentle waves"
    duration_seconds: float = 10.0
    frame_rate: float = 24.0
    block_size: int = 6
    window_blocks: int = 4
    model_checkpoint: str = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-22b-distilled.safetensors"
    text_encoder_path: str = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/gemma"
    replay_sample_path: str = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/data_distilled/00000.pt"
    manifest_path: str = "ode/switch_episodes_smoke.jsonl"
    episode_id: str = ""
    output_path: str = "outputs/streaming_switch_demo.mp4"
    height: int = OFFICIAL_SMALL_RESOLUTION.height
    width: int = OFFICIAL_SMALL_RESOLUTION.width
    chunk_num_frames: int = 17
    num_inference_steps: int = 12
    guidance_scale: float = 1.0
    stg_scale: float = 0.0
    chunks_per_segment: int = 1
    reference_window_chunks: int = 1
    reference_max_frames: int = 17
    reference_downscale_factor: int = 1
    switch_recache_enabled: bool = True
    switch_recache_window_chunks: int = 1
    switch_recache_max_frames: int = 17
    ssm_streaming_enabled: bool = False
    ssm_checkpoint_path: str = ""
    ssm_d_state: int = 64
    ssm_gate_bias: float = -2.0
    ssm_switch_state_decay: float = 1.0
    ssm_disable_post_chunk_compression: bool = False
    prompt_cache_device: str = "cpu"
    prompt_cache_load_in_8bit: bool = False
    skip_audio: bool = False
    plan_only: bool = False
    overwrite: bool = False
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"


def compute_block_frame_spans(total_latent_frames: int, block_size: int) -> list[tuple[int, int]]:
    if total_latent_frames <= 0:
        return []
    spans = [(0, min(1, total_latent_frames))]
    offset = spans[0][1]
    while offset < total_latent_frames:
        end = min(offset + block_size, total_latent_frames)
        spans.append((offset, end))
        offset = end
    return spans


def select_reference_video(
    generated_videos: list[torch.Tensor],
    *,
    window_chunks: int,
    max_frames: int | None = None,
) -> torch.Tensor | None:
    if window_chunks <= 0 or not generated_videos:
        return None
    recent_videos = generated_videos[-window_chunks:]
    reference = torch.cat([video.permute(1, 0, 2, 3) for video in recent_videos], dim=0)
    if max_frames is not None and max_frames > 0 and reference.shape[0] > max_frames:
        reference = reference[-max_frames:]
    return reference


def select_switch_recache_video(
    generated_videos: list[torch.Tensor],
    *,
    window_chunks: int,
    max_frames: int | None = None,
) -> torch.Tensor | None:
    reference = select_reference_video(
        generated_videos,
        window_chunks=window_chunks,
        max_frames=max_frames,
    )
    if reference is None:
        return None
    valid_frames = ((reference.shape[0] - 1) // 8) * 8 + 1
    if valid_frames <= 1:
        return None
    if valid_frames != reference.shape[0]:
        reference = reference[-valid_frames:]
    return reference


def apply_ssm_switch_state_decay(stream_state: Any, decay: float) -> bool:
    if decay < 0:
        raise ValueError(f"ssm_switch_state_decay must be non-negative, got {decay}")
    if decay >= 1.0 or stream_state is None:
        return False

    ssm_state = getattr(stream_state, "ssm_state", None)
    if ssm_state is None:
        return False

    scale_fn = getattr(ssm_state, "scale_", None)
    if callable(scale_fn):
        scale_fn(decay)
        return True

    states = getattr(ssm_state, "states", None)
    if isinstance(states, dict):
        applied = False
        for key, value in states.items():
            if torch.is_tensor(value):
                states[key] = value * decay
                applied = True
        return applied

    return False


def stitch_generated_chunks(
    *,
    videos: list[torch.Tensor],
    audios: list[torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not videos:
        raise ValueError("videos must not be empty")
    stitched_video = torch.cat(videos, dim=1)
    valid_audios = [audio for audio in audios if audio is not None]
    if not valid_audios:
        return stitched_video, None
    stitched_audio = torch.cat(valid_audios, dim=1)
    return stitched_video, stitched_audio


def snapshot_media_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


def summarize_stream_state(stream_state: Any) -> dict[str, Any]:
    ssm_state = getattr(stream_state, "ssm_state", None)
    states = getattr(ssm_state, "states", None)
    pending_video_chunks = getattr(stream_state, "pending_video_chunks", None)
    pending_audio_chunks = getattr(stream_state, "pending_audio_chunks", None)
    pending_video_chunk_count = len(pending_video_chunks) if isinstance(pending_video_chunks, list) else 0
    pending_audio_chunk_count = len(pending_audio_chunks) if isinstance(pending_audio_chunks, list) else 0
    if not isinstance(states, dict) or not states:
        return {
            "num_tensors": 0,
            "total_numel": 0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "pending_video_chunks": pending_video_chunk_count,
            "pending_audio_chunks": pending_audio_chunk_count,
        }

    total_numel = 0
    abs_sum = 0.0
    max_abs = 0.0
    for tensor in states.values():
        if not torch.is_tensor(tensor):
            continue
        total_numel += tensor.numel()
        abs_sum += float(tensor.abs().sum())
        max_abs = max(max_abs, float(tensor.abs().max()))

    mean_abs = abs_sum / total_numel if total_numel > 0 else 0.0
    return {
        "num_tensors": len(states),
        "total_numel": total_numel,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "pending_video_chunks": pending_video_chunk_count,
        "pending_audio_chunks": pending_audio_chunk_count,
    }


def build_switch_generation_plan(
    *,
    episode: SwitchEpisode,
    chunks_per_segment: int,
    chunk_num_frames: int,
    frame_rate: float,
    reference_window_chunks: int,
    reference_max_frames: int,
    seed: int = 42,
) -> dict[str, Any]:
    if chunks_per_segment <= 0:
        raise ValueError("chunks_per_segment must be positive")

    chunks: list[dict[str, Any]] = []
    chunk_idx = 0
    for segment_idx, segment in enumerate(episode.segments):
        for repeat_idx in range(chunks_per_segment):
            previous_prompt = chunks[-1]["prompt"] if chunks else None
            chunks.append(
                {
                    "chunk_idx": chunk_idx,
                    "segment_idx": segment_idx,
                    "repeat_idx": repeat_idx,
                    "prompt": segment.prompt,
                    "category": segment.category,
                    "start_seconds": segment.start_seconds,
                    "duration_seconds": segment.duration_seconds,
                    "prompt_switch": previous_prompt is not None and previous_prompt != segment.prompt,
                    "seed": seed + chunk_idx,
                }
            )
            chunk_idx += 1

    return {
        "episode_id": episode.episode_id,
        "num_chunks": len(chunks),
        "chunk_num_frames": chunk_num_frames,
        "frame_rate": frame_rate,
        "reference_window_chunks": reference_window_chunks,
        "reference_max_frames": reference_max_frames,
        "chunks": chunks,
    }


def build_chunk_output_path(output_path: Path, chunk_idx: int) -> Path:
    chunk_dir = output_path.parent / output_path.stem
    return chunk_dir / f"chunk_{chunk_idx:03d}.mp4"


def write_metadata(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def resolve_switch_episode(config: StreamingConfig, episodes: list[SwitchEpisode]) -> SwitchEpisode:
    if not episodes:
        raise ValueError(f"No switch episodes found in manifest: {config.manifest_path}")
    if config.episode_id:
        episode = select_switch_episode_by_id(episodes, config.episode_id)
        if episode is None:
            raise ValueError(f"Episode {config.episode_id!r} not found in manifest: {config.manifest_path}")
        return episode
    return episodes[0]


def decode_latents_to_mp4(
    *,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    output_path: Path,
    checkpoint_path: Path,
    fps: int,
    device: torch.device,
) -> None:
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.model.video_vae import (
        TilingConfig,
        decode_video as vae_decode_video,
        get_video_chunks_number,
    )
    from ltx_core.types import VideoLatentShape
    from ltx_pipelines.utils import ModelLedger
    from ltx_pipelines.utils.media_io import encode_video

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tiling_config = TilingConfig.default()
    ledger = ModelLedger(dtype=dtype, device=device, checkpoint_path=str(checkpoint_path))
    decoder = ledger.video_decoder()
    audio_decoder = ledger.audio_decoder()
    vocoder = ledger.vocoder()
    try:
        with torch.inference_mode():
            video = vae_decode_video(video_latent.to(device=device, dtype=dtype), decoder, tiling_config=tiling_config)
            audio = vae_decode_audio(audio_latent.to(device=device, dtype=dtype), audio_decoder, vocoder)
            num_frames = VideoLatentShape.from_torch_shape(video_latent.shape).upscale().frames
            video_chunks = get_video_chunks_number(num_frames=num_frames, tiling_config=tiling_config)
            encode_video(
                video=video,
                fps=fps,
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=video_chunks,
            )
    finally:
        decoder.to("cpu")
        audio_decoder.to("cpu")
        vocoder.to("cpu")
        del decoder
        del audio_decoder
        del vocoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_replay_mode(config: StreamingConfig) -> None:
    sample_path = Path(config.replay_sample_path).resolve()
    checkpoint_path = Path(config.model_checkpoint).resolve()
    output_path = Path(config.output_path).resolve()
    payload = torch.load(sample_path, map_location="cpu", weights_only=True)

    video_latent = payload["stage2_video_traj"][-1].unsqueeze(0)
    audio_latent = payload["stage2_audio_traj"][-1].unsqueeze(0)
    prompt = payload.get("prompt", config.prompt)

    total_latent_frames = video_latent.shape[2]
    video_spans = compute_block_frame_spans(total_latent_frames, config.block_size)
    audio_spans = split_uniform_spans(audio_latent.shape[2], len(video_spans))

    replay_blocks: list[dict[str, Any]] = []
    live_window: list[int] = []
    evicted_blocks: list[int] = []

    for block_idx, ((v_start, v_end), (a_start, a_end)) in enumerate(zip(video_spans, audio_spans)):
        block = {
            "block_idx": block_idx,
            "video_span": [v_start, v_end],
            "audio_span": [a_start, a_end],
            "video_latents": video_latent[:, :, v_start:v_end, :, :].clone(),
            "audio_latents": audio_latent[:, :, a_start:a_end, :].clone(),
        }
        replay_blocks.append(block)
        live_window.append(block_idx)
        if len(live_window) > config.window_blocks:
            evicted_blocks.append(live_window.pop(0))

    stitched_video = torch.cat([block["video_latents"] for block in replay_blocks], dim=2)
    stitched_audio = torch.cat([block["audio_latents"] for block in replay_blocks], dim=2)

    if stitched_video.shape != video_latent.shape:
        raise ValueError(f"Replay video shape mismatch: {stitched_video.shape} vs {video_latent.shape}")
    if stitched_audio.shape != audio_latent.shape:
        raise ValueError(f"Replay audio shape mismatch: {stitched_audio.shape} vs {audio_latent.shape}")

    logger.info(
        "Replay mode: sample=%s prompt=%s blocks=%d evicted=%d",
        sample_path.name,
        prompt[:80],
        len(replay_blocks),
        len(evicted_blocks),
    )
    decode_latents_to_mp4(
        video_latent=stitched_video,
        audio_latent=stitched_audio,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        fps=int(config.frame_rate),
        device=torch.device(config.device if torch.cuda.is_available() else "cpu"),
    )

    metadata = {
        "mode": "replay",
        "sample_path": str(sample_path),
        "prompt": prompt,
        "block_spans": [
            {
                "block_idx": block["block_idx"],
                "video_span": block["video_span"],
                "audio_span": block["audio_span"],
            }
            for block in replay_blocks
        ],
        "evicted_blocks": evicted_blocks,
        "window_blocks": config.window_blocks,
        "block_size": config.block_size,
    }
    metadata_path = write_metadata(output_path, metadata)
    logger.info("Saved replay video to %s", output_path)
    logger.info("Saved replay metadata to %s", metadata_path)


def run_switch_mode(config: StreamingConfig) -> None:
    if config.ssm_switch_state_decay < 0:
        raise ValueError(f"ssm_switch_state_decay must be non-negative, got {config.ssm_switch_state_decay}")
    if config.window_blocks < 0:
        raise ValueError(f"window_blocks must be non-negative, got {config.window_blocks}")

    manifest_path = Path(config.manifest_path).resolve()
    output_path = Path(config.output_path).resolve()
    episode = resolve_switch_episode(config, load_switch_episodes(manifest_path))
    plan = build_switch_generation_plan(
        episode=episode,
        chunks_per_segment=config.chunks_per_segment,
        chunk_num_frames=config.chunk_num_frames,
        frame_rate=config.frame_rate,
        reference_window_chunks=config.reference_window_chunks,
        reference_max_frames=config.reference_max_frames,
        seed=config.seed,
    )

    memory_mode = "ssm_streaming" if config.ssm_streaming_enabled else "reference_video"
    metadata: dict[str, Any] = {
        "mode": "switch",
        "manifest_path": str(manifest_path),
        "episode_id": episode.episode_id,
        "output_path": str(output_path),
        "height": config.height,
        "width": config.width,
        "chunk_num_frames": config.chunk_num_frames,
        "frame_rate": config.frame_rate,
        "plan_only": config.plan_only,
        "plan": plan,
        "audio_enabled": not config.skip_audio,
        "memory_mode": memory_mode,
        "ssm_streaming_enabled": config.ssm_streaming_enabled,
        "ssm_checkpoint_path": config.ssm_checkpoint_path,
        "ssm_d_state": config.ssm_d_state,
        "ssm_gate_bias": config.ssm_gate_bias,
        "ssm_switch_state_decay": config.ssm_switch_state_decay,
        "window_blocks": config.window_blocks,
        "ssm_disable_post_chunk_compression": config.ssm_disable_post_chunk_compression,
        "switch_recache_enabled": config.switch_recache_enabled,
        "switch_recache_window_chunks": config.switch_recache_window_chunks,
        "switch_recache_max_frames": config.switch_recache_max_frames,
        "chunk_outputs": [],
    }
    if config.plan_only:
        metadata_path = write_metadata(output_path, metadata)
        logger.info("Saved switch plan metadata to %s", metadata_path)
        return

    from baseline_audit import load_prompt_cache, resolve_device
    from ltx_trainer.model_loader import load_model
    from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler
    from ltx_trainer.video_utils import save_video

    unique_prompts = sorted({chunk["prompt"] for chunk in plan["chunks"]})
    prompt_cache = load_prompt_cache(
        prompts=unique_prompts,
        checkpoint_path=Path(config.model_checkpoint).resolve(),
        text_encoder_path=Path(config.text_encoder_path).resolve(),
        device=resolve_device(config.prompt_cache_device),
        guidance_scale=config.guidance_scale,
        load_in_8bit=config.prompt_cache_load_in_8bit,
    )

    sampler = None
    components = None
    generated_videos: list[torch.Tensor] = []
    generated_audios: list[torch.Tensor | None] = []
    chunk_outputs: list[dict[str, Any]] = []
    stream_state: Any = None
    device_name = config.device if torch.cuda.is_available() else "cpu"
    with_reference_encoder = (
        not config.ssm_streaming_enabled
        and (
            config.reference_window_chunks > 0
            or (config.switch_recache_enabled and config.switch_recache_window_chunks > 0)
        )
    )
    switch_recache_active = (
        not config.ssm_streaming_enabled
        and config.switch_recache_enabled
        and config.switch_recache_window_chunks > 0
        and config.switch_recache_max_frames > 1
    )
    if config.ssm_streaming_enabled and config.switch_recache_enabled:
        logger.info("SSM streaming enabled: switch recache path is bypassed")

    try:
        components = load_model(
            checkpoint_path=str(Path(config.model_checkpoint).resolve()),
            text_encoder_path=None,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=with_reference_encoder,
            with_video_vae_decoder=True,
            with_audio_vae_decoder=not config.skip_audio,
            with_vocoder=not config.skip_audio,
            with_text_encoder=False,
        )
        if config.ssm_streaming_enabled:
            from ltx_core.model.transformer.ssm_integration import SSMAugmentedLTXModel
            from ltx_core.model.transformer.ssm_memory import SSMConfig

            ssm_config = SSMConfig(enabled=True, d_state=config.ssm_d_state, gate_bias=config.ssm_gate_bias)
            components.transformer = SSMAugmentedLTXModel.from_base(components.transformer, ssm_config)
            if config.ssm_checkpoint_path:
                checkpoint_payload = torch.load(Path(config.ssm_checkpoint_path).resolve(), map_location="cpu")
                if isinstance(checkpoint_payload, dict):
                    for candidate_key in ("state_dict", "model_state_dict", "model"):
                        nested = checkpoint_payload.get(candidate_key)
                        if isinstance(nested, dict):
                            checkpoint_payload = nested
                            break
                if not isinstance(checkpoint_payload, dict):
                    raise ValueError("SSM checkpoint must resolve to a state_dict dictionary")
                ssm_layers_state_dict: dict[str, torch.Tensor] = {}
                for key, value in checkpoint_payload.items():
                    if not torch.is_tensor(value):
                        continue
                    if key.startswith("ssm_layers."):
                        ssm_layers_state_dict[key[len("ssm_layers."):]] = value
                    elif ".ssm_layers." in key:
                        ssm_layers_state_dict[key.split(".ssm_layers.", 1)[1]] = value
                if not ssm_layers_state_dict:
                    tensor_only = {
                        key: value
                        for key, value in checkpoint_payload.items()
                        if torch.is_tensor(value)
                    }
                    if any(key and key[0].isdigit() for key in tensor_only):
                        ssm_layers_state_dict = tensor_only
                if not ssm_layers_state_dict:
                    raise ValueError(f"No ssm_layers weights found in checkpoint: {config.ssm_checkpoint_path}")
                missing_keys, unexpected_keys = components.transformer.ssm_layers.load_state_dict(
                    ssm_layers_state_dict,
                    strict=False,
                )
                logger.info(
                    "Loaded SSM checkpoint: path=%s loaded=%d missing=%d unexpected=%d",
                    config.ssm_checkpoint_path,
                    len(ssm_layers_state_dict),
                    len(missing_keys),
                    len(unexpected_keys),
                )

        sampler = ValidationSampler(
            transformer=components.transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=None,
            audio_decoder=components.audio_vae_decoder if not config.skip_audio else None,
            vocoder=components.vocoder if not config.skip_audio else None,
            embeddings_processor=None,
        )

        for chunk in plan["chunks"]:
            chunk_output_path = build_chunk_output_path(output_path, int(chunk["chunk_idx"]))
            reference_video = None
            if not config.ssm_streaming_enabled:
                reference_video = select_reference_video(
                    generated_videos,
                    window_chunks=config.reference_window_chunks,
                    max_frames=config.reference_max_frames,
                )
            reference_source = "history" if reference_video is not None else "none"
            stream_state_enabled = config.ssm_streaming_enabled
            stream_state_returned = False
            switch_recache_frames = 0
            switch_recache_source_frames = 0
            ssm_switch_state_decay_applied = False

            if switch_recache_active and chunk["prompt_switch"]:
                recache_source_video = select_switch_recache_video(
                    generated_videos,
                    window_chunks=config.switch_recache_window_chunks,
                    max_frames=config.switch_recache_max_frames,
                )
                switch_recache_source_frames = int(recache_source_video.shape[0]) if recache_source_video is not None else 0
                if recache_source_video is not None:
                    recache_config = GenerationConfig(
                        prompt=str(chunk["prompt"]),
                        height=config.height,
                        width=config.width,
                        num_frames=switch_recache_source_frames,
                        frame_rate=config.frame_rate,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        seed=int(chunk["seed"]),
                        condition_image=recache_source_video[0],
                        generate_audio=False,
                        cached_embeddings=prompt_cache[str(chunk["prompt"])],
                        stg_scale=config.stg_scale,
                        stg_blocks=[29] if config.stg_scale > 0 else None,
                    )
                    logger.info(
                        "Recaching switch history: episode=%s chunk=%d frames=%d seed=%d",
                        episode.episode_id,
                        chunk["chunk_idx"],
                        switch_recache_source_frames,
                        chunk["seed"],
                    )
                    recached_video, _ = sampler.generate(config=recache_config, device=device_name)
                    reference_video = recached_video.permute(1, 0, 2, 3).contiguous()
                    reference_source = "switch_recache"
                    switch_recache_frames = int(reference_video.shape[0])

            reference_frames = int(reference_video.shape[0]) if reference_video is not None else 0
            generation_kwargs = {
                "prompt": str(chunk["prompt"]),
                "height": config.height,
                "width": config.width,
                "num_frames": config.chunk_num_frames,
                "frame_rate": config.frame_rate,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "seed": int(chunk["seed"]),
                "reference_video": reference_video,
                "reference_downscale_factor": config.reference_downscale_factor,
                "generate_audio": not config.skip_audio,
                "cached_embeddings": prompt_cache[str(chunk["prompt"])],
                "stg_scale": config.stg_scale,
                "stg_blocks": [29] if config.stg_scale > 0 else None,
                "disable_post_chunk_ssm_compression": config.ssm_disable_post_chunk_compression,
            }
            if config.ssm_streaming_enabled:
                generation_kwargs["ssm_window_blocks"] = config.window_blocks

            generation_config = GenerationConfig(
                **generation_kwargs,
            )
            logger.info(
                "Generating switch chunk: episode=%s chunk=%d segment=%d repeat=%d switch=%s seed=%d reference=%s frames=%d",
                episode.episode_id,
                chunk["chunk_idx"],
                chunk["segment_idx"],
                chunk["repeat_idx"],
                chunk["prompt_switch"],
                chunk["seed"],
                reference_source,
                reference_frames,
            )
            if config.ssm_streaming_enabled and chunk["prompt_switch"] and config.ssm_switch_state_decay < 1.0:
                ssm_switch_state_decay_applied = apply_ssm_switch_state_decay(
                    stream_state,
                    config.ssm_switch_state_decay,
                )
            if config.ssm_streaming_enabled:
                video, audio, next_stream_state = sampler.generate(
                    config=generation_config,
                    device=device_name,
                    stream_state=stream_state,
                    return_stream_state=True,
                )
                stream_state = next_stream_state
                stream_state_returned = next_stream_state is not None
                stream_state_stats = summarize_stream_state(stream_state)
                logger.info(
                    "Chunk %d stream_state stats: tensors=%d numel=%d mean_abs=%.6f max_abs=%.6f pending_video_chunks=%d pending_audio_chunks=%d",
                    chunk["chunk_idx"],
                    stream_state_stats["num_tensors"],
                    stream_state_stats["total_numel"],
                    stream_state_stats["mean_abs"],
                    stream_state_stats["max_abs"],
                    stream_state_stats["pending_video_chunks"],
                    stream_state_stats["pending_audio_chunks"],
                )
            else:
                video, audio = sampler.generate(config=generation_config, device=device_name)
                stream_state_stats = None
            audio_sample_rate = None
            if audio is not None and components.vocoder is not None:
                audio_sample_rate = components.vocoder.output_sampling_rate
            save_video(
                video_tensor=video,
                output_path=chunk_output_path,
                fps=config.frame_rate,
                audio=audio,
                audio_sample_rate=audio_sample_rate,
            )
            # ValidationSampler outputs can reuse backing buffers across chunks.
            # Keep an immutable CPU snapshot for history references and final stitching.
            generated_videos.append(snapshot_media_tensor(video))
            generated_audios.append(snapshot_media_tensor(audio))
            chunk_outputs.append(
                {
                    **chunk,
                    "output_path": str(chunk_output_path),
                    "reference_frames": reference_frames,
                    "reference_source": reference_source,
                    "stream_state_enabled": stream_state_enabled,
                    "stream_state_returned": stream_state_returned,
                    "stream_state_stats": stream_state_stats,
                    "switch_recache_source_frames": switch_recache_source_frames,
                    "switch_recache_frames": switch_recache_frames,
                    "ssm_switch_state_decay_applied": ssm_switch_state_decay_applied,
                    "audio_present": audio is not None,
                }
            )

        stitched_video, stitched_audio = stitch_generated_chunks(videos=generated_videos, audios=generated_audios)
        stitched_audio_sample_rate = None
        if stitched_audio is not None and components.vocoder is not None:
            stitched_audio_sample_rate = components.vocoder.output_sampling_rate
        save_video(
            video_tensor=stitched_video,
            output_path=output_path,
            fps=config.frame_rate,
            audio=stitched_audio,
            audio_sample_rate=stitched_audio_sample_rate,
        )
    finally:
        if sampler is not None:
            del sampler
        if components is not None:
            del components
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata["chunk_outputs"] = chunk_outputs
    metadata["stitched_output_path"] = str(output_path)
    metadata_path = write_metadata(output_path, metadata)
    logger.info("Saved switch video to %s", output_path)
    logger.info("Saved switch metadata to %s", metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming AV inference utilities")
    parser.add_argument("--mode", type=str, default="replay", choices=["replay", "switch"])
    parser.add_argument("--prompt", type=str, default="A calm ocean scene with gentle waves")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="outputs/streaming_switch_demo.mp4")
    parser.add_argument(
        "--replay-sample",
        type=str,
        default="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/ode/data_distilled/00000.pt",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/ltx-2.3-22b-distilled.safetensors",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/model/gemma",
    )
    parser.add_argument("--manifest", type=str, default="ode/switch_episodes_smoke.jsonl")
    parser.add_argument("--episode-id", type=str, default="")
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--window-blocks", type=int, default=4)
    parser.add_argument("--height", type=int, default=OFFICIAL_SMALL_RESOLUTION.height)
    parser.add_argument("--width", type=int, default=OFFICIAL_SMALL_RESOLUTION.width)
    parser.add_argument("--chunk-num-frames", type=int, default=17)
    parser.add_argument("--frame-rate", type=float, default=8.0)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--stg-scale", type=float, default=0.0)
    parser.add_argument("--chunks-per-segment", type=int, default=1)
    parser.add_argument("--reference-window-chunks", type=int, default=1)
    parser.add_argument("--reference-max-frames", type=int, default=17)
    parser.add_argument("--reference-downscale-factor", type=int, default=1)
    parser.add_argument("--switch-recache-window-chunks", type=int, default=1)
    parser.add_argument("--switch-recache-max-frames", type=int, default=17)
    parser.add_argument("--disable-switch-recache", action="store_true")
    parser.add_argument("--ssm-streaming", action="store_true")
    parser.add_argument("--ssm-checkpoint", type=str, default="")
    parser.add_argument("--ssm-d-state", type=int, default=64)
    parser.add_argument("--ssm-gate-bias", type=float, default=-2.0)
    parser.add_argument("--ssm-switch-state-decay", type=float, default=1.0)
    parser.add_argument("--disable-ssm-post-compress", action="store_true")
    parser.add_argument("--prompt-cache-device", type=str, default="cpu")
    parser.add_argument("--prompt-cache-load-in-8bit", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StreamingConfig(
        mode=args.mode,
        prompt=args.prompt,
        duration_seconds=args.duration,
        frame_rate=args.frame_rate,
        block_size=args.block_size,
        window_blocks=args.window_blocks,
        model_checkpoint=args.model_checkpoint,
        text_encoder_path=args.text_encoder_path,
        replay_sample_path=args.replay_sample,
        manifest_path=args.manifest,
        episode_id=args.episode_id,
        output_path=args.output,
        height=args.height,
        width=args.width,
        chunk_num_frames=args.chunk_num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        stg_scale=args.stg_scale,
        chunks_per_segment=args.chunks_per_segment,
        reference_window_chunks=args.reference_window_chunks,
        reference_max_frames=args.reference_max_frames,
        reference_downscale_factor=args.reference_downscale_factor,
        switch_recache_enabled=not args.disable_switch_recache,
        switch_recache_window_chunks=args.switch_recache_window_chunks,
        switch_recache_max_frames=args.switch_recache_max_frames,
        ssm_streaming_enabled=args.ssm_streaming,
        ssm_checkpoint_path=args.ssm_checkpoint,
        ssm_d_state=args.ssm_d_state,
        ssm_gate_bias=args.ssm_gate_bias,
        ssm_switch_state_decay=args.ssm_switch_state_decay,
        ssm_disable_post_chunk_compression=args.disable_ssm_post_compress,
        prompt_cache_device=args.prompt_cache_device,
        prompt_cache_load_in_8bit=args.prompt_cache_load_in_8bit,
        skip_audio=args.skip_audio,
        plan_only=args.plan_only,
        overwrite=args.overwrite,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )

    if config.mode == "replay":
        run_replay_mode(config)
        return
    if config.mode == "switch":
        run_switch_mode(config)
        return
    raise ValueError(f"Unsupported mode: {config.mode}")


if __name__ == "__main__":
    main()
