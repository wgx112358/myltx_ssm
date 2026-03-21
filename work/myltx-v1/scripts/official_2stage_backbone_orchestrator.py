#!/usr/bin/env python3
"""Official two-stage backbone preview orchestrator for persistent SSM smoke runs."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from ltx_pipelines.distilled_streaming import OfficialDistilledChunkConfig, OfficialDistilledChunkRunner
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

from official_generation_defaults import get_official_2_stage_resolution
from self_forcing_data import SwitchEpisode, load_switch_episodes, select_switch_episode_by_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackbonePreviewConfig:
    manifest_path: str
    episode_id: str
    output_path: str
    distilled_checkpoint_path: str
    gemma_root: str
    spatial_upsampler_path: str
    ssm_checkpoint_path: str = ""
    preset: str = "small"
    chunk_num_frames: int = 121
    frame_rate: float = 24.0
    chunks_per_segment: int = 6
    window_blocks: int = 2
    seed: int = 42
    overwrite: bool = False
    ssm_d_state: int = 64
    ssm_gate_bias: float = -2.0

    @property
    def height(self) -> int:
        return get_official_2_stage_resolution(self.preset).height

    @property
    def width(self) -> int:
        return get_official_2_stage_resolution(self.preset).width


def build_chunk_output_path(output_path: Path, chunk_idx: int) -> Path:
    chunk_dir = output_path.parent / output_path.stem
    return chunk_dir / f"chunk_{chunk_idx:03d}.mp4"


def write_metadata(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def _resolve_episode(config: BackbonePreviewConfig) -> SwitchEpisode:
    episodes = load_switch_episodes(config.manifest_path)
    if not episodes:
        raise ValueError(f"No episodes found in manifest: {config.manifest_path}")
    if config.episode_id:
        episode = select_switch_episode_by_id(episodes, config.episode_id)
        if episode is None:
            raise ValueError(f"Episode {config.episode_id!r} not found in manifest: {config.manifest_path}")
        return episode
    return episodes[0]


def build_backbone_chunk_plan(config: BackbonePreviewConfig) -> dict[str, Any]:
    episode = _resolve_episode(config)
    if len(episode.segments) != 1:
        raise ValueError("Phase-1 backbone preview expects a single no-switch segment in the manifest")

    segment = episode.segments[0]
    chunks = [
        {
            "chunk_idx": chunk_idx,
            "segment_idx": 0,
            "repeat_idx": chunk_idx,
            "prompt": segment.prompt,
            "category": segment.category,
            "prompt_switch": False,
            "seed": config.seed + chunk_idx,
        }
        for chunk_idx in range(config.chunks_per_segment)
    ]
    return {
        "episode_id": episode.episode_id,
        "num_chunks": len(chunks),
        "chunk_num_frames": config.chunk_num_frames,
        "frame_rate": config.frame_rate,
        "chunks_per_segment": config.chunks_per_segment,
        "window_blocks": config.window_blocks,
        "chunks": chunks,
    }


def _snapshot_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.detach().cpu().clone()


def summarize_ssm_state(ssm_state: object | None) -> dict[str, Any]:
    states = getattr(ssm_state, "states", None)
    if not isinstance(states, dict) or not states:
        return {
            "num_tensors": 0,
            "total_numel": 0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
        }

    tensor_values = [value for value in states.values() if torch.is_tensor(value)]
    if not tensor_values:
        return {
            "num_tensors": 0,
            "total_numel": 0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
        }

    total_numel = sum(tensor.numel() for tensor in tensor_values)
    max_abs = max(float(tensor.abs().max()) for tensor in tensor_values)
    mean_abs = mean(float(tensor.abs().mean()) for tensor in tensor_values)
    return {
        "num_tensors": len(tensor_values),
        "total_numel": total_numel,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
    }


def _prepare_output_paths(config: BackbonePreviewConfig) -> Path:
    output_path = Path(config.output_path).resolve()
    chunk_dir = output_path.parent / output_path.stem
    metadata_path = output_path.with_suffix(".json")

    if config.overwrite:
        if output_path.exists():
            output_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
    else:
        if output_path.exists() or metadata_path.exists() or chunk_dir.exists():
            raise FileExistsError(
                f"Output already exists for {output_path}. Pass overwrite=True or --overwrite to replace it."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return output_path


def _resolve_ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _stitch_chunk_files(chunk_paths: list[Path], output_path: Path) -> None:
    if not chunk_paths:
        raise ValueError("chunk_paths must not be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(chunk_paths) == 1:
        shutil.copyfile(chunk_paths[0], output_path)
        return

    concat_file = output_path.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in chunk_paths),
        encoding="utf-8",
    )
    try:
        ffmpeg = _resolve_ffmpeg_executable()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        if concat_file.exists():
            concat_file.unlink()


def run_backbone_preview(config: BackbonePreviewConfig) -> dict[str, Any]:
    if config.window_blocks < 0:
        raise ValueError(f"window_blocks must be non-negative, got {config.window_blocks}")
    output_path = _prepare_output_paths(config)
    plan = build_backbone_chunk_plan(config)
    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path=config.distilled_checkpoint_path,
        gemma_root=config.gemma_root,
        spatial_upsampler_path=config.spatial_upsampler_path,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(config.chunk_num_frames, tiling_config)
    pending_queue: list[dict[str, Any]] = []
    chunk_outputs: list[dict[str, Any]] = []
    chunk_paths: list[Path] = []
    ssm_state: object | None = None

    for chunk in plan["chunks"]:
        chunk_config = OfficialDistilledChunkConfig(
            preset=config.preset,
            distilled_checkpoint_path=config.distilled_checkpoint_path,
            gemma_root=config.gemma_root,
            spatial_upsampler_path=config.spatial_upsampler_path,
            num_frames=config.chunk_num_frames,
            frame_rate=config.frame_rate,
            prompt=str(chunk["prompt"]),
            seed=int(chunk["seed"]),
            ssm_streaming_enabled=True,
            ssm_d_state=config.ssm_d_state,
            ssm_gate_bias=config.ssm_gate_bias,
            ssm_checkpoint_path=config.ssm_checkpoint_path,
        )
        chunk_result = runner.run_chunk(chunk_config, ssm_state=ssm_state)
        ssm_state = chunk_result.next_ssm_state

        pending_queue.append(
            {
                "chunk_idx": int(chunk["chunk_idx"]),
                "video_tokens": _snapshot_tokens(chunk_result.evictable_video_tokens),
                "audio_tokens": _snapshot_tokens(chunk_result.evictable_audio_tokens),
            }
        )

        compression_applied = False
        compressed_chunk_idx: int | None = None
        if len(pending_queue) > config.window_blocks:
            evicted = pending_queue.pop(0)
            ssm_state = runner.compress_evicted_tokens(
                ssm_state,
                evicted["video_tokens"],
                evicted["audio_tokens"],
            )
            compression_applied = True
            compressed_chunk_idx = int(evicted["chunk_idx"])

        chunk_output_path = build_chunk_output_path(output_path, int(chunk["chunk_idx"]))
        chunk_video = chunk_result.final_chunk_video
        chunk_audio = chunk_result.final_chunk_audio
        del chunk_result
        encode_video(
            video=chunk_video,
            fps=int(config.frame_rate),
            audio=chunk_audio,
            output_path=str(chunk_output_path),
            video_chunks_number=video_chunks_number,
        )
        chunk_paths.append(chunk_output_path)

        chunk_outputs.append(
            {
                **chunk,
                "output_path": str(chunk_output_path),
                "pending_queue_length": len(pending_queue),
                "compression_applied": compression_applied,
                "compressed_chunk_idx": compressed_chunk_idx,
                "state_present": ssm_state is not None,
                "state_stats": summarize_ssm_state(ssm_state),
            }
        )
        logger.info(
            "Backbone chunk complete: episode=%s chunk=%d queue=%d compressed=%s state_present=%s",
            plan["episode_id"],
            chunk["chunk_idx"],
            len(pending_queue),
            compression_applied,
            ssm_state is not None,
        )

    _stitch_chunk_files(chunk_paths, output_path)

    metadata = {
        "mode": "backbone_preview",
        "memory_mode": "persistent_ssm",
        "manifest_path": str(Path(config.manifest_path).resolve()),
        "episode_id": plan["episode_id"],
        "output_path": str(output_path),
        "stitched_output_path": str(output_path),
        "preset": config.preset,
        "height": config.height,
        "width": config.width,
        "chunk_num_frames": config.chunk_num_frames,
        "frame_rate": config.frame_rate,
        "chunks_per_segment": config.chunks_per_segment,
        "window_blocks": config.window_blocks,
        "distilled_checkpoint_path": config.distilled_checkpoint_path,
        "gemma_root": config.gemma_root,
        "spatial_upsampler_path": config.spatial_upsampler_path,
        "ssm_checkpoint_path": config.ssm_checkpoint_path,
        "plan": plan,
        "chunk_outputs": chunk_outputs,
        "final_state_present": ssm_state is not None,
        "final_state_stats": summarize_ssm_state(ssm_state),
    }
    write_metadata(output_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official two-stage backbone preview orchestrator")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--episode-id", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--distilled-checkpoint-path", type=str, required=True)
    parser.add_argument("--gemma-root", type=str, required=True)
    parser.add_argument("--spatial-upsampler-path", type=str, required=True)
    parser.add_argument("--ssm-checkpoint", type=str, default="")
    parser.add_argument("--preset", type=str, default="small")
    parser.add_argument("--chunk-num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--chunks-per-segment", type=int, default=6)
    parser.add_argument("--window-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssm-d-state", type=int, default=64)
    parser.add_argument("--ssm-gate-bias", type=float, default=-2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BackbonePreviewConfig(
        manifest_path=args.manifest,
        episode_id=args.episode_id,
        output_path=args.output,
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        ssm_checkpoint_path=args.ssm_checkpoint,
        preset=args.preset,
        chunk_num_frames=args.chunk_num_frames,
        frame_rate=args.frame_rate,
        chunks_per_segment=args.chunks_per_segment,
        window_blocks=args.window_blocks,
        seed=args.seed,
        overwrite=args.overwrite,
        ssm_d_state=args.ssm_d_state,
        ssm_gate_bias=args.ssm_gate_bias,
    )
    run_backbone_preview(config)


if __name__ == "__main__":
    main()
