#!/usr/bin/env python3
"""Score-only audit for streaming switch outputs.

Expected input layout (from scripts/streaming_inference.py --mode switch):
- <output_dir>/episode_0000.json
- <output_dir>/episode_0000/chunk_000.mp4
- <output_dir>/episode_0000/chunk_001.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from baseline_audit import (
    ClipScorer,
    compute_prompt_margin,
    resolve_device,
    sample_frame_indices,
    tensor_frame_to_pil,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score streaming switch outputs")
    parser.add_argument("--streaming-output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--metric-device", type=str, default="cpu")
    return parser.parse_args()


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def discover_episode_metadata(streaming_output_dir: Path) -> list[Path]:
    return sorted(path for path in streaming_output_dir.glob("episode_*.json") if path.is_file())


def resolve_chunk_output_path(streaming_output_dir: Path, episode_id: str, chunk: dict[str, Any]) -> Path:
    output_path = chunk.get("output_path")
    if isinstance(output_path, str) and output_path:
        return Path(output_path)
    chunk_idx = int(chunk.get("chunk_idx", 0))
    return streaming_output_dir / episode_id / f"chunk_{chunk_idx:03d}.mp4"


def summarize_streaming_episode_metrics(
    *,
    chunk_metrics: list[dict[str, Any]],
    boundary_scores: list[float],
    total_chunks: int,
    audio_present_count: int,
    broken_count: int,
) -> dict[str, float | int]:
    valid_switch_metrics = [
        metric for metric in chunk_metrics if metric.get("prompt_switch") and not metric.get("broken", False)
    ]
    continuity_score = round(mean(boundary_scores), 6) if boundary_scores else 0.0
    switch_response_score = (
        round(mean(float(metric["prompt_score"]) for metric in valid_switch_metrics), 6)
        if valid_switch_metrics
        else 0.0
    )
    switch_response_margin = (
        round(mean(float(metric["prompt_margin"]) for metric in valid_switch_metrics), 6)
        if valid_switch_metrics
        else 0.0
    )
    denominator = total_chunks if total_chunks > 0 else 1
    return {
        "continuity_score": continuity_score,
        "switch_response_score": switch_response_score,
        "switch_response_margin": switch_response_margin,
        "audio_present_rate": round(audio_present_count / denominator, 6),
        "broken_rate": round(broken_count / denominator, 6),
        "num_chunks": total_chunks,
        "num_switch_chunks": len(valid_switch_metrics),
        "num_boundaries": len(boundary_scores),
        "num_broken_chunks": broken_count,
    }


def build_aggregate_summary(episode_results: list[dict[str, Any]]) -> dict[str, float | int]:
    if not episode_results:
        return {
            "continuity_score": 0.0,
            "switch_response_score": 0.0,
            "switch_response_margin": 0.0,
            "audio_present_rate": 0.0,
            "broken_rate": 0.0,
            "num_episodes": 0,
        }

    summaries = [episode["summary"] for episode in episode_results]
    return {
        "continuity_score": round(mean(float(summary["continuity_score"]) for summary in summaries), 6),
        "switch_response_score": round(mean(float(summary["switch_response_score"]) for summary in summaries), 6),
        "switch_response_margin": round(mean(float(summary["switch_response_margin"]) for summary in summaries), 6),
        "audio_present_rate": round(mean(float(summary["audio_present_rate"]) for summary in summaries), 6),
        "broken_rate": round(mean(float(summary["broken_rate"]) for summary in summaries), 6),
        "num_episodes": len(summaries),
    }


def score_single_episode(
    *,
    metadata: dict[str, Any],
    streaming_output_dir: Path,
    scorer: ClipScorer,
    frames_per_clip: int,
) -> dict[str, Any]:
    from ltx_trainer.video_utils import read_video

    episode_id = str(metadata.get("episode_id", "unknown_episode"))
    raw_chunks = metadata.get("chunk_outputs", [])
    if not isinstance(raw_chunks, list):
        raw_chunks = []

    prompts = _ordered_unique([str(chunk.get("prompt", "")) for chunk in raw_chunks])
    text_features = scorer.encode_texts(prompts) if prompts else None
    prompt_to_idx = {prompt: idx for idx, prompt in enumerate(prompts)}

    chunk_metrics: list[dict[str, Any]] = []
    boundary_scores: list[float] = []
    previous_last_feature: torch.Tensor | None = None
    broken_count = 0
    audio_present_count = 0

    for chunk in raw_chunks:
        chunk_idx = int(chunk.get("chunk_idx", len(chunk_metrics)))
        prompt = str(chunk.get("prompt", ""))
        prompt_switch = bool(chunk.get("prompt_switch", False))
        audio_present = bool(chunk.get("audio_present", False))
        if audio_present:
            audio_present_count += 1

        clip_path = resolve_chunk_output_path(streaming_output_dir, episode_id, chunk)
        metric_entry: dict[str, Any] = {
            "chunk_idx": chunk_idx,
            "segment_idx": int(chunk.get("segment_idx", 0)),
            "prompt": prompt,
            "prompt_switch": prompt_switch,
            "audio_present": audio_present,
            "clip_path": str(clip_path),
            "broken": False,
        }

        if text_features is None or prompt not in prompt_to_idx:
            metric_entry["broken"] = True
            metric_entry["error"] = "missing_prompt_text_features"
            broken_count += 1
            previous_last_feature = None
            chunk_metrics.append(metric_entry)
            continue

        if not clip_path.exists():
            metric_entry["broken"] = True
            metric_entry["error"] = "missing_chunk_file"
            broken_count += 1
            previous_last_feature = None
            chunk_metrics.append(metric_entry)
            continue

        try:
            video, _ = read_video(clip_path)
        except Exception as exc:  # pragma: no cover - runtime decode errors are environment-dependent
            metric_entry["broken"] = True
            metric_entry["error"] = f"read_video_failed: {exc}"
            broken_count += 1
            previous_last_feature = None
            chunk_metrics.append(metric_entry)
            continue

        if video.shape[0] <= 0:
            metric_entry["broken"] = True
            metric_entry["error"] = "empty_video"
            broken_count += 1
            previous_last_feature = None
            chunk_metrics.append(metric_entry)
            continue

        frame_indices = sample_frame_indices(video.shape[0], frames_per_clip)
        if not frame_indices:
            metric_entry["broken"] = True
            metric_entry["error"] = "no_sampled_frames"
            broken_count += 1
            previous_last_feature = None
            chunk_metrics.append(metric_entry)
            continue

        sampled_frames = [tensor_frame_to_pil(video[idx]) for idx in frame_indices]
        image_features = scorer.encode_images(sampled_frames)
        similarities = torch.matmul(image_features, text_features.T).mean(dim=0).tolist()

        correct_idx = prompt_to_idx[prompt]
        metric_entry["prompt_score"] = round(float(similarities[correct_idx]), 6)
        metric_entry["prompt_margin"] = compute_prompt_margin(similarities, correct_idx)
        metric_entry["all_prompt_scores"] = [round(float(score), 6) for score in similarities]

        if previous_last_feature is not None:
            boundary_score = round(float(torch.dot(previous_last_feature, image_features[0]).item()), 6)
            boundary_scores.append(boundary_score)
        previous_last_feature = image_features[-1]

        chunk_metrics.append(metric_entry)

    summary = summarize_streaming_episode_metrics(
        chunk_metrics=chunk_metrics,
        boundary_scores=boundary_scores,
        total_chunks=len(raw_chunks),
        audio_present_count=audio_present_count,
        broken_count=broken_count,
    )
    return {
        "episode_id": episode_id,
        "chunk_metrics": chunk_metrics,
        "boundary_scores": boundary_scores,
        "summary": summary,
    }


def score_streaming_switch_outputs(args: argparse.Namespace) -> dict[str, Any]:
    streaming_output_dir = args.streaming_output_dir.resolve()
    metadata_paths = discover_episode_metadata(streaming_output_dir)
    if args.max_episodes > 0:
        metadata_paths = metadata_paths[: args.max_episodes]
    if not metadata_paths:
        raise ValueError(f"No episode metadata found under: {streaming_output_dir}")

    scorer = ClipScorer(args.clip_model, resolve_device(args.metric_device))
    episode_results: list[dict[str, Any]] = []

    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("mode") != "switch" or "chunk_outputs" not in metadata:
            logger.info("Skipping non-switch metadata: %s", metadata_path)
            continue
        logger.info("Scoring streaming episode from %s", metadata_path)
        episode_results.append(
            score_single_episode(
                metadata=metadata,
                streaming_output_dir=streaming_output_dir,
                scorer=scorer,
                frames_per_clip=args.frames_per_clip,
            )
        )

    aggregate = build_aggregate_summary(episode_results)
    return {"episodes": episode_results, "aggregate": aggregate}


def main() -> None:
    args = parse_args()
    payload = {
        "config": {
            "streaming_output_dir": str(args.streaming_output_dir),
            "clip_model": args.clip_model,
            "frames_per_clip": args.frames_per_clip,
            "max_episodes": args.max_episodes,
            "metric_device": args.metric_device,
        },
        **score_streaming_switch_outputs(args),
    }

    output_json = args.output_json
    if output_json is None:
        output_json = args.streaming_output_dir.resolve() / "streaming_score_summary.json"
    else:
        output_json = output_json.resolve()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved streaming switch summary to %s", output_json)
    logger.info("Aggregate summary: %s", payload["aggregate"])


if __name__ == "__main__":
    main()
