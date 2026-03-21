#!/usr/bin/env python3
"""Score long-horizon no-switch streaming outputs with lightweight AV drift proxies."""

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
    parser = argparse.ArgumentParser(description="Score long-horizon no-switch streaming outputs")
    parser.add_argument("--streaming-output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--metric-device", type=str, default="cpu")
    parser.add_argument("--audio-metric-device", type=str, default="cpu")
    parser.add_argument("--audio-target-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-mel-bins", type=int, default=64)
    parser.add_argument("--audio-hop-length", type=int, default=160)
    parser.add_argument("--audio-n-fft", type=int, default=1024)
    return parser.parse_args()


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _mean_or_zero(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _last_or_zero(values: list[float]) -> float:
    return round(float(values[-1]), 6) if values else 0.0


def _normalize_feature(feature: torch.Tensor) -> torch.Tensor | None:
    if feature.numel() <= 0 or not torch.isfinite(feature).all():
        return None
    norm = torch.linalg.vector_norm(feature)
    if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
        return None
    return (feature / norm).detach().cpu()


def discover_episode_metadata(streaming_output_dir: Path) -> list[Path]:
    return sorted(path for path in streaming_output_dir.glob("episode_*.json") if path.is_file())


def resolve_chunk_output_path(streaming_output_dir: Path, episode_id: str, chunk: dict[str, Any]) -> Path:
    output_path = chunk.get("output_path")
    if isinstance(output_path, str) and output_path:
        return Path(output_path)
    chunk_idx = int(chunk.get("chunk_idx", 0))
    return streaming_output_dir / episode_id / f"chunk_{chunk_idx:03d}.mp4"


class AudioFeatureExtractor:
    def __init__(
        self,
        *,
        device: torch.device,
        target_sample_rate: int,
        mel_bins: int,
        hop_length: int,
        n_fft: int,
    ) -> None:
        self._device = device
        self._available = False
        self._processor: Any = None
        self._audio_cls: Any = None
        self._torchaudio: Any = None

        try:
            import torchaudio
            from ltx_core.model.audio_vae.ops import AudioProcessor
            from ltx_core.types import Audio
        except Exception as exc:  # pragma: no cover - dependency availability is runtime-specific
            logger.warning("Audio drift proxies disabled: %s", exc)
            return

        self._torchaudio = torchaudio
        self._audio_cls = Audio
        self._processor = AudioProcessor(
            target_sample_rate=target_sample_rate,
            mel_bins=mel_bins,
            mel_hop_length=hop_length,
            n_fft=n_fft,
        ).to(device)
        self._available = True

    def encode_file(self, clip_path: Path) -> torch.Tensor | None:
        if not self._available:
            return None

        try:
            waveform, sample_rate = self._torchaudio.load(str(clip_path))
        except Exception as exc:  # pragma: no cover - codec availability is runtime-specific
            logger.warning("Failed to load audio from %s: %s", clip_path, exc)
            return None

        if waveform.numel() <= 0:
            return None

        waveform = waveform.to(device=self._device, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        audio = self._audio_cls(waveform=waveform.unsqueeze(0), sampling_rate=int(sample_rate))
        with torch.inference_mode():
            mel = self._processor.waveform_to_mel(audio)
        feature = mel.mean(dim=2).flatten()
        return _normalize_feature(feature)


def summarize_backbone_episode_metrics(
    *,
    chunk_metrics: list[dict[str, Any]],
    visual_anchor_scores: list[float],
    boundary_scores: list[float],
    audio_anchor_scores: list[float],
    audio_boundary_scores: list[float],
    av_alignment_scores: list[float],
    total_chunks: int,
    audio_present_count: int,
    broken_count: int,
    prompt_switch_count: int,
) -> dict[str, float | int]:
    valid_chunk_metrics = [metric for metric in chunk_metrics if not metric.get("broken", False)]
    denominator = total_chunks if total_chunks > 0 else 1

    return {
        "mean_prompt_score": _mean_or_zero([float(metric["prompt_score"]) for metric in valid_chunk_metrics]),
        "mean_prompt_margin": _mean_or_zero([float(metric["prompt_margin"]) for metric in valid_chunk_metrics]),
        "mean_visual_anchor_similarity": _mean_or_zero(visual_anchor_scores),
        "final_visual_anchor_similarity": _last_or_zero(visual_anchor_scores),
        "mean_boundary_similarity": _mean_or_zero(boundary_scores),
        "mean_audio_anchor_similarity": _mean_or_zero(audio_anchor_scores),
        "final_audio_anchor_similarity": _last_or_zero(audio_anchor_scores),
        "mean_audio_boundary_similarity": _mean_or_zero(audio_boundary_scores),
        "mean_av_alignment_proxy": _mean_or_zero(av_alignment_scores),
        "final_av_alignment_proxy": _last_or_zero(av_alignment_scores),
        "audio_present_rate": round(audio_present_count / denominator, 6),
        "broken_rate": round(broken_count / denominator, 6),
        "num_chunks": total_chunks,
        "num_prompt_switches": prompt_switch_count,
        "num_boundaries": len(boundary_scores),
        "num_audio_chunks": len(audio_anchor_scores),
        "num_av_proxy_chunks": len(av_alignment_scores),
        "num_broken_chunks": broken_count,
    }


def build_aggregate_summary(episode_results: list[dict[str, Any]]) -> dict[str, float | int]:
    if not episode_results:
        return {
            "mean_prompt_score": 0.0,
            "mean_prompt_margin": 0.0,
            "mean_visual_anchor_similarity": 0.0,
            "final_visual_anchor_similarity": 0.0,
            "mean_boundary_similarity": 0.0,
            "mean_audio_anchor_similarity": 0.0,
            "final_audio_anchor_similarity": 0.0,
            "mean_audio_boundary_similarity": 0.0,
            "mean_av_alignment_proxy": 0.0,
            "final_av_alignment_proxy": 0.0,
            "audio_present_rate": 0.0,
            "broken_rate": 0.0,
            "num_episodes": 0,
        }

    summaries = [episode["summary"] for episode in episode_results]
    return {
        "mean_prompt_score": _mean_or_zero([float(summary["mean_prompt_score"]) for summary in summaries]),
        "mean_prompt_margin": _mean_or_zero([float(summary["mean_prompt_margin"]) for summary in summaries]),
        "mean_visual_anchor_similarity": _mean_or_zero(
            [float(summary["mean_visual_anchor_similarity"]) for summary in summaries]
        ),
        "final_visual_anchor_similarity": _mean_or_zero(
            [float(summary["final_visual_anchor_similarity"]) for summary in summaries]
        ),
        "mean_boundary_similarity": _mean_or_zero([float(summary["mean_boundary_similarity"]) for summary in summaries]),
        "mean_audio_anchor_similarity": _mean_or_zero(
            [float(summary["mean_audio_anchor_similarity"]) for summary in summaries]
        ),
        "final_audio_anchor_similarity": _mean_or_zero(
            [float(summary["final_audio_anchor_similarity"]) for summary in summaries]
        ),
        "mean_audio_boundary_similarity": _mean_or_zero(
            [float(summary["mean_audio_boundary_similarity"]) for summary in summaries]
        ),
        "mean_av_alignment_proxy": _mean_or_zero([float(summary["mean_av_alignment_proxy"]) for summary in summaries]),
        "final_av_alignment_proxy": _mean_or_zero([float(summary["final_av_alignment_proxy"]) for summary in summaries]),
        "audio_present_rate": _mean_or_zero([float(summary["audio_present_rate"]) for summary in summaries]),
        "broken_rate": _mean_or_zero([float(summary["broken_rate"]) for summary in summaries]),
        "num_episodes": len(summaries),
    }


def score_single_episode(
    *,
    metadata: dict[str, Any],
    streaming_output_dir: Path,
    scorer: ClipScorer,
    audio_extractor: AudioFeatureExtractor,
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
    visual_anchor_scores: list[float] = []
    audio_anchor_scores: list[float] = []
    audio_boundary_scores: list[float] = []
    av_alignment_scores: list[float] = []

    visual_anchor_feature: torch.Tensor | None = None
    previous_last_feature: torch.Tensor | None = None
    audio_anchor_feature: torch.Tensor | None = None
    previous_audio_feature: torch.Tensor | None = None

    broken_count = 0
    audio_present_count = 0
    prompt_switch_count = 0

    for chunk in raw_chunks:
        chunk_idx = int(chunk.get("chunk_idx", len(chunk_metrics)))
        prompt = str(chunk.get("prompt", ""))
        prompt_switch = bool(chunk.get("prompt_switch", False))
        if prompt_switch:
            prompt_switch_count += 1

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
            previous_audio_feature = None
            chunk_metrics.append(metric_entry)
            continue

        if not clip_path.exists():
            metric_entry["broken"] = True
            metric_entry["error"] = "missing_chunk_file"
            broken_count += 1
            previous_last_feature = None
            previous_audio_feature = None
            chunk_metrics.append(metric_entry)
            continue

        try:
            video, _ = read_video(clip_path)
        except Exception as exc:  # pragma: no cover - runtime decode errors are environment-dependent
            metric_entry["broken"] = True
            metric_entry["error"] = f"read_video_failed: {exc}"
            broken_count += 1
            previous_last_feature = None
            previous_audio_feature = None
            chunk_metrics.append(metric_entry)
            continue

        if video.shape[0] <= 0:
            metric_entry["broken"] = True
            metric_entry["error"] = "empty_video"
            broken_count += 1
            previous_last_feature = None
            previous_audio_feature = None
            chunk_metrics.append(metric_entry)
            continue

        frame_indices = sample_frame_indices(video.shape[0], frames_per_clip)
        if not frame_indices:
            metric_entry["broken"] = True
            metric_entry["error"] = "no_sampled_frames"
            broken_count += 1
            previous_last_feature = None
            previous_audio_feature = None
            chunk_metrics.append(metric_entry)
            continue

        sampled_frames = [tensor_frame_to_pil(video[idx]) for idx in frame_indices]
        image_features = scorer.encode_images(sampled_frames)
        chunk_feature = _normalize_feature(image_features.mean(dim=0))
        if chunk_feature is None:
            metric_entry["broken"] = True
            metric_entry["error"] = "invalid_image_features"
            broken_count += 1
            previous_last_feature = None
            previous_audio_feature = None
            chunk_metrics.append(metric_entry)
            continue

        similarities = torch.matmul(image_features, text_features.T).mean(dim=0).tolist()
        correct_idx = prompt_to_idx[prompt]
        metric_entry["prompt_score"] = round(float(similarities[correct_idx]), 6)
        metric_entry["prompt_margin"] = compute_prompt_margin(similarities, correct_idx)
        metric_entry["all_prompt_scores"] = [round(float(score), 6) for score in similarities]

        if visual_anchor_feature is None:
            visual_anchor_feature = chunk_feature
        visual_anchor_similarity = round(float(torch.dot(visual_anchor_feature, chunk_feature).item()), 6)
        metric_entry["visual_anchor_similarity"] = visual_anchor_similarity
        visual_anchor_scores.append(visual_anchor_similarity)

        if previous_last_feature is not None:
            boundary_similarity = round(float(torch.dot(previous_last_feature, image_features[0]).item()), 6)
            metric_entry["boundary_similarity"] = boundary_similarity
            boundary_scores.append(boundary_similarity)
        previous_last_feature = image_features[-1]

        audio_feature = audio_extractor.encode_file(clip_path) if audio_present else None
        if audio_feature is None:
            previous_audio_feature = None
            if audio_present:
                metric_entry["audio_metric_missing"] = True
        else:
            if audio_anchor_feature is None:
                audio_anchor_feature = audio_feature
            audio_anchor_similarity = round(float(torch.dot(audio_anchor_feature, audio_feature).item()), 6)
            metric_entry["audio_anchor_similarity"] = audio_anchor_similarity
            audio_anchor_scores.append(audio_anchor_similarity)

            if previous_audio_feature is not None:
                audio_boundary_similarity = round(float(torch.dot(previous_audio_feature, audio_feature).item()), 6)
                metric_entry["audio_boundary_similarity"] = audio_boundary_similarity
                audio_boundary_scores.append(audio_boundary_similarity)
            previous_audio_feature = audio_feature

            av_alignment_proxy = 1.0 - abs(visual_anchor_similarity - audio_anchor_similarity) / 2.0
            av_alignment_proxy = round(float(max(0.0, min(1.0, av_alignment_proxy))), 6)
            metric_entry["av_alignment_proxy"] = av_alignment_proxy
            av_alignment_scores.append(av_alignment_proxy)

        chunk_metrics.append(metric_entry)

    summary = summarize_backbone_episode_metrics(
        chunk_metrics=chunk_metrics,
        visual_anchor_scores=visual_anchor_scores,
        boundary_scores=boundary_scores,
        audio_anchor_scores=audio_anchor_scores,
        audio_boundary_scores=audio_boundary_scores,
        av_alignment_scores=av_alignment_scores,
        total_chunks=len(raw_chunks),
        audio_present_count=audio_present_count,
        broken_count=broken_count,
        prompt_switch_count=prompt_switch_count,
    )
    return {
        "episode_id": episode_id,
        "chunk_metrics": chunk_metrics,
        "boundary_scores": boundary_scores,
        "visual_anchor_scores": visual_anchor_scores,
        "audio_anchor_scores": audio_anchor_scores,
        "audio_boundary_scores": audio_boundary_scores,
        "av_alignment_scores": av_alignment_scores,
        "summary": summary,
    }


def score_streaming_backbone_outputs(args: argparse.Namespace) -> dict[str, Any]:
    streaming_output_dir = args.streaming_output_dir.resolve()
    metadata_paths = discover_episode_metadata(streaming_output_dir)
    if args.max_episodes > 0:
        metadata_paths = metadata_paths[: args.max_episodes]
    if not metadata_paths:
        raise ValueError(f"No episode metadata found under: {streaming_output_dir}")

    scorer = ClipScorer(args.clip_model, resolve_device(args.metric_device))
    audio_extractor = AudioFeatureExtractor(
        device=resolve_device(args.audio_metric_device),
        target_sample_rate=args.audio_target_sample_rate,
        mel_bins=args.audio_mel_bins,
        hop_length=args.audio_hop_length,
        n_fft=args.audio_n_fft,
    )
    episode_results: list[dict[str, Any]] = []

    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "chunk_outputs" not in metadata:
            logger.info("Skipping metadata without chunk outputs: %s", metadata_path)
            continue
        logger.info("Scoring no-switch backbone episode from %s", metadata_path)
        episode_result = score_single_episode(
            metadata=metadata,
            streaming_output_dir=streaming_output_dir,
            scorer=scorer,
            audio_extractor=audio_extractor,
            frames_per_clip=args.frames_per_clip,
        )
        if int(episode_result["summary"]["num_prompt_switches"]) > 0:
            logger.warning(
                "Episode %s contains %d prompt switches; backbone audit is meant for no-switch runs",
                episode_result["episode_id"],
                episode_result["summary"]["num_prompt_switches"],
            )
        episode_results.append(episode_result)

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
            "audio_metric_device": args.audio_metric_device,
            "audio_target_sample_rate": args.audio_target_sample_rate,
            "audio_mel_bins": args.audio_mel_bins,
            "audio_hop_length": args.audio_hop_length,
            "audio_n_fft": args.audio_n_fft,
        },
        **score_streaming_backbone_outputs(args),
    }

    output_json = args.output_json
    if output_json is None:
        output_json = args.streaming_output_dir.resolve() / "streaming_backbone_summary.json"
    else:
        output_json = output_json.resolve()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved streaming backbone summary to %s", output_json)
    logger.info("Aggregate summary: %s", payload["aggregate"])


if __name__ == "__main__":
    main()
