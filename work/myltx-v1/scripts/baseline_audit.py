#!/usr/bin/env python3
"""Cheap baseline audit for prompt-switch behavior on the untouched base model.

The audit protocol is intentionally minimal and smoke-friendly:
1. Read prompt-switch episodes from the manifest.
2. Generate a short clip for each segment prompt using the unmodified base model.
3. Score each clip with CLIP against all prompts in the same episode.
4. Report prompt-alignment margin and boundary image-embedding continuity.

This is not a final paper-grade evaluation suite. It is a low-cost baseline audit to
measure whether the base model already handles prompt switches well enough to justify
or falsify further method work.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from official_generation_defaults import get_official_2_stage_resolution
from self_forcing_data import load_switch_episodes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
OFFICIAL_SMALL_RESOLUTION = get_official_2_stage_resolution("small")


def trim_episodes(episodes: list[Any], max_episodes: int) -> list[Any]:
    if max_episodes <= 0:
        return []
    return episodes[:max_episodes]


def sample_frame_indices(num_frames: int, max_samples: int) -> list[int]:
    if num_frames <= 0 or max_samples <= 0:
        return []
    if num_frames <= max_samples:
        return list(range(num_frames))
    if max_samples == 1:
        return [num_frames // 2]
    last_index = num_frames - 1
    return [int(idx * last_index / (max_samples - 1)) for idx in range(max_samples)]


def compute_prompt_margin(similarities: list[float], correct_index: int) -> float:
    correct = float(similarities[correct_index])
    incorrect_scores = [float(score) for idx, score in enumerate(similarities) if idx != correct_index]
    if not incorrect_scores:
        return 0.0
    return round(correct - max(incorrect_scores), 6)


def summarize_episode_metrics(
    segment_metrics: list[dict[str, float]],
    boundary_scores: list[float],
) -> dict[str, float | int]:
    return {
        "mean_prompt_score": round(mean(metric["prompt_score"] for metric in segment_metrics), 6),
        "mean_prompt_margin": round(mean(metric["prompt_margin"] for metric in segment_metrics), 6),
        "mean_boundary_similarity": round(mean(boundary_scores), 6) if boundary_scores else 0.0,
        "num_segments": len(segment_metrics),
        "num_boundaries": len(boundary_scores),
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-switch baseline audit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=4)
    parser.add_argument("--height", type=int, default=OFFICIAL_SMALL_RESOLUTION.height)
    parser.add_argument("--width", type=int, default=OFFICIAL_SMALL_RESOLUTION.width)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--frame-rate", type=float, default=8.0)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--stg-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metric-device", type=str, default="cpu")
    parser.add_argument("--prompt-cache-device", type=str, default="cpu")
    parser.add_argument("--prompt-cache-load-in-8bit", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args()


def tensor_frame_to_pil(frame: torch.Tensor) -> Any:
    from PIL import Image

    array = frame.detach().cpu().clamp(0.0, 1.0).mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def build_output_path(output_dir: Path, episode_id: str, segment_idx: int) -> Path:
    return output_dir / episode_id / f"segment_{segment_idx:02d}.mp4"


def write_plan(output_dir: Path, episodes: list[Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "audit_plan.json"
    plan_payload = {
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "prompts": [segment.prompt for segment in episode.segments],
            }
            for episode in episodes
        ]
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_path


def load_prompt_cache(
    *,
    prompts: list[str],
    checkpoint_path: Path,
    text_encoder_path: Path,
    device: torch.device,
    guidance_scale: float,
    load_in_8bit: bool = False,
) -> dict[str, Any]:
    from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
    from ltx_trainer.validation_sampler import CachedPromptEmbeddings

    logger.info("Loading prompt cache for %d prompts on %s (8bit=%s)", len(prompts), device, load_in_8bit)
    text_encoder = load_text_encoder(
        str(text_encoder_path),
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
    )
    embeddings_processor = load_embeddings_processor(str(checkpoint_path), device=device, dtype=torch.bfloat16)

    try:
        negative_video = None
        negative_audio = None
        cache: dict[str, Any] = {}
        with torch.inference_mode():
            if guidance_scale != 1.0:
                neg_hidden_states, neg_mask = text_encoder.encode("")
                neg_out = embeddings_processor.process_hidden_states(neg_hidden_states, neg_mask)
                negative_video = neg_out.video_encoding[0].cpu().contiguous()
                negative_audio = neg_out.audio_encoding[0].cpu().contiguous()
                del neg_hidden_states, neg_mask, neg_out
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            for prompt_idx, prompt in enumerate(prompts, start=1):
                logger.info("Encoding prompt %d/%d", prompt_idx, len(prompts))
                hidden_states, prompt_mask = text_encoder.encode(prompt)
                prompt_out = embeddings_processor.process_hidden_states(hidden_states, prompt_mask)
                cache[prompt] = CachedPromptEmbeddings(
                    video_context_positive=prompt_out.video_encoding[0].cpu().contiguous(),
                    audio_context_positive=prompt_out.audio_encoding[0].cpu().contiguous(),
                    video_context_negative=negative_video,
                    audio_context_negative=negative_audio,
                )
                del hidden_states, prompt_mask, prompt_out
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        return cache
    finally:
        del text_encoder
        del embeddings_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_segments(args: argparse.Namespace, episodes: list[Any]) -> None:
    from ltx_trainer.model_loader import load_model
    from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler
    from ltx_trainer.video_utils import save_video

    output_dir = args.output_dir.resolve()
    unique_prompts = sorted({segment.prompt for episode in episodes for segment in episode.segments})
    prompt_cache = load_prompt_cache(
        prompts=unique_prompts,
        checkpoint_path=args.checkpoint.resolve(),
        text_encoder_path=args.text_encoder_path.resolve(),
        device=resolve_device(args.prompt_cache_device),
        guidance_scale=args.guidance_scale,
        load_in_8bit=args.prompt_cache_load_in_8bit,
    )

    components = load_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        text_encoder_path=None,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=False,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=not args.skip_audio,
        with_vocoder=not args.skip_audio,
        with_text_encoder=False,
    )
    sampler = ValidationSampler(
        transformer=components.transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=None,
        audio_decoder=components.audio_vae_decoder if not args.skip_audio else None,
        vocoder=components.vocoder if not args.skip_audio else None,
        embeddings_processor=None,
    )

    try:
        for episode_idx, episode in enumerate(episodes):
            segment_seed = args.seed + episode_idx
            for segment_idx, segment in enumerate(episode.segments):
                output_path = build_output_path(output_dir, episode.episode_id, segment_idx)
                if output_path.exists() and not args.overwrite:
                    logger.info("Skipping existing clip: %s", output_path)
                    continue

                gen_config = GenerationConfig(
                    prompt=segment.prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=segment_seed,
                    generate_audio=not args.skip_audio,
                    cached_embeddings=prompt_cache[segment.prompt],
                    stg_scale=args.stg_scale,
                    stg_blocks=[29] if args.stg_scale > 0 else None,
                )
                logger.info(
                    "Generating baseline clip: episode=%s segment=%d seed=%d",
                    episode.episode_id,
                    segment_idx,
                    segment_seed,
                )
                video, audio = sampler.generate(config=gen_config, device=args.device)
                audio_sample_rate = None
                if audio is not None and components.vocoder is not None:
                    audio_sample_rate = components.vocoder.output_sampling_rate
                save_video(
                    video_tensor=video,
                    output_path=output_path,
                    fps=args.frame_rate,
                    audio=audio,
                    audio_sample_rate=audio_sample_rate,
                )
    finally:
        del sampler
        del components
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ClipScorer:
    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import AutoProcessor, CLIPModel

        self._device = device
        local_only = Path(model_name).exists()
        self._processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_only)
        self._model = CLIPModel.from_pretrained(model_name, local_files_only=local_only).to(device)
        self._model.eval()

    @torch.inference_mode()
    def encode_texts(self, prompts: list[str]) -> torch.Tensor:
        inputs = self._processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to(self._device)
        features = self._model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1).cpu()

    @torch.inference_mode()
    def encode_images(self, images: list[Any]) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        features = self._model.get_image_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1).cpu()


def score_segments(args: argparse.Namespace, episodes: list[Any]) -> dict[str, Any]:
    from ltx_trainer.video_utils import read_video

    scorer = ClipScorer(args.clip_model, torch.device(args.metric_device))
    episode_results: list[dict[str, Any]] = []

    for episode in episodes:
        prompts = [segment.prompt for segment in episode.segments]
        text_features = scorer.encode_texts(prompts)
        segment_metrics: list[dict[str, Any]] = []
        boundary_scores: list[float] = []
        boundary_frames: list[torch.Tensor] = []

        for segment_idx, segment in enumerate(episode.segments):
            clip_path = build_output_path(args.output_dir.resolve(), episode.episode_id, segment_idx)
            video, _ = read_video(clip_path)
            frame_indices = sample_frame_indices(video.shape[0], args.frames_per_clip)
            sampled_frames = [tensor_frame_to_pil(video[idx]) for idx in frame_indices]
            image_features = scorer.encode_images(sampled_frames)
            similarities = torch.matmul(image_features, text_features.T).mean(dim=0).tolist()
            segment_metrics.append(
                {
                    "segment_idx": segment_idx,
                    "clip_path": str(clip_path),
                    "prompt_score": round(float(similarities[segment_idx]), 6),
                    "prompt_margin": compute_prompt_margin(similarities, segment_idx),
                    "all_prompt_scores": [round(float(score), 6) for score in similarities],
                }
            )
            boundary_frames.append(image_features[-1])

        for segment_idx in range(len(boundary_frames) - 1):
            next_clip_path = build_output_path(args.output_dir.resolve(), episode.episode_id, segment_idx + 1)
            next_video, _ = read_video(next_clip_path)
            next_first = scorer.encode_images([tensor_frame_to_pil(next_video[0])])[0]
            boundary_score = round(float(torch.dot(boundary_frames[segment_idx], next_first).item()), 6)
            boundary_scores.append(boundary_score)

        summary = summarize_episode_metrics(segment_metrics, boundary_scores)
        episode_results.append(
            {
                "episode_id": episode.episode_id,
                "segment_metrics": segment_metrics,
                "boundary_scores": boundary_scores,
                "summary": summary,
            }
        )

    aggregate = {
        "mean_prompt_score": round(mean(item["summary"]["mean_prompt_score"] for item in episode_results), 6),
        "mean_prompt_margin": round(mean(item["summary"]["mean_prompt_margin"] for item in episode_results), 6),
        "mean_boundary_similarity": round(mean(item["summary"]["mean_boundary_similarity"] for item in episode_results), 6),
        "num_episodes": len(episode_results),
    }
    return {"episodes": episode_results, "aggregate": aggregate}


def main() -> None:
    args = parse_args()
    episodes = trim_episodes(load_switch_episodes(args.manifest), args.max_episodes)
    if not episodes:
        raise ValueError(f"No episodes available from {args.manifest}")

    plan_path = write_plan(args.output_dir.resolve(), episodes)
    logger.info("Wrote audit plan to %s", plan_path)
    if args.plan_only:
        return

    if not args.score_only:
        generate_segments(args, episodes)
    results = score_segments(args, episodes)
    payload = {
        "config": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "text_encoder_path": str(args.text_encoder_path),
            "output_dir": str(args.output_dir),
            "max_episodes": args.max_episodes,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "frame_rate": args.frame_rate,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "stg_scale": args.stg_scale,
            "seed": args.seed,
            "frames_per_clip": args.frames_per_clip,
            "clip_model": args.clip_model,
            "skip_audio": args.skip_audio,
        },
        **results,
    }
    summary_path = args.output_dir.resolve() / "baseline_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved baseline summary to %s", summary_path)
    logger.info("Aggregate summary: %s", payload["aggregate"])


if __name__ == "__main__":
    main()
