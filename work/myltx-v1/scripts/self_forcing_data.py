#!/usr/bin/env python3
"""Utilities for loading `.precomputed` ODE-regression triplets for self-forcing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_SAMPLE_INDEX_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class SwitchSegment:
    category: str
    prompt: str
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class SwitchEpisode:
    episode_id: str
    segments: tuple[SwitchSegment, ...]
    sample_id: str | None = None


def resolve_precomputed_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if root.name == ".precomputed":
        return root
    precomputed = root / ".precomputed"
    if precomputed.exists():
        return precomputed
    return root


def discover_precomputed_sample_ids(data_root: str | Path, limit: int | None = None) -> list[str]:
    precomputed_root = resolve_precomputed_root(data_root)
    latents_dir = precomputed_root / "latents"
    audio_dir = precomputed_root / "audio_latents"
    conditions_dir = precomputed_root / "conditions"
    if not latents_dir.exists() or not audio_dir.exists() or not conditions_dir.exists():
        return []

    sample_ids: list[str] = []
    for path in sorted(latents_dir.glob("*.pt")):
        sample_id = path.stem
        if (audio_dir / f"{sample_id}.pt").exists() and (conditions_dir / f"{sample_id}.pt").exists():
            sample_ids.append(sample_id)
            if limit is not None and limit > 0 and len(sample_ids) >= limit:
                break
    return sample_ids


def load_ode_precomputed_sample(data_root: str | Path, sample_id: str) -> dict[str, Any]:
    precomputed_root = resolve_precomputed_root(data_root)
    return {
        "latents": torch.load(precomputed_root / "latents" / f"{sample_id}.pt", map_location="cpu", weights_only=True),
        "audio_latents": torch.load(precomputed_root / "audio_latents" / f"{sample_id}.pt", map_location="cpu", weights_only=True),
        "conditions": torch.load(precomputed_root / "conditions" / f"{sample_id}.pt", map_location="cpu", weights_only=True),
    }


def load_switch_episodes(manifest_path: str | Path) -> list[SwitchEpisode]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []

    episodes: list[SwitchEpisode] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            segments = tuple(
                SwitchSegment(
                    category=str(segment.get("category", "")),
                    prompt=str(segment.get("prompt", "")),
                    start_seconds=float(segment.get("start_seconds", 0.0)),
                    duration_seconds=float(segment.get("duration_seconds", 0.0)),
                )
                for segment in payload.get("segments", [])
                if segment.get("prompt")
            )
            if segments:
                raw_sample_id = payload.get("sample_id")
                sample_id = str(raw_sample_id) if raw_sample_id is not None else None
                episodes.append(
                    SwitchEpisode(
                        episode_id=str(payload.get("episode_id", "")),
                        sample_id=sample_id,
                        segments=segments,
                    )
                )
    return episodes


def _clip_sample_id(sample_id: str) -> str:
    return sample_id.split("__step_", maxsplit=1)[0]


def _match_episode_by_manifest_sample_id(sample_id: str, episodes: list[SwitchEpisode]) -> SwitchEpisode | None:
    # Prefer explicit manifest sample_id mapping when available.
    for episode in episodes:
        if episode.sample_id and episode.sample_id == sample_id:
            return episode

    target_clip_id = _clip_sample_id(sample_id)
    for episode in episodes:
        if episode.sample_id and _clip_sample_id(episode.sample_id) == target_clip_id:
            return episode
    return None


def _extract_sample_index(sample_id: str) -> int:
    clip_id = sample_id.split("__step_", maxsplit=1)[0]
    match = _SAMPLE_INDEX_RE.search(clip_id)
    if match is None:
        return 0
    return int(match.group(1))


def select_switch_episode(sample_id: str, episodes: list[SwitchEpisode]) -> SwitchEpisode | None:
    if not episodes:
        return None

    matched = _match_episode_by_manifest_sample_id(sample_id, episodes)
    if matched is not None:
        return matched

    sample_index = _extract_sample_index(sample_id)
    return episodes[sample_index % len(episodes)]


def select_switch_episode_by_id(episodes: list[SwitchEpisode], episode_id: str) -> SwitchEpisode | None:
    for episode in episodes:
        if episode.episode_id == episode_id:
            return episode
    return None


def assign_episode_segments_to_chunks(episode: SwitchEpisode, num_chunks: int) -> list[int]:
    if num_chunks <= 0 or not episode.segments:
        return []

    durations = [max(float(segment.duration_seconds), 1e-6) for segment in episode.segments]
    total_duration = sum(durations)
    cumulative_end_times: list[float] = []
    running = 0.0
    for duration in durations:
        running += duration
        cumulative_end_times.append(running)

    segment_indices: list[int] = []
    for chunk_idx in range(num_chunks):
        chunk_midpoint = (chunk_idx + 0.5) * total_duration / num_chunks
        segment_index = len(cumulative_end_times) - 1
        for idx, segment_end in enumerate(cumulative_end_times):
            if chunk_midpoint < segment_end:
                segment_index = idx
                break
        segment_indices.append(segment_index)
    return segment_indices


def build_episode_chunk_plan(episode: SwitchEpisode, num_chunks: int) -> dict[str, Any]:
    segment_indices = assign_episode_segments_to_chunks(episode, num_chunks)
    prompts = [episode.segments[index].prompt for index in segment_indices]
    categories = [episode.segments[index].category for index in segment_indices]
    switch_flags = [False]
    switch_flags.extend(
        segment_indices[idx] != segment_indices[idx - 1]
        for idx in range(1, len(segment_indices))
    )
    return {
        "episode_id": episode.episode_id,
        "segment_indices": segment_indices,
        "prompts": prompts,
        "categories": categories,
        "switch_flags": switch_flags,
    }


def build_chunk_prompt_schedule(
    sample_id: str,
    num_chunks: int,
    episodes: list[SwitchEpisode],
) -> dict[str, Any] | None:
    episode = select_switch_episode(sample_id, episodes)
    if episode is None:
        return None
    return build_episode_chunk_plan(episode, num_chunks)


def split_uniform_spans(total_tokens: int, num_chunks: int) -> list[tuple[int, int]]:
    if num_chunks <= 0:
        return []
    base, remainder = divmod(total_tokens, num_chunks)
    spans: list[tuple[int, int]] = []
    offset = 0
    for idx in range(num_chunks):
        size = base + (1 if idx < remainder else 0)
        end = offset + size
        spans.append((offset, end))
        offset = end
    return spans
