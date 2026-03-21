#!/usr/bin/env python3
"""Build prompt-switch episode manifests.

Supports two minimally-scoped generation modes:
1) `category_cycle` (existing smoke behavior): sample different categories per episode.
2) `longlive_two_segment`: one sample -> one two-segment switch episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    category: str
    text_prompt: str


@dataclass(frozen=True)
class EpisodeSegment:
    category: str
    prompt: str
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class SwitchEpisode:
    episode_id: str
    segments: list[EpisodeSegment]
    sample_id: str | None = None
    prefix_prompt: str | None = None
    switch_prompt: str | None = None
    switch_time_seconds: float | None = None
    keep_factors: tuple[str, ...] | None = None
    edit_factors: tuple[str, ...] | None = None
    modality_scope: str | None = None


def load_prompt_records(csv_path: str | Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = (row.get("category") or "").strip()
            text_prompt = (row.get("text_prompt") or "").strip()
            if category and text_prompt:
                records.append(PromptRecord(category=category, text_prompt=text_prompt))
    return records


def load_prompt_texts(
    prompt_source: str | Path,
    *,
    prompt_column: str = "text_prompt",
    jsonl_prompt_field: str = "prompt",
) -> list[str]:
    source_path = Path(prompt_source)
    suffix = source_path.suffix.lower()

    if suffix == ".csv":
        prompts: list[str] = []
        with source_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or prompt_column not in reader.fieldnames:
                raise ValueError(f"CSV missing prompt column: {prompt_column}")
            for row in reader:
                text_prompt = (row.get(prompt_column) or "").strip()
                if text_prompt:
                    prompts.append(text_prompt)
    elif suffix == ".txt":
        prompts = [line.strip() for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".jsonl":
        prompts = []
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, str):
                    text_prompt = payload.strip()
                else:
                    text_prompt = str(
                        payload.get(jsonl_prompt_field, payload.get(prompt_column, payload.get("text", "")))
                    ).strip()
                if text_prompt:
                    prompts.append(text_prompt)
    else:
        raise ValueError(f"Unsupported prompt source suffix: {suffix}")

    if not prompts:
        raise ValueError(f"No prompts loaded from {source_path}")
    return prompts


def parse_switch_choices_seconds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("switch_choices_seconds must contain at least one value")
    return values


def _parse_factor_list(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, list | tuple):
        values = [str(item).strip() for item in raw if str(item).strip()]
        return tuple(values) if values else None

    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if str(item).strip()]
            return tuple(values) if values else None

    values = [item.strip() for item in text.split(",") if item.strip()]
    return tuple(values) if values else None


def load_v1_metadata_by_sample_id(metadata_source: str | Path) -> dict[str, dict[str, Any]]:
    source_path = Path(metadata_source)
    suffix = source_path.suffix.lower()

    rows: list[dict[str, Any]] = []
    if suffix == ".csv":
        with source_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    elif suffix == ".jsonl":
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    else:
        raise ValueError(f"Unsupported V1 metadata source suffix: {suffix}")

    metadata_by_sample_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue

        switch_time_raw = row.get("switch_time_seconds")
        switch_time: float | None = None
        if switch_time_raw is not None and str(switch_time_raw).strip() != "":
            switch_time = float(switch_time_raw)

        modality_scope_raw = row.get("modality_scope")
        modality_scope = str(modality_scope_raw).strip() if modality_scope_raw is not None else None
        if modality_scope == "":
            modality_scope = None

        metadata_by_sample_id[sample_id] = {
            "prefix_prompt": (str(row.get("prefix_prompt") or "").strip() or None),
            "switch_prompt": (str(row.get("switch_prompt") or "").strip() or None),
            "switch_time_seconds": switch_time,
            "keep_factors": _parse_factor_list(row.get("keep_factors")),
            "edit_factors": _parse_factor_list(row.get("edit_factors")),
            "modality_scope": modality_scope,
        }

    return metadata_by_sample_id


def build_switch_episodes(
    records: list[PromptRecord],
    *,
    max_episodes: int,
    segments_per_episode: int,
    seed: int,
    segment_duration_seconds: float = 5.0,
) -> list[SwitchEpisode]:
    if max_episodes <= 0:
        return []
    if segments_per_episode < 2:
        raise ValueError("segments_per_episode must be at least 2")

    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record.text_prompt)

    categories = [category for category, prompts in grouped.items() if prompts]
    if len(categories) < segments_per_episode:
        raise ValueError(
            f"Need at least {segments_per_episode} categories, found {len(categories)}"
        )

    for prompts in grouped.values():
        rng.shuffle(prompts)

    cursors = {category: 0 for category in categories}
    episodes: list[SwitchEpisode] = []
    for episode_idx in range(max_episodes):
        selected_categories = rng.sample(categories, k=segments_per_episode)
        segments: list[EpisodeSegment] = []
        start_seconds = 0.0
        for category in selected_categories:
            prompts = grouped[category]
            prompt = prompts[cursors[category] % len(prompts)]
            cursors[category] += 1
            segments.append(
                EpisodeSegment(
                    category=category,
                    prompt=prompt,
                    start_seconds=start_seconds,
                    duration_seconds=segment_duration_seconds,
                )
            )
            start_seconds += segment_duration_seconds
        episodes.append(
            SwitchEpisode(
                episode_id=f"episode_{episode_idx:04d}",
                segments=segments,
            )
        )

    return episodes


def _choose_second_prompt(
    prompts: list[str],
    *,
    first_index: int,
    rng: random.Random,
) -> str:
    if len(prompts) < 2:
        raise ValueError("longlive_two_segment requires at least two prompts")

    first_prompt = prompts[first_index]
    candidate_indices = list(range(len(prompts)))
    rng.shuffle(candidate_indices)
    for idx in candidate_indices:
        if idx == first_index:
            continue
        if prompts[idx] != first_prompt:
            return prompts[idx]

    raise ValueError("Could not find a distinct second prompt")


def build_two_segment_switch_episodes(
    prompts: list[str],
    *,
    max_episodes: int,
    seed: int,
    episode_duration_seconds: float,
    switch_choices_seconds: list[float],
    v1_metadata_by_sample_id: dict[str, dict[str, Any]] | None = None,
) -> list[SwitchEpisode]:
    if not prompts:
        return []
    if episode_duration_seconds <= 0:
        raise ValueError("episode_duration_seconds must be > 0")
    if not switch_choices_seconds:
        raise ValueError("switch_choices_seconds must not be empty")

    for switch_time in switch_choices_seconds:
        if switch_time <= 0 or switch_time >= episode_duration_seconds:
            raise ValueError("Each switch choice must satisfy 0 < switch_time < episode_duration_seconds")

    num_episodes = max_episodes if max_episodes > 0 else len(prompts)
    rng = random.Random(seed)

    episodes: list[SwitchEpisode] = []
    for episode_idx in range(num_episodes):
        sample_index = episode_idx % len(prompts)
        sample_id = f"{sample_index:05d}"

        default_prompt_a = prompts[sample_index]
        default_prompt_b = _choose_second_prompt(prompts, first_index=sample_index, rng=rng)
        switch_time = float(rng.choice(switch_choices_seconds))

        metadata = (v1_metadata_by_sample_id or {}).get(sample_id)
        if metadata is not None:
            metadata_switch_time = metadata.get("switch_time_seconds")
            if metadata_switch_time is not None:
                switch_time = float(metadata_switch_time)
            if switch_time <= 0 or switch_time >= episode_duration_seconds:
                raise ValueError("metadata switch_time_seconds must satisfy 0 < switch_time < episode_duration_seconds")

            prompt_a = str(metadata.get("prefix_prompt") or default_prompt_a)
            prompt_b = str(metadata.get("switch_prompt") or default_prompt_b)

            modality_scope = metadata.get("modality_scope")
            if modality_scope is not None and modality_scope not in {"video_only", "audio_only", "both"}:
                raise ValueError(f"Invalid modality_scope for sample_id={sample_id}: {modality_scope}")

            keep_factors = metadata.get("keep_factors")
            edit_factors = metadata.get("edit_factors")
        else:
            prompt_a = default_prompt_a
            prompt_b = default_prompt_b
            keep_factors = None
            edit_factors = None
            modality_scope = None

        segments = [
            EpisodeSegment(
                category="",
                prompt=prompt_a,
                start_seconds=0.0,
                duration_seconds=switch_time,
            ),
            EpisodeSegment(
                category="",
                prompt=prompt_b,
                start_seconds=switch_time,
                duration_seconds=episode_duration_seconds - switch_time,
            ),
        ]
        episodes.append(
            SwitchEpisode(
                episode_id=f"episode_{episode_idx:04d}",
                segments=segments,
                sample_id=sample_id,
                prefix_prompt=prompt_a if metadata is not None else None,
                switch_prompt=prompt_b if metadata is not None else None,
                switch_time_seconds=switch_time if metadata is not None else None,
                keep_factors=keep_factors,
                edit_factors=edit_factors,
                modality_scope=modality_scope,
            )
        )

    return episodes


def write_jsonl(episodes: list[SwitchEpisode], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            payload = {
                "episode_id": episode.episode_id,
                "segments": [asdict(segment) for segment in episode.segments],
            }
            if episode.sample_id is not None:
                payload["sample_id"] = episode.sample_id
            if episode.prefix_prompt is not None:
                payload["prefix_prompt"] = episode.prefix_prompt
            if episode.switch_prompt is not None:
                payload["switch_prompt"] = episode.switch_prompt
            if episode.switch_time_seconds is not None:
                payload["switch_time_seconds"] = episode.switch_time_seconds
            if episode.keep_factors is not None:
                payload["keep_factors"] = list(episode.keep_factors)
            if episode.edit_factors is not None:
                payload["edit_factors"] = list(episode.edit_factors)
            if episode.modality_scope is not None:
                payload["modality_scope"] = episode.modality_scope
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt-switch episode manifests.")
    parser.add_argument("--mode", choices=["category_cycle", "longlive_two_segment"], default="category_cycle")

    # Existing smoke mode arguments
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--segments-per-episode", type=int, default=3)
    parser.add_argument("--segment-duration-seconds", type=float, default=5.0)

    # LongLive-style mode arguments
    parser.add_argument("--prompt-source", type=Path, default=None)
    parser.add_argument("--episode-duration-seconds", type=float, default=15.0)
    parser.add_argument("--switch-choices-seconds", type=str, default="5.0,10.0")
    parser.add_argument("--prompt-column", type=str, default="text_prompt")
    parser.add_argument("--jsonl-prompt-field", type=str, default="prompt")
    parser.add_argument("--v1-metadata-source", type=Path, default=None)

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "category_cycle":
        if args.input_csv is None:
            raise ValueError("--input-csv is required for mode=category_cycle")
        records = load_prompt_records(args.input_csv)
        episodes = build_switch_episodes(
            records,
            max_episodes=args.max_episodes,
            segments_per_episode=args.segments_per_episode,
            seed=args.seed,
            segment_duration_seconds=args.segment_duration_seconds,
        )
    else:
        prompt_source = args.prompt_source if args.prompt_source is not None else args.input_csv
        if prompt_source is None:
            raise ValueError("--prompt-source (or --input-csv) is required for mode=longlive_two_segment")

        prompts = load_prompt_texts(
            prompt_source,
            prompt_column=args.prompt_column,
            jsonl_prompt_field=args.jsonl_prompt_field,
        )
        switch_choices_seconds = parse_switch_choices_seconds(args.switch_choices_seconds)
        v1_metadata_by_sample_id = (
            load_v1_metadata_by_sample_id(args.v1_metadata_source)
            if args.v1_metadata_source is not None
            else None
        )
        episodes = build_two_segment_switch_episodes(
            prompts,
            max_episodes=args.max_episodes,
            seed=args.seed,
            episode_duration_seconds=args.episode_duration_seconds,
            switch_choices_seconds=switch_choices_seconds,
            v1_metadata_by_sample_id=v1_metadata_by_sample_id,
        )

    write_jsonl(episodes, args.output)
    print(f"wrote {len(episodes)} switch episodes to {args.output}")


if __name__ == "__main__":
    main()
