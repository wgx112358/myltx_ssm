from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_switch_manifest import (  # noqa: E402
    PromptRecord,
    build_switch_episodes,
    build_two_segment_switch_episodes,
    load_prompt_texts,
    load_v1_metadata_by_sample_id,
    write_jsonl,
)


def test_build_switch_episodes_changes_category_across_switches() -> None:
    records = [
        PromptRecord(category="Nature", text_prompt="ocean waves"),
        PromptRecord(category="Nature", text_prompt="windy forest"),
        PromptRecord(category="Urban", text_prompt="city subway"),
        PromptRecord(category="Urban", text_prompt="night traffic"),
        PromptRecord(category="Human", text_prompt="chef plating food"),
        PromptRecord(category="Human", text_prompt="drummer solo"),
    ]

    episodes = build_switch_episodes(
        records,
        max_episodes=4,
        segments_per_episode=3,
        seed=7,
    )

    assert len(episodes) == 4
    for episode in episodes:
        assert len(episode.segments) == 3
        categories = [segment.category for segment in episode.segments]
        assert len(set(categories)) == len(categories)


def test_build_switch_episodes_drops_underpopulated_categories() -> None:
    records = [
        PromptRecord(category="Nature", text_prompt="ocean waves"),
        PromptRecord(category="Nature", text_prompt="windy forest"),
        PromptRecord(category="Urban", text_prompt="city subway"),
    ]

    episodes = build_switch_episodes(
        records,
        max_episodes=2,
        segments_per_episode=2,
        seed=3,
    )

    assert episodes
    for episode in episodes:
        assert {segment.category for segment in episode.segments} == {"Nature", "Urban"}


def test_build_two_segment_switch_episodes_longlive_style() -> None:
    episodes = build_two_segment_switch_episodes(
        ["prompt-a", "prompt-b", "prompt-c"],
        max_episodes=5,
        seed=13,
        episode_duration_seconds=15.0,
        switch_choices_seconds=[5.0, 10.0],
    )

    assert len(episodes) == 5
    for episode in episodes:
        assert episode.sample_id is not None
        assert len(episode.segments) == 2
        first, second = episode.segments
        assert first.start_seconds == 0.0
        assert second.start_seconds in {5.0, 10.0}
        assert first.prompt != second.prompt
        total_duration = first.duration_seconds + second.duration_seconds
        assert abs(total_duration - 15.0) < 1e-6


def test_load_prompt_texts_csv_and_default_sample_id_rollover(tmp_path: Path) -> None:
    csv_path = tmp_path / "prompts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "text_prompt"])
        writer.writeheader()
        writer.writerow({"category": "A", "text_prompt": "alpha"})
        writer.writerow({"category": "B", "text_prompt": "beta"})
        writer.writerow({"category": "C", "text_prompt": "gamma"})

    prompts = load_prompt_texts(csv_path)
    assert prompts == ["alpha", "beta", "gamma"]

    episodes = build_two_segment_switch_episodes(
        prompts,
        max_episodes=5,
        seed=0,
        episode_duration_seconds=15.0,
        switch_choices_seconds=[5.0, 10.0],
    )
    assert [episode.sample_id for episode in episodes] == ["00000", "00001", "00002", "00000", "00001"]

    output_path = tmp_path / "manifest.jsonl"
    write_jsonl(episodes, output_path)
    payloads = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 5
    assert payloads[0]["sample_id"] == "00000"
    assert len(payloads[0]["segments"]) == 2
    assert set(payloads[0].keys()) == {"episode_id", "segments", "sample_id"}


def test_build_two_segment_switch_episodes_emits_v1_metadata_when_provided(tmp_path: Path) -> None:
    prompts = ["alpha", "beta", "gamma"]

    metadata_path = tmp_path / "v1_metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "prefix_prompt",
                "switch_prompt",
                "switch_time_seconds",
                "keep_factors",
                "edit_factors",
                "modality_scope",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "00000",
                "prefix_prompt": "alpha-prefix",
                "switch_prompt": "alpha-switch",
                "switch_time_seconds": "6.0",
                "keep_factors": "identity,scene",
                "edit_factors": "speech_voice_emotion",
                "modality_scope": "audio_only",
            }
        )

    metadata = load_v1_metadata_by_sample_id(metadata_path)
    episodes = build_two_segment_switch_episodes(
        prompts,
        max_episodes=3,
        seed=0,
        episode_duration_seconds=15.0,
        switch_choices_seconds=[5.0, 10.0],
        v1_metadata_by_sample_id=metadata,
    )

    assert len(episodes) == 3
    first_episode = episodes[0]
    assert first_episode.sample_id == "00000"
    assert first_episode.prefix_prompt == "alpha-prefix"
    assert first_episode.switch_prompt == "alpha-switch"
    assert first_episode.switch_time_seconds == 6.0
    assert first_episode.keep_factors == ("identity", "scene")
    assert first_episode.edit_factors == ("speech_voice_emotion",)
    assert first_episode.modality_scope == "audio_only"
    assert len(first_episode.segments) == 2
    assert first_episode.segments[0].prompt == "alpha-prefix"
    assert first_episode.segments[1].prompt == "alpha-switch"
    assert first_episode.segments[0].duration_seconds == 6.0
    assert first_episode.segments[1].start_seconds == 6.0

    output_path = tmp_path / "manifest_v1.jsonl"
    write_jsonl(episodes, output_path)
    payloads = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    payload0 = payloads[0]

    assert payload0["sample_id"] == "00000"
    assert payload0["prefix_prompt"] == "alpha-prefix"
    assert payload0["switch_prompt"] == "alpha-switch"
    assert payload0["switch_time_seconds"] == 6.0
    assert payload0["keep_factors"] == ["identity", "scene"]
    assert payload0["edit_factors"] == ["speech_voice_emotion"]
    assert payload0["modality_scope"] == "audio_only"
    assert len(payload0["segments"]) == 2
