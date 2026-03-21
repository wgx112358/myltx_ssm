from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from self_forcing_data import (  # noqa: E402
    assign_episode_segments_to_chunks,
    build_chunk_prompt_schedule,
    discover_precomputed_sample_ids,
    load_ode_precomputed_sample,
    load_switch_episodes,
    resolve_precomputed_root,
)


def _write_triplet(root: Path, sample_id: str) -> None:
    precomputed = root / ".precomputed"
    (precomputed / "latents").mkdir(parents=True, exist_ok=True)
    (precomputed / "audio_latents").mkdir(parents=True, exist_ok=True)
    (precomputed / "conditions").mkdir(parents=True, exist_ok=True)

    torch.save({
        "latents": torch.zeros(128, 16, 32, 48),
        "num_frames": 16,
        "height": 32,
        "width": 48,
        "fps": 24.0,
        "ode_target_latents": torch.ones(128, 16, 32, 48),
        "ode_sigma": 0.5,
    }, precomputed / "latents" / f"{sample_id}.pt")
    torch.save({
        "latents": torch.zeros(8, 126, 16),
        "num_time_steps": 126,
        "frequency_bins": 16,
        "duration": 5.0,
        "ode_target_latents": torch.ones(8, 126, 16),
        "ode_sigma": 0.5,
    }, precomputed / "audio_latents" / f"{sample_id}.pt")
    torch.save({
        "video_prompt_embeds": torch.zeros(384, 4096),
        "audio_prompt_embeds": torch.zeros(384, 2048),
        "prompt_attention_mask": torch.ones(384, dtype=torch.int64),
    }, precomputed / "conditions" / f"{sample_id}.pt")


def _write_switch_manifest(path: Path, *, include_sample_ids: bool = False, shuffled: bool = False) -> None:
    episodes = [
        {
            "episode_id": "episode_0000",
            "segments": [
                {"category": "A", "prompt": "prompt-a0", "start_seconds": 0.0, "duration_seconds": 2.0},
                {"category": "B", "prompt": "prompt-a1", "start_seconds": 2.0, "duration_seconds": 1.0},
                {"category": "C", "prompt": "prompt-a2", "start_seconds": 3.0, "duration_seconds": 1.0},
            ],
        },
        {
            "episode_id": "episode_0001",
            "segments": [
                {"category": "X", "prompt": "prompt-b0", "start_seconds": 0.0, "duration_seconds": 1.0},
                {"category": "Y", "prompt": "prompt-b1", "start_seconds": 1.0, "duration_seconds": 1.0},
            ],
        },
    ]
    if include_sample_ids:
        episodes[0]["sample_id"] = "00000"
        episodes[1]["sample_id"] = "00001"
    if shuffled:
        episodes = [episodes[1], episodes[0]]

    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode) + "\n")


def test_resolve_and_discover_precomputed_samples(tmp_path: Path) -> None:
    _write_triplet(tmp_path, "0000")
    _write_triplet(tmp_path, "0001")

    precomputed_root = resolve_precomputed_root(tmp_path)
    assert precomputed_root.name == ".precomputed"

    sample_ids = discover_precomputed_sample_ids(tmp_path)
    assert sample_ids == ["0000", "0001"]


def test_load_ode_precomputed_sample_reads_matching_triplet(tmp_path: Path) -> None:
    _write_triplet(tmp_path, "0003")

    sample = load_ode_precomputed_sample(tmp_path, "0003")

    assert sample["latents"]["latents"].shape == (128, 16, 32, 48)
    assert sample["audio_latents"]["latents"].shape == (8, 126, 16)
    assert sample["conditions"]["video_prompt_embeds"].shape == (384, 4096)
    assert sample["latents"]["ode_sigma"] == 0.5


def test_discover_precomputed_sample_ids_skips_incomplete_triplets(tmp_path: Path) -> None:
    _write_triplet(tmp_path, "0000")
    precomputed = resolve_precomputed_root(tmp_path)
    (precomputed / "audio_latents" / "0000.pt").unlink()

    sample_ids = discover_precomputed_sample_ids(tmp_path)

    assert sample_ids == []


def test_assign_episode_segments_to_chunks_uses_duration_weighted_midpoints(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch.jsonl"
    _write_switch_manifest(manifest_path)
    episodes = load_switch_episodes(manifest_path)

    segment_indices = assign_episode_segments_to_chunks(episodes[0], num_chunks=4)

    assert segment_indices == [0, 0, 1, 2]


def test_build_chunk_prompt_schedule_wraps_episode_selection_by_sample_prefix(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch.jsonl"
    _write_switch_manifest(manifest_path)
    episodes = load_switch_episodes(manifest_path)

    schedule = build_chunk_prompt_schedule("00003__step_001", num_chunks=3, episodes=episodes)

    assert schedule is not None
    assert schedule["episode_id"] == "episode_0001"
    assert schedule["prompts"] == ["prompt-b0", "prompt-b1", "prompt-b1"]
    assert schedule["switch_flags"] == [False, True, False]


def test_build_chunk_prompt_schedule_prefers_manifest_sample_id_match_over_order(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch_shuffled.jsonl"
    _write_switch_manifest(manifest_path, include_sample_ids=True, shuffled=True)
    episodes = load_switch_episodes(manifest_path)

    schedule = build_chunk_prompt_schedule("00000__step_000", num_chunks=3, episodes=episodes)

    assert schedule is not None
    assert schedule["episode_id"] == "episode_0000"
    assert schedule["prompts"] == ["prompt-a0", "prompt-a1", "prompt-a2"]


def test_build_chunk_prompt_schedule_falls_back_to_order_when_manifest_sample_id_missing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch_shuffled_no_sample_id.jsonl"
    _write_switch_manifest(manifest_path, include_sample_ids=False, shuffled=True)
    episodes = load_switch_episodes(manifest_path)

    schedule = build_chunk_prompt_schedule("00000__step_000", num_chunks=3, episodes=episodes)

    assert schedule is not None
    assert schedule["episode_id"] == "episode_0001"
    assert schedule["prompts"] == ["prompt-b0", "prompt-b1", "prompt-b1"]
