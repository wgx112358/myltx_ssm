from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_audit import (  # noqa: E402
    compute_prompt_margin,
    parse_args,
    resolve_device,
    sample_frame_indices,
    summarize_episode_metrics,
    trim_episodes,
)
from official_generation_defaults import get_official_2_stage_resolution  # noqa: E402


def test_trim_episodes_limits_without_reordering() -> None:
    episodes = [{"episode_id": f"episode_{idx:04d}"} for idx in range(5)]

    trimmed = trim_episodes(episodes, max_episodes=3)

    assert [episode["episode_id"] for episode in trimmed] == ["episode_0000", "episode_0001", "episode_0002"]


def test_sample_frame_indices_spreads_evenly_and_keeps_endpoints() -> None:
    indices = sample_frame_indices(num_frames=10, max_samples=4)

    assert indices == [0, 3, 6, 9]


def test_sample_frame_indices_returns_all_frames_when_short() -> None:
    assert sample_frame_indices(num_frames=3, max_samples=8) == [0, 1, 2]


def test_compute_prompt_margin_uses_best_incorrect_prompt() -> None:
    margin = compute_prompt_margin([0.2, 0.7, 0.4], correct_index=1)

    assert margin == 0.3


def test_summarize_episode_metrics_aggregates_segment_and_boundary_scores() -> None:
    summary = summarize_episode_metrics(
        segment_metrics=[
            {"prompt_score": 0.8, "prompt_margin": 0.3},
            {"prompt_score": 0.6, "prompt_margin": 0.1},
            {"prompt_score": 0.7, "prompt_margin": 0.2},
        ],
        boundary_scores=[0.5, 0.7],
    )

    assert summary == {
        "mean_prompt_score": 0.7,
        "mean_prompt_margin": 0.2,
        "mean_boundary_similarity": 0.6,
        "num_segments": 3,
        "num_boundaries": 2,
    }


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr("baseline_audit.torch.cuda.is_available", lambda: False)

    assert resolve_device("cuda").type == "cpu"
    assert resolve_device("cpu").type == "cpu"


def test_compute_prompt_margin_returns_zero_for_single_prompt() -> None:
    assert compute_prompt_margin([0.42], correct_index=0) == 0.0


def test_parse_args_defaults_to_smaller_official_resolution(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    checkpoint = tmp_path / "model.safetensors"
    text_encoder = tmp_path / "gemma"
    output_dir = tmp_path / "audit_out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline_audit.py",
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--text-encoder-path",
            str(text_encoder),
            "--output-dir",
            str(output_dir),
        ],
    )

    args = parse_args()
    official_small = get_official_2_stage_resolution("small")

    assert args.height == official_small.height
    assert args.width == official_small.width
