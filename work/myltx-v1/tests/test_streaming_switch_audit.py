from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from streaming_switch_audit import (  # noqa: E402
    build_aggregate_summary,
    summarize_streaming_episode_metrics,
)


def test_summarize_streaming_episode_metrics_computes_required_scores() -> None:
    summary = summarize_streaming_episode_metrics(
        chunk_metrics=[
            {"prompt_switch": False, "broken": False, "prompt_score": 0.2, "prompt_margin": 0.1},
            {"prompt_switch": True, "broken": False, "prompt_score": 0.8, "prompt_margin": 0.3},
            {"prompt_switch": True, "broken": True},
        ],
        boundary_scores=[0.5, 0.7],
        total_chunks=3,
        audio_present_count=2,
        broken_count=1,
    )

    assert summary == {
        "continuity_score": 0.6,
        "switch_response_score": 0.8,
        "switch_response_margin": 0.3,
        "audio_present_rate": 0.666667,
        "broken_rate": 0.333333,
        "num_chunks": 3,
        "num_switch_chunks": 1,
        "num_boundaries": 2,
        "num_broken_chunks": 1,
    }


def test_build_aggregate_summary_averages_episode_summaries() -> None:
    aggregate = build_aggregate_summary(
        [
            {
                "summary": {
                    "continuity_score": 0.5,
                    "switch_response_score": 0.7,
                    "switch_response_margin": 0.2,
                    "audio_present_rate": 1.0,
                    "broken_rate": 0.0,
                }
            },
            {
                "summary": {
                    "continuity_score": 0.7,
                    "switch_response_score": 0.5,
                    "switch_response_margin": 0.1,
                    "audio_present_rate": 0.5,
                    "broken_rate": 0.25,
                }
            },
        ]
    )

    assert aggregate == {
        "continuity_score": 0.6,
        "switch_response_score": 0.6,
        "switch_response_margin": 0.15,
        "audio_present_rate": 0.75,
        "broken_rate": 0.125,
        "num_episodes": 2,
    }
