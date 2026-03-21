from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_route35_group_manifest import (  # noqa: E402
    GroupSchemaError,
    load_v2_groups,
    write_group_manifest,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_v2_groups_validates_and_normalizes_payload() -> None:
    payload = {
        "group_id": "r35v2_000001",
        "sample_id": "00001",
        "episode_duration_seconds": 15.0,
        "switch_time_seconds": 5.0,
        "shared_prefix_prompt": "same host in same room",
        "shared_keep_factors": ["identity", "scene"],
        "modality_scope": "audio_only",
        "counterfactual_axis": "speech_voice_emotion",
        "branches": [
            {
                "branch_id": "B",
                "branch_prompt": "same host speaks loudly",
                "edit_factors": ["speech_voice_emotion"],
                "touched_modalities": ["audio"],
                "factor_ops": {"speech_voice_emotion": "edit", "identity": "keep"},
                "edited_factor_values": {"speech_voice_emotion": "loud"},
            },
            {
                "branch_id": "A",
                "branch_prompt": "same host whispers",
                "edit_factors": ["speech_voice_emotion"],
                "touched_modalities": ["audio"],
                "factor_ops": {"speech_voice_emotion": "edit", "identity": "keep"},
                "edited_factor_values": {"speech_voice_emotion": "whisper"},
            },
        ],
    }

    groups = load_v2_groups_from_rows([payload])

    assert len(groups) == 1
    row = groups[0]
    assert row["group_id"] == "r35v2_000001"
    assert row["sample_id"] == "00001"
    assert row["switch_time_seconds"] == 5.0
    assert row["shared_prefix_prompt"] == "same host in same room"
    assert row["shared_keep_factors"] == ["identity", "scene"]
    assert row["modality_scope"] == "audio_only"
    assert row["counterfactual_axis"] == "speech_voice_emotion"

    # Builder normalizes deterministic branch order.
    assert [branch["branch_id"] for branch in row["branches"]] == ["A", "B"]


def test_load_v2_groups_raises_on_invalid_modality_scope() -> None:
    invalid_payload = {
        "group_id": "r35v2_bad",
        "sample_id": "00001",
        "episode_duration_seconds": 15.0,
        "switch_time_seconds": 5.0,
        "shared_prefix_prompt": "x",
        "shared_keep_factors": ["identity"],
        "modality_scope": "invalid_scope",
        "branches": [
            {
                "branch_id": "A",
                "branch_prompt": "x",
                "edit_factors": ["camera"],
                "touched_modalities": ["video"],
                "factor_ops": {"camera": "edit"},
            },
            {
                "branch_id": "B",
                "branch_prompt": "y",
                "edit_factors": ["camera"],
                "touched_modalities": ["video"],
                "factor_ops": {"camera": "edit"},
            },
        ],
    }

    with pytest.raises(GroupSchemaError):
        load_v2_groups_from_rows([invalid_payload])


def test_write_group_manifest_roundtrip(tmp_path: Path) -> None:
    payload = {
        "group_id": "r35v2_000003",
        "sample_id": "00003",
        "episode_duration_seconds": 15.0,
        "switch_time_seconds": 6.0,
        "shared_prefix_prompt": "same chef in kitchen",
        "shared_keep_factors": ["identity", "scene", "camera"],
        "modality_scope": "both",
        "branches": [
            {
                "branch_id": "A",
                "branch_prompt": "same chef, warm look",
                "edit_factors": ["style_look"],
                "touched_modalities": ["video"],
                "factor_ops": {"style_look": "edit"},
            },
            {
                "branch_id": "B",
                "branch_prompt": "same chef, cool look",
                "edit_factors": ["style_look"],
                "touched_modalities": ["video"],
                "factor_ops": {"style_look": "edit"},
            },
        ],
        "split": "train",
    }

    groups = load_v2_groups_from_rows([payload])
    output_path = tmp_path / "group_manifest.jsonl"
    write_group_manifest(groups, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["group_id"] == "r35v2_000003"
    assert rows[0]["sample_id"] == "00003"
    assert rows[0]["switch_time_seconds"] == 6.0
    assert rows[0]["modality_scope"] == "both"
    assert rows[0]["split"] == "train"
    assert len(rows[0]["branches"]) == 2


def load_v2_groups_from_rows(rows: list[dict]) -> list[dict]:
    tmp_path = Path("/tmp/route35_group_builder_test_input.jsonl")
    _write_jsonl(tmp_path, rows)
    return load_v2_groups(tmp_path)
