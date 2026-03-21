from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_precomputed_alignment import validate_alignment  # noqa: E402


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _touch_triplet(export_root: Path, sample_id: str) -> None:
    precomputed = export_root / ".precomputed"
    (precomputed / "latents").mkdir(parents=True, exist_ok=True)
    (precomputed / "audio_latents").mkdir(parents=True, exist_ok=True)
    (precomputed / "conditions").mkdir(parents=True, exist_ok=True)

    (precomputed / "latents" / f"{sample_id}.pt").touch()
    (precomputed / "audio_latents" / f"{sample_id}.pt").touch()
    (precomputed / "conditions" / f"{sample_id}.pt").touch()


def _touch_partial_latents_only(export_root: Path, sample_id: str) -> None:
    precomputed = export_root / ".precomputed"
    (precomputed / "latents").mkdir(parents=True, exist_ok=True)
    (precomputed / "audio_latents").mkdir(parents=True, exist_ok=True)
    (precomputed / "conditions").mkdir(parents=True, exist_ok=True)

    (precomputed / "latents" / f"{sample_id}.pt").touch()


def test_validate_alignment_reports_clean_coverage_with_manifest_duplicates(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch_manifest.jsonl"
    export_root = tmp_path / "export"

    rows = [
        {"episode_id": "e0", "sample_id": "00000"},
        {"episode_id": "e1", "sample_id": "00000"},
        {"episode_id": "e2", "sample_id": "00001"},
    ]
    _write_manifest(manifest_path, rows)
    _touch_triplet(export_root, "00000")
    _touch_triplet(export_root, "00001")

    summary = validate_alignment(manifest_path, export_root)

    assert summary["manifest"]["row_count"] == 3
    assert summary["manifest"]["rows_with_sample_id"] == 3
    assert summary["manifest"]["rows_missing_sample_id"] == 0
    assert summary["manifest"]["unique_sample_id_count"] == 2
    assert summary["manifest"]["duplicate_sample_id_count"] == 1

    assert summary["coverage"]["manifest_missing_in_precomputed_clip_count"] == 0
    assert summary["coverage"]["orphan_precomputed_clip_count"] == 0
    # Backward-compatible aliases.
    assert summary["coverage"]["manifest_missing_in_precomputed_count"] == 0
    assert summary["coverage"]["orphan_precomputed_sample_count"] == 0

    assert summary["precomputed"]["complete_triplet_count"] == 2
    assert summary["alignment_ok"] is True


def test_validate_alignment_reports_missing_orphan_and_partial_triplets(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch_manifest.jsonl"
    export_root = tmp_path / "export"

    rows = [
        {"episode_id": "e0", "sample_id": "00000"},
        {"episode_id": "e1", "sample_id": "00001"},
        {"episode_id": "e2"},
    ]
    _write_manifest(manifest_path, rows)

    _touch_triplet(export_root, "00000")
    _touch_triplet(export_root, "00002")  # orphan complete triplet
    _touch_partial_latents_only(export_root, "00003")

    summary = validate_alignment(manifest_path, export_root)

    assert summary["manifest"]["row_count"] == 3
    assert summary["manifest"]["rows_with_sample_id"] == 2
    assert summary["manifest"]["rows_missing_sample_id"] == 1

    assert summary["coverage"]["manifest_missing_in_precomputed_clip_count"] == 1
    assert summary["coverage"]["manifest_missing_in_precomputed_clip_preview"] == ["00001"]

    assert summary["coverage"]["orphan_precomputed_clip_count"] == 1
    assert summary["coverage"]["orphan_precomputed_clip_preview"] == ["00002"]

    # Backward-compatible aliases.
    assert summary["coverage"]["manifest_missing_in_precomputed_count"] == 1
    assert summary["coverage"]["manifest_missing_in_precomputed_preview"] == ["00001"]
    assert summary["coverage"]["orphan_precomputed_sample_count"] == 1
    assert summary["coverage"]["orphan_precomputed_sample_preview"] == ["00002"]

    assert summary["precomputed"]["complete_triplet_count"] == 2
    assert summary["precomputed"]["partial_triplet_count"] == 1
    assert summary["precomputed"]["partial_triplet_sample_ids_preview"] == ["00003"]

    assert summary["alignment_ok"] is False


def test_validate_alignment_step_suffix_clip_level_match(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch_manifest.jsonl"
    export_root = tmp_path / "export"

    rows = [
        {"episode_id": "e0", "sample_id": "00000"},
        {"episode_id": "e1", "sample_id": "00001"},
    ]
    _write_manifest(manifest_path, rows)

    _touch_triplet(export_root, "00000__step_000")
    _touch_triplet(export_root, "00001__step_001")
    _touch_triplet(export_root, "00002__step_000")  # clip-level orphan

    summary = validate_alignment(manifest_path, export_root)

    # Exact sample-id view does not match when files carry __step_* suffix.
    assert summary["coverage"]["manifest_missing_in_precomputed_exact_count"] == 2
    assert summary["coverage"]["orphan_precomputed_exact_sample_count"] == 3

    # Clip-level view should match manifest IDs.
    assert summary["coverage"]["manifest_missing_in_precomputed_clip_count"] == 0
    assert summary["coverage"]["orphan_precomputed_clip_count"] == 1
    assert summary["coverage"]["orphan_precomputed_clip_preview"] == ["00002"]

    assert summary["precomputed"]["complete_triplet_count"] == 3
    assert summary["precomputed"]["complete_triplet_clip_id_count"] == 3
    assert summary["alignment_ok"] is True
