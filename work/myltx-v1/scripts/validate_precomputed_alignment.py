#!/usr/bin/env python3
"""Validate alignment between switch manifest sample IDs and precomputed triplets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _clip_sample_id(sample_id: str) -> str:
    return sample_id.split("__step_", maxsplit=1)[0]


def resolve_precomputed_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if root.name == ".precomputed":
        return root
    precomputed = root / ".precomputed"
    if precomputed.exists():
        return precomputed
    return root


def collect_precomputed_index(data_root: str | Path) -> dict[str, Any]:
    precomputed_root = resolve_precomputed_root(data_root)
    latents_dir = precomputed_root / "latents"
    audio_dir = precomputed_root / "audio_latents"
    conditions_dir = precomputed_root / "conditions"

    latents = {path.stem for path in latents_dir.glob("*.pt")} if latents_dir.exists() else set()
    audio = {path.stem for path in audio_dir.glob("*.pt")} if audio_dir.exists() else set()
    conditions = {path.stem for path in conditions_dir.glob("*.pt")} if conditions_dir.exists() else set()

    complete = latents & audio & conditions
    complete_clip_ids = {_clip_sample_id(sample_id) for sample_id in complete}

    return {
        "precomputed_root": str(precomputed_root),
        "latents_count": len(latents),
        "audio_latents_count": len(audio),
        "conditions_count": len(conditions),
        "complete_triplet_count": len(complete),
        "complete_triplet_sample_ids": sorted(complete),
        "complete_triplet_clip_id_count": len(complete_clip_ids),
        "complete_triplet_clip_ids": sorted(complete_clip_ids),
        "latents_only_sample_ids": sorted(latents - audio - conditions),
        "audio_only_sample_ids": sorted(audio - latents - conditions),
        "conditions_only_sample_ids": sorted(conditions - latents - audio),
        "partial_triplet_sample_ids": sorted((latents | audio | conditions) - complete),
    }


def collect_manifest_index(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    rows = 0
    sample_id_counter: Counter[str] = Counter()
    rows_missing_sample_id = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            rows += 1
            sample_id = str(payload.get("sample_id") or "").strip()
            if not sample_id:
                rows_missing_sample_id += 1
                continue
            sample_id_counter[sample_id] += 1

    unique_sample_ids = set(sample_id_counter)
    duplicate_sample_ids = sorted([sample_id for sample_id, count in sample_id_counter.items() if count > 1])

    return {
        "manifest_path": str(path.resolve()),
        "manifest_row_count": rows,
        "manifest_rows_with_sample_id": sum(sample_id_counter.values()),
        "manifest_rows_missing_sample_id": rows_missing_sample_id,
        "manifest_unique_sample_id_count": len(unique_sample_ids),
        "manifest_unique_sample_ids": sorted(unique_sample_ids),
        "manifest_duplicate_sample_ids": duplicate_sample_ids,
    }


def build_alignment_summary(
    manifest_stats: dict[str, Any],
    precomputed_stats: dict[str, Any],
    *,
    preview_limit: int,
) -> dict[str, Any]:
    manifest_ids = set(manifest_stats["manifest_unique_sample_ids"])
    precomputed_exact_ids = set(precomputed_stats["complete_triplet_sample_ids"])
    precomputed_clip_ids = set(precomputed_stats["complete_triplet_clip_ids"])

    missing_exact = sorted(manifest_ids - precomputed_exact_ids)
    missing_clip = sorted(manifest_ids - precomputed_clip_ids)
    orphan_exact = sorted(precomputed_exact_ids - manifest_ids)
    orphan_clip = sorted(precomputed_clip_ids - manifest_ids)

    # Training alignment should accept clip-level IDs when precomputed naming
    # carries per-step suffixes like "<sample_id>__step_000".
    alignment_ok = len(missing_clip) == 0 and manifest_stats["manifest_rows_missing_sample_id"] == 0

    def _preview(items: list[str]) -> list[str]:
        return items[:preview_limit]

    summary = {
        "manifest": {
            "path": manifest_stats["manifest_path"],
            "row_count": manifest_stats["manifest_row_count"],
            "rows_with_sample_id": manifest_stats["manifest_rows_with_sample_id"],
            "rows_missing_sample_id": manifest_stats["manifest_rows_missing_sample_id"],
            "unique_sample_id_count": manifest_stats["manifest_unique_sample_id_count"],
            "duplicate_sample_id_count": len(manifest_stats["manifest_duplicate_sample_ids"]),
            "duplicate_sample_ids_preview": _preview(manifest_stats["manifest_duplicate_sample_ids"]),
        },
        "precomputed": {
            "root": precomputed_stats["precomputed_root"],
            "latents_count": precomputed_stats["latents_count"],
            "audio_latents_count": precomputed_stats["audio_latents_count"],
            "conditions_count": precomputed_stats["conditions_count"],
            "complete_triplet_count": precomputed_stats["complete_triplet_count"],
            "complete_triplet_clip_id_count": precomputed_stats["complete_triplet_clip_id_count"],
            "partial_triplet_count": len(precomputed_stats["partial_triplet_sample_ids"]),
            "partial_triplet_sample_ids_preview": _preview(precomputed_stats["partial_triplet_sample_ids"]),
        },
        "coverage": {
            "manifest_missing_in_precomputed_exact_count": len(missing_exact),
            "manifest_missing_in_precomputed_exact_preview": _preview(missing_exact),
            "manifest_missing_in_precomputed_clip_count": len(missing_clip),
            "manifest_missing_in_precomputed_clip_preview": _preview(missing_clip),
            "orphan_precomputed_exact_sample_count": len(orphan_exact),
            "orphan_precomputed_exact_sample_preview": _preview(orphan_exact),
            "orphan_precomputed_clip_count": len(orphan_clip),
            "orphan_precomputed_clip_preview": _preview(orphan_clip),
            # Backward-compatible aliases using clip-level view.
            "manifest_missing_in_precomputed_count": len(missing_clip),
            "manifest_missing_in_precomputed_preview": _preview(missing_clip),
            "orphan_precomputed_sample_count": len(orphan_clip),
            "orphan_precomputed_sample_preview": _preview(orphan_clip),
        },
        "alignment_ok": alignment_ok,
    }
    return summary


def validate_alignment(manifest_path: str | Path, precomputed_root: str | Path, *, preview_limit: int = 50) -> dict[str, Any]:
    manifest_stats = collect_manifest_index(manifest_path)
    precomputed_stats = collect_precomputed_index(precomputed_root)
    return build_alignment_summary(manifest_stats, precomputed_stats, preview_limit=preview_limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manifest/.precomputed alignment.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--precomputed-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--preview-limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_alignment(
        args.manifest,
        args.precomputed_root,
        preview_limit=max(1, args.preview_limit),
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
