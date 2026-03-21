#!/usr/bin/env python3
"""Build a Route3.5 V2 grouped-branch manifest from verified prompt-pilot artifacts.

This script is intentionally isolated from V1 switch-manifest logic.
It validates required V2 grouped fields and writes a manifest-like JSONL that
future loaders can consume directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_VALID_MODALITY_SCOPE = {"video_only", "audio_only", "both"}
_VALID_TOUCHED_MODALITIES = {"video", "audio", "both"}


class GroupSchemaError(ValueError):
    """Raised when grouped branch payload is malformed."""


def _ensure_nonempty_str(payload: dict[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GroupSchemaError(f"{context}: missing or empty string field '{field}'")
    return value.strip()


def _ensure_number(payload: dict[str, Any], field: str, *, context: str) -> float:
    value = payload.get(field)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise GroupSchemaError(f"{context}: field '{field}' must be numeric") from exc
    return numeric


def _ensure_str_list(payload: dict[str, Any], field: str, *, context: str, allow_empty: bool = False) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise GroupSchemaError(f"{context}: field '{field}' must be a list")
    items = [str(item).strip() for item in value if str(item).strip()]
    if not allow_empty and not items:
        raise GroupSchemaError(f"{context}: field '{field}' must not be empty")
    return items


def _ensure_mapping(payload: dict[str, Any], field: str, *, context: str, allow_missing: bool = True) -> dict[str, Any]:
    value = payload.get(field)
    if value is None and allow_missing:
        return {}
    if not isinstance(value, dict):
        raise GroupSchemaError(f"{context}: field '{field}' must be an object")
    return value


def _normalize_touched_modalities(branch: dict[str, Any], *, context: str) -> list[str]:
    touched = _ensure_str_list(branch, "touched_modalities", context=context)
    for token in touched:
        if token not in _VALID_TOUCHED_MODALITIES:
            raise GroupSchemaError(f"{context}: unsupported touched modality '{token}'")
    return touched


def validate_group_payload(payload: dict[str, Any], *, index: int) -> dict[str, Any]:
    context = f"group[{index}]"

    group_id = _ensure_nonempty_str(payload, "group_id", context=context)
    sample_id = _ensure_nonempty_str(payload, "sample_id", context=context)
    shared_prefix_prompt = _ensure_nonempty_str(payload, "shared_prefix_prompt", context=context)
    episode_duration_seconds = _ensure_number(payload, "episode_duration_seconds", context=context)
    switch_time_seconds = _ensure_number(payload, "switch_time_seconds", context=context)

    if not (0.0 < switch_time_seconds < episode_duration_seconds):
        raise GroupSchemaError(
            f"{context}: switch_time_seconds must satisfy 0 < switch_time_seconds < episode_duration_seconds"
        )

    shared_keep_factors = _ensure_str_list(payload, "shared_keep_factors", context=context)
    modality_scope = _ensure_nonempty_str(payload, "modality_scope", context=context)
    if modality_scope not in _VALID_MODALITY_SCOPE:
        raise GroupSchemaError(f"{context}: unsupported modality_scope '{modality_scope}'")

    branches_raw = payload.get("branches")
    if not isinstance(branches_raw, list) or len(branches_raw) < 2:
        raise GroupSchemaError(f"{context}: field 'branches' must be a list with at least 2 entries")

    normalized_branches: list[dict[str, Any]] = []
    seen_branch_ids: set[str] = set()
    for branch_idx, branch in enumerate(branches_raw):
        branch_context = f"{context}.branch[{branch_idx}]"
        if not isinstance(branch, dict):
            raise GroupSchemaError(f"{branch_context}: branch payload must be an object")

        branch_id = _ensure_nonempty_str(branch, "branch_id", context=branch_context)
        if branch_id in seen_branch_ids:
            raise GroupSchemaError(f"{branch_context}: duplicate branch_id '{branch_id}'")
        seen_branch_ids.add(branch_id)

        branch_prompt = _ensure_nonempty_str(branch, "branch_prompt", context=branch_context)
        edit_factors = _ensure_str_list(branch, "edit_factors", context=branch_context)
        touched_modalities = _normalize_touched_modalities(branch, context=branch_context)
        factor_ops = _ensure_mapping(branch, "factor_ops", context=branch_context, allow_missing=False)
        edited_factor_values = _ensure_mapping(branch, "edited_factor_values", context=branch_context, allow_missing=True)

        normalized_branches.append(
            {
                "branch_id": branch_id,
                "branch_prompt": branch_prompt,
                "edit_factors": edit_factors,
                "touched_modalities": touched_modalities,
                "factor_ops": factor_ops,
                "edited_factor_values": edited_factor_values,
            }
        )

    normalized_branches.sort(key=lambda item: item["branch_id"])

    manifest_row: dict[str, Any] = {
        "group_id": group_id,
        "sample_id": sample_id,
        "episode_duration_seconds": episode_duration_seconds,
        "switch_time_seconds": switch_time_seconds,
        "shared_prefix_prompt": shared_prefix_prompt,
        "shared_keep_factors": shared_keep_factors,
        "modality_scope": modality_scope,
        "branches": normalized_branches,
    }

    # Cheap optional passthrough fields that are useful for future loaders/debugging.
    for optional_field in (
        "shared_factor_values",
        "counterfactual_axis",
        "category",
        "family_id",
        "family_name",
        "split",
    ):
        if optional_field in payload:
            manifest_row[optional_field] = payload[optional_field]

    return manifest_row


def load_v2_groups(input_jsonl: str | Path) -> list[dict[str, Any]]:
    source = Path(input_jsonl)
    groups: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise GroupSchemaError(f"group[{index}]: payload must be an object")
            groups.append(validate_group_payload(payload, index=index))
    return groups


def write_group_manifest(groups: list[dict[str, Any]], output_jsonl: str | Path) -> None:
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Route3.5 grouped-branch manifest JSONL.")
    parser.add_argument("--input-v2-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=0, help="If >0, only emit the first N validated groups")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = load_v2_groups(args.input_v2_jsonl)

    if args.max_groups > 0:
        groups = groups[: args.max_groups]

    write_group_manifest(groups, args.output)
    print(f"wrote {len(groups)} grouped manifest rows to {args.output}")


if __name__ == "__main__":
    main()
