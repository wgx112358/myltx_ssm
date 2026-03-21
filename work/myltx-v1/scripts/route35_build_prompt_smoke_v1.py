#!/usr/bin/env python3
"""Build a smoke-size structured prompt dataset for Route 3 (V1) and Route 3.5 (V2).

This script is API-free and uses fixed seed families/templates to unblock
builder/loader integration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_MODALITY_SCOPE = {"audio_only", "video_only", "both"}
ALL_FACTORS = [
    "identity",
    "scene",
    "motion_action",
    "camera",
    "style_look",
    "audio_event",
    "speech_voice_emotion",
    "ambiance_music",
]


@dataclass(frozen=True)
class SeedFamily:
    family_id: str
    category: str
    identity: str
    scene: str
    motion_action: str
    camera: str
    style_look: str
    audio_event: str
    speech_voice_emotion: str
    ambiance_music: str
    edited_factor: str
    modality_scope: str
    branch_a_value: str
    branch_b_value: str


def render_prompt(values: dict[str, str]) -> str:
    template = (
        "%(identity)s in %(scene)s, %(motion_action)s, %(camera)s, %(style_look)s, "
        "with %(audio_event)s, %(speech_voice_emotion)s, over %(ambiance_music)s"
    )
    return template % values


def build_factor_ops(edited_factor: str) -> dict[str, str]:
    ops = {factor: "keep" for factor in ALL_FACTORS}
    ops[edited_factor] = "edit"
    return ops


def touched_modalities(modality_scope: str) -> list[str]:
    if modality_scope == "audio_only":
        return ["audio"]
    if modality_scope == "video_only":
        return ["video"]
    return ["audio", "video"]


def build_seeds() -> list[SeedFamily]:
    # 12 hand-authored families aligned with ROUTE35_PILOT_MATRIX_2026-03-19.md
    return [
        SeedFamily("F01", "performance", "the same female singer", "the same dim jazz bar", "performing at the microphone", "in a medium shot", "warm cinematic lighting", "light applause", "singing normally", "soft room reverb", "speech_voice_emotion", "audio_only", "whispering softly", "singing brightly"),
        SeedFamily("F02", "host_or_narration", "the same male host", "the same quiet bookstore", "speaking to camera", "in a static medium shot", "soft documentary lighting", "faint page-turn sounds", "calm narration", "low room tone", "speech_voice_emotion", "audio_only", "gentle whisper narration", "energetic narration"),
        SeedFamily("F03", "indoor_work_or_craft", "the same chef", "the same busy kitchen", "chopping vegetables", "in a medium shot", "crisp natural lighting", "normal kitchen sounds", "speaking calmly", "steady room tone", "audio_event", "audio_only", "a sharp knife tapping rhythm", "a loud pan sizzling burst"),
        SeedFamily("F04", "street_or_public_space", "the same street vendor", "the same night market stall", "packing snacks", "in a medium shot", "neon evening lighting", "crowd chatter", "calling out calmly", "city ambience", "audio_event", "audio_only", "paper bag rustling close-up", "metal tray clatter"),
        SeedFamily("F05", "travel_or_walkthrough", "the same travel host", "the same indoor market", "walking and presenting", "in a medium tracking shot", "vivid natural lighting", "distant chatter", "cheerful narration", "soft room tone", "ambiance_music", "audio_only", "quiet room tone only", "light upbeat background music"),
        SeedFamily("F06", "indoor_work_or_craft", "the same painter", "the same art studio", "painting on a canvas", "in a medium shot", "warm tungsten lighting", "brush strokes", "describing the process calmly", "gentle studio hum", "ambiance_music", "audio_only", "bare studio ambience", "soft lo-fi background music"),
        SeedFamily("F07", "host_or_narration", "the same narrator", "the same library aisle", "speaking to camera", "in a static medium shot", "soft balanced lighting", "quiet page turns", "calm narration", "low room tone", "camera", "video_only", "in a handheld close-up", "in a wide locked-off shot"),
        SeedFamily("F08", "performance", "the same dancer", "the same rehearsal room", "demonstrating a short routine", "in a medium shot", "neutral studio lighting", "shoe steps", "counting softly", "light hall reverb", "camera", "video_only", "in a low-angle close shot", "in a high-angle wide shot"),
        SeedFamily("F09", "performance", "the same singer", "the same small stage", "performing while standing", "in a medium shot", "warm stage lighting", "light applause", "singing steadily", "soft hall reverb", "style_look", "video_only", "high-contrast noir lighting", "bright pastel stage lighting"),
        SeedFamily("F10", "indoor_work_or_craft", "the same craftsperson", "the same workshop", "assembling a wooden frame", "in a medium shot", "natural daylight look", "tool clicks", "explaining each step", "low workshop hum", "style_look", "video_only", "cool desaturated industrial look", "golden cinematic look"),
        SeedFamily("F11", "street_or_public_space", "the same street drummer", "the same city corner", "playing a steady rhythm", "in a wide shot", "neon city lighting", "nearby footsteps", "focused shouting", "light traffic hum", "motion_action", "both", "now dancing between drum hits", "now marching while drumming"),
        SeedFamily("F12", "travel_or_walkthrough", "the same traveler", "the same rainy train platform", "walking toward the train", "in a medium tracking shot", "moody rainy lighting", "rolling suitcase sounds", "speaking briefly", "rain ambience", "motion_action", "both", "now breaking into a light run", "now slowing to a cautious stop"),
    ]


def build_v2_group(seed: SeedFamily, group_index: int) -> dict[str, Any]:
    if seed.modality_scope not in VALID_MODALITY_SCOPE:
        raise ValueError(f"Invalid modality_scope: {seed.modality_scope}")

    base_values = {
        "identity": seed.identity,
        "scene": seed.scene,
        "motion_action": seed.motion_action,
        "camera": seed.camera,
        "style_look": seed.style_look,
        "audio_event": seed.audio_event,
        "speech_voice_emotion": seed.speech_voice_emotion,
        "ambiance_music": seed.ambiance_music,
    }

    a_values = dict(base_values)
    b_values = dict(base_values)
    a_values[seed.edited_factor] = seed.branch_a_value
    b_values[seed.edited_factor] = seed.branch_b_value

    keep_factors = [f for f in ALL_FACTORS if f != seed.edited_factor]
    edit_factors = [seed.edited_factor]
    ops = build_factor_ops(seed.edited_factor)

    return {
        "group_id": f"r35v2_smoke_{group_index:05d}",
        "sample_id": f"{group_index:05d}",
        "episode_duration_seconds": 15.0,
        "switch_time_seconds": 5.0,
        "category": seed.category,
        "family_id": seed.family_id,
        "modality_scope": seed.modality_scope,
        "shared_prefix_prompt": render_prompt(base_values),
        "shared_keep_factors": keep_factors,
        "shared_factor_values": dict(base_values, modality_scope=seed.modality_scope),
        "branches": [
            {
                "branch_id": "A",
                "branch_prompt": render_prompt(a_values),
                "edit_factors": edit_factors,
                "edited_factor_values": {seed.edited_factor: seed.branch_a_value},
                "touched_modalities": touched_modalities(seed.modality_scope),
                "factor_ops": ops,
            },
            {
                "branch_id": "B",
                "branch_prompt": render_prompt(b_values),
                "edit_factors": edit_factors,
                "edited_factor_values": {seed.edited_factor: seed.branch_b_value},
                "touched_modalities": touched_modalities(seed.modality_scope),
                "factor_ops": ops,
            },
        ],
    }


def build_v1_rows_from_v2_group(group: dict[str, Any], episode_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, branch in enumerate(group["branches"]):
        episode_id = f"r3v1_smoke_{episode_offset + idx:05d}"
        prefix_prompt = group["shared_prefix_prompt"]
        switch_prompt = branch["branch_prompt"]
        rows.append(
            {
                "episode_id": episode_id,
                "sample_id": group["sample_id"],
                "group_id": group["group_id"],
                "parent_branch_id": branch["branch_id"],
                "category": group["category"],
                "family_id": group["family_id"],
                "episode_duration_seconds": group["episode_duration_seconds"],
                "switch_time_seconds": group["switch_time_seconds"],
                "prefix_prompt": prefix_prompt,
                "switch_prompt": switch_prompt,
                "segments": [
                    {
                        "start_time_seconds": 0.0,
                        "end_time_seconds": group["switch_time_seconds"],
                        "prompt": prefix_prompt,
                    },
                    {
                        "start_time_seconds": group["switch_time_seconds"],
                        "end_time_seconds": group["episode_duration_seconds"],
                        "prompt": switch_prompt,
                    },
                ],
                "keep_factors": group["shared_keep_factors"],
                "edit_factors": branch["edit_factors"],
                "modality_scope": group["modality_scope"],
                "factor_values_old": group["shared_factor_values"],
                "factor_values_new": {
                    **group["shared_factor_values"],
                    **branch["edited_factor_values"],
                },
                "clean_edit": True,
            }
        )
    return rows


def validate_v2_groups(groups: list[dict[str, Any]]) -> None:
    required_group_fields = {
        "group_id",
        "sample_id",
        "episode_duration_seconds",
        "switch_time_seconds",
        "shared_prefix_prompt",
        "shared_keep_factors",
        "shared_factor_values",
        "branches",
        "modality_scope",
    }
    required_branch_fields = {
        "branch_id",
        "branch_prompt",
        "edit_factors",
        "edited_factor_values",
        "touched_modalities",
        "factor_ops",
    }

    seen_ids: set[str] = set()
    for group in groups:
        missing = required_group_fields - set(group.keys())
        if missing:
            raise ValueError(f"group {group.get('group_id')} missing fields: {sorted(missing)}")
        group_id = str(group["group_id"])
        if group_id in seen_ids:
            raise ValueError(f"duplicate group_id: {group_id}")
        seen_ids.add(group_id)

        if group["modality_scope"] not in VALID_MODALITY_SCOPE:
            raise ValueError(f"invalid modality_scope for {group_id}: {group['modality_scope']}")
        if len(group["branches"]) < 2:
            raise ValueError(f"{group_id} must have at least two branches")

        for branch in group["branches"]:
            branch_missing = required_branch_fields - set(branch.keys())
            if branch_missing:
                raise ValueError(f"group {group_id} branch missing fields: {sorted(branch_missing)}")
            edits = list(branch["edit_factors"])
            if len(edits) != 1:
                raise ValueError(f"group {group_id} branch {branch['branch_id']} must have exactly one edited factor")
            if set(edits) & set(group["shared_keep_factors"]):
                raise ValueError(f"group {group_id} has overlapping keep/edit factors")
            expected_touch = touched_modalities(group["modality_scope"])
            if list(branch["touched_modalities"]) != expected_touch:
                raise ValueError(
                    f"group {group_id} branch {branch['branch_id']} touched_modalities mismatch: "
                    f"expected {expected_touch}, got {branch['touched_modalities']}"
                )


def validate_v1_rows(rows: list[dict[str, Any]]) -> None:
    required_fields = {
        "episode_id",
        "sample_id",
        "episode_duration_seconds",
        "switch_time_seconds",
        "segments",
        "prefix_prompt",
        "switch_prompt",
        "keep_factors",
        "edit_factors",
        "modality_scope",
    }
    seen_ids: set[str] = set()
    for row in rows:
        missing = required_fields - set(row.keys())
        if missing:
            raise ValueError(f"episode {row.get('episode_id')} missing fields: {sorted(missing)}")
        episode_id = str(row["episode_id"])
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        seen_ids.add(episode_id)
        if len(row["segments"]) != 2:
            raise ValueError(f"episode {episode_id} must contain exactly two segments")
        if set(row["keep_factors"]) & set(row["edit_factors"]):
            raise ValueError(f"episode {episode_id} has overlapping keep/edit factors")
        if row["modality_scope"] not in VALID_MODALITY_SCOPE:
            raise ValueError(f"episode {episode_id} has invalid modality_scope: {row['modality_scope']}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_dataset(output_dir: Path, max_groups: int) -> tuple[Path, Path, dict[str, int]]:
    seeds = build_seeds()
    if max_groups <= 0:
        raise ValueError("max_groups must be > 0")

    selected = seeds[:max_groups]
    groups = [build_v2_group(seed, i) for i, seed in enumerate(selected)]
    rows: list[dict[str, Any]] = []
    for i, group in enumerate(groups):
        rows.extend(build_v1_rows_from_v2_group(group, i * 2))

    validate_v2_groups(groups)
    validate_v1_rows(rows)

    v2_path = output_dir / "route35_v2_groups_smoke_v1.jsonl"
    v1_path = output_dir / "route3_v1_switch_from_v2_smoke_v1.jsonl"
    write_jsonl(v2_path, groups)
    write_jsonl(v1_path, rows)

    modality_counts = {"audio_only": 0, "video_only": 0, "both": 0}
    for group in groups:
        modality_counts[group["modality_scope"]] += 1

    summary = {
        "v2_groups": len(groups),
        "v1_rows": len(rows),
        "audio_only_groups": modality_counts["audio_only"],
        "video_only_groups": modality_counts["video_only"],
        "both_groups": modality_counts["both"],
    }
    return v2_path, v1_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Route35 smoke prompt datasets without API calls")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/route35_prompt_smoke_v1"),
        help="Output directory under qb-ilm",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=12,
        help="Number of V2 groups to emit (recommended 8-12 for smoke)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v2_path, v1_path, summary = build_dataset(output_dir=args.output_dir, max_groups=args.max_groups)
    print(f"[route35-smoke] wrote V2 groups: {v2_path}")
    print(f"[route35-smoke] wrote V1 switch rows: {v1_path}")
    print("[route35-smoke] summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
