#!/usr/bin/env python3
"""Generate structured Route3 prompt pilots (V1 switch + V2 counterfactual branches).

This script provides an executable S11-S12 path without depending on API availability.
It follows the frozen Route 3.5 schema and writes artifacts under qb-ilm by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FACTOR_ORDER = [
    "identity",
    "scene",
    "motion_action",
    "camera",
    "style_look",
    "audio_event",
    "speech_voice_emotion",
    "ambiance_music",
]

VALID_MODALITY_SCOPE = {"audio_only", "video_only", "both"}

BASE_AXIS_TARGETS = {
    "speech_voice_emotion": 18,
    "audio_event": 10,
    "ambiance_music": 8,
    "camera": 10,
    "style_look": 8,
    "motion_action": 12,
    "style_look_paired_audio": 6,
}

AXIS_TO_MODALITY = {
    "speech_voice_emotion": ("audio_only", ["audio"]),
    "audio_event": ("audio_only", ["audio"]),
    "ambiance_music": ("audio_only", ["audio"]),
    "camera": ("video_only", ["video"]),
    "style_look": ("video_only", ["video"]),
    "motion_action": ("both", ["video", "audio"]),
    "style_look_paired_audio": ("both", ["video", "audio"]),
}

AXIS_VARIANTS = {
    "speech_voice_emotion": [
        ("whispering softly", "speaking with bright excitement"),
        ("calm narration", "intense emphatic narration"),
        ("gentle singing", "powerful energetic singing"),
    ],
    "audio_event": [
        ("light keyboard taps", "sharp metal clanks"),
        ("faint crowd chatter", "loud cheering"),
        ("soft footsteps", "rapid running footsteps"),
    ],
    "ambiance_music": [
        ("low room tone", "upbeat electronic music bed"),
        ("soft room reverb", "tense cinematic drone"),
        ("quiet ambient hum", "warm jazz backing track"),
    ],
    "camera": [
        ("static medium shot", "handheld close-up shot"),
        ("wide fixed shot", "slow push-in close shot"),
        ("front-facing medium shot", "side-angle tracking shot"),
    ],
    "style_look": [
        ("soft documentary lighting", "high-contrast noir lighting"),
        ("warm cinematic lighting", "cool desaturated grading"),
        ("natural daylight look", "neon night look"),
    ],
    "motion_action": [
        ("moving steadily while presenting", "stopping to gesture sharply"),
        ("walking calmly", "running quickly"),
        ("playing a steady rhythm", "switching to energetic movement"),
    ],
    "style_look_paired_audio": [
        (("warm cinematic lighting", "low-key neon contrast"), ("soft room reverb", "tense pulsing synth bed")),
        (("natural daylight look", "stormy dark grading"), ("quiet ambient hum", "dramatic bass rumble")),
        (("clean studio lighting", "strobe club lighting"), ("low room tone", "heavy electronic beat")),
    ],
}


@dataclass(frozen=True)
class FamilySeed:
    family_id: str
    family_name: str
    category: str
    identities: tuple[str, ...]
    scenes: tuple[str, ...]
    motions: tuple[str, ...]
    cameras: tuple[str, ...]
    styles: tuple[str, ...]
    audio_events: tuple[str, ...]
    speeches: tuple[str, ...]
    ambiances: tuple[str, ...]


FAMILY_SEEDS: dict[str, FamilySeed] = {
    "F01": FamilySeed(
        "F01", "same singer in same jazz bar", "Human",
        ("the same young female singer", "the same male vocalist"),
        ("a dim jazz bar", "a small live-music lounge"),
        ("performing at the microphone", "performing on a small stage"),
        ("medium shot", "waist-up shot"),
        ("warm cinematic lighting", "soft tungsten lighting"),
        ("light applause", "gentle clapping"),
        ("singing normally", "steady vocal delivery"),
        ("soft room reverb", "quiet hall ambience"),
    ),
    "F02": FamilySeed(
        "F02", "same host in same bookstore", "Built_World",
        ("the same male host", "the same female presenter"),
        ("a quiet bookstore", "a cozy reading room"),
        ("speaking to camera", "introducing books to camera"),
        ("static medium shot", "front-facing medium shot"),
        ("soft documentary lighting", "clean warm lighting"),
        ("faint page-turn sounds", "subtle shelf creaks"),
        ("calm narration", "steady speech"),
        ("low room tone", "quiet indoor ambience"),
    ),
    "F03": FamilySeed(
        "F03", "same chef in same kitchen", "Object",
        ("the same chef",),
        ("a busy kitchen", "an open kitchen"),
        ("chopping vegetables", "preparing ingredients"),
        ("medium shot",),
        ("crisp documentary lighting",),
        ("normal kitchen sounds",),
        ("speaking calmly",),
        ("steady room tone",),
    ),
    "F04": FamilySeed(
        "F04", "same street vendor in same night market", "Built_World",
        ("the same street vendor",),
        ("a crowded night market",),
        ("serving customers", "arranging food items"),
        ("wide shot", "medium-wide shot"),
        ("neon city lighting",),
        ("distant crowd chatter",),
        ("short spoken calls",),
        ("light traffic hum",),
    ),
    "F05": FamilySeed(
        "F05", "same travel host in same indoor market", "Natural_World",
        ("the same travel host",),
        ("an indoor market",),
        ("walking through stalls", "talking while walking"),
        ("medium tracking shot",),
        ("vivid natural lighting",),
        ("distant chatter",),
        ("cheerful narration",),
        ("soft room tone",),
    ),
    "F06": FamilySeed(
        "F06", "same painter in same studio", "Object",
        ("the same painter",),
        ("an art studio",),
        ("painting on a canvas",),
        ("medium shot",),
        ("clean studio lighting",),
        ("brush strokes",),
        ("quiet self-talk",),
        ("quiet ambient hum",),
    ),
    "F07": FamilySeed(
        "F07", "same narrator in same library", "Built_World",
        ("the same narrator",),
        ("a quiet library aisle",),
        ("speaking toward camera",),
        ("static medium shot", "wide fixed shot"),
        ("soft documentary lighting",),
        ("faint page-turn sounds",),
        ("calm narration",),
        ("low room tone",),
    ),
    "F08": FamilySeed(
        "F08", "same dancer in same rehearsal room", "Human",
        ("the same dancer",),
        ("a rehearsal room",),
        ("demonstrating movement",),
        ("wide shot", "front-facing medium shot"),
        ("clean studio lighting",),
        ("light shoe squeaks",),
        ("brief spoken cues",),
        ("quiet room ambience",),
    ),
    "F09": FamilySeed(
        "F09", "same singer on same small stage", "Human",
        ("the same stage singer",),
        ("a small indoor stage",),
        ("performing live",),
        ("medium shot",),
        ("warm cinematic lighting", "natural daylight look"),
        ("light applause",),
        ("steady vocal delivery",),
        ("soft hall ambience",),
    ),
    "F10": FamilySeed(
        "F10", "same craftsperson in same workshop", "Object",
        ("the same craftsperson",),
        ("a wood workshop",),
        ("assembling a small object",),
        ("medium shot",),
        ("natural daylight look", "clean studio lighting"),
        ("tool clicks",),
        ("short spoken notes",),
        ("low workshop ambience",),
    ),
    "F11": FamilySeed(
        "F11", "same street drummer on same corner", "Human",
        ("the same street drummer",),
        ("a busy night corner",),
        ("playing a steady rhythm",),
        ("wide shot",),
        ("neon city lighting",),
        ("nearby footsteps",),
        ("focused shouting",),
        ("light traffic hum",),
    ),
    "F12": FamilySeed(
        "F12", "same traveler on same rainy platform", "Natural_World",
        ("the same traveler",),
        ("a rainy train platform",),
        ("waiting and gesturing",),
        ("medium-wide shot",),
        ("cool desaturated grading",),
        ("distant train sounds",),
        ("short spoken comments",),
        ("windy station ambience",),
    ),
}

AXIS_FAMILY_ORDER = {
    "speech_voice_emotion": ["F01", "F02"],
    "audio_event": ["F03", "F04"],
    "ambiance_music": ["F05", "F06"],
    "camera": ["F07", "F08"],
    "style_look": ["F09", "F10"],
    "motion_action": ["F11", "F12"],
    "style_look_paired_audio": ["F11", "F12"],
}


def render_prompt(values: dict[str, str]) -> str:
    return (
        f"{values['identity']} in {values['scene']}, {values['motion_action']}, "
        f"{values['camera']}, {values['style_look']}, with {values['audio_event']}, "
        f"{values['speech_voice_emotion']}, over {values['ambiance_music']}"
    )


def factor_ops(edit_factors: list[str]) -> dict[str, str]:
    return {name: ("edit" if name in edit_factors else "keep") for name in FACTOR_ORDER}


def scale_axis_targets(total_groups: int) -> dict[str, int]:
    if total_groups <= 0:
        raise ValueError("target_v2_groups must be > 0")
    base_total = sum(BASE_AXIS_TARGETS.values())
    if total_groups == base_total:
        return dict(BASE_AXIS_TARGETS)

    scaled: dict[str, int] = {}
    residuals: list[tuple[float, str]] = []
    running = 0
    for axis, base in BASE_AXIS_TARGETS.items():
        exact = total_groups * base / base_total
        count = int(math.floor(exact))
        scaled[axis] = count
        running += count
        residuals.append((exact - count, axis))

    remain = total_groups - running
    for _, axis in sorted(residuals, reverse=True):
        if remain <= 0:
            break
        scaled[axis] += 1
        remain -= 1

    if sum(scaled.values()) != total_groups:
        raise RuntimeError("failed to scale axis targets")
    return scaled


def choose(seq: tuple[str, ...], idx: int) -> str:
    return seq[idx % len(seq)]


def base_values_for_family(seed: FamilySeed, idx: int) -> dict[str, str]:
    return {
        "identity": choose(seed.identities, idx),
        "scene": choose(seed.scenes, idx),
        "motion_action": choose(seed.motions, idx),
        "camera": choose(seed.cameras, idx),
        "style_look": choose(seed.styles, idx),
        "audio_event": choose(seed.audio_events, idx),
        "speech_voice_emotion": choose(seed.speeches, idx),
        "ambiance_music": choose(seed.ambiances, idx),
    }


def branch_values(axis: str, base: dict[str, str], idx: int) -> tuple[dict[str, str], dict[str, str], list[str]]:
    a = dict(base)
    b = dict(base)
    if axis == "style_look_paired_audio":
        style_pair, audio_pair = AXIS_VARIANTS[axis][idx % len(AXIS_VARIANTS[axis])]
        a["style_look"], b["style_look"] = style_pair
        a["ambiance_music"], b["ambiance_music"] = audio_pair
        return a, b, ["style_look", "ambiance_music"]

    values = AXIS_VARIANTS[axis][idx % len(AXIS_VARIANTS[axis])]
    a[axis], b[axis] = values
    return a, b, [axis]


def parse_split_counts(raw: str | None, total: int) -> tuple[int, int, int]:
    if not raw:
        train = int(total * 2 / 3)
        val = int((total - train) / 2)
        test = total - train - val
        return train, val, test

    items = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(items) != 3:
        raise ValueError("split_counts must be 'train,val,test'")
    if sum(items) != total:
        raise ValueError(f"split_counts sum {sum(items)} != total_groups {total}")
    return items[0], items[1], items[2]


def build_groups(target_v2_groups: int, seed: int, split_counts: tuple[int, int, int]) -> list[dict[str, Any]]:
    axis_targets = scale_axis_targets(target_v2_groups)
    rng = random.Random(seed)

    groups: list[dict[str, Any]] = []
    family_hit_count: dict[str, int] = defaultdict(int)

    group_index = 0
    for axis, axis_count in axis_targets.items():
        family_ids = AXIS_FAMILY_ORDER[axis]
        for i in range(axis_count):
            family_id = family_ids[i % len(family_ids)]
            family_seed = FAMILY_SEEDS[family_id]
            family_hit_count[family_id] += 1
            base = base_values_for_family(family_seed, family_hit_count[family_id])
            branch_a_vals, branch_b_vals, edit_factors = branch_values(axis, base, i)
            modality_scope, touched_modalities = AXIS_TO_MODALITY[axis]

            keep_factors = [f for f in FACTOR_ORDER if f not in edit_factors]
            group_id = f"r35v2_{group_index:06d}"
            sample_id = f"{group_index:05d}"

            group = {
                "group_id": group_id,
                "sample_id": sample_id,
                "episode_duration_seconds": 15.0,
                "switch_time_seconds": 5.0,
                "shared_prefix_prompt": render_prompt(base),
                "shared_keep_factors": keep_factors,
                "shared_factor_values": {**base, "modality_scope": modality_scope},
                "modality_scope": modality_scope,
                "category": family_seed.category,
                "family_id": family_seed.family_id,
                "family_name": family_seed.family_name,
                "counterfactual_axis": axis,
                "branches": [
                    {
                        "branch_id": "A",
                        "branch_prompt": render_prompt(branch_a_vals),
                        "edit_factors": edit_factors,
                        "edited_factor_values": {k: branch_a_vals[k] for k in edit_factors},
                        "touched_modalities": touched_modalities,
                        "factor_ops": factor_ops(edit_factors),
                    },
                    {
                        "branch_id": "B",
                        "branch_prompt": render_prompt(branch_b_vals),
                        "edit_factors": edit_factors,
                        "edited_factor_values": {k: branch_b_vals[k] for k in edit_factors},
                        "touched_modalities": touched_modalities,
                        "factor_ops": factor_ops(edit_factors),
                    },
                ],
            }
            groups.append(group)
            group_index += 1

    if len(groups) != target_v2_groups:
        raise RuntimeError(f"generated {len(groups)} groups, expected {target_v2_groups}")

    rng.shuffle(groups)

    train_n, val_n, test_n = split_counts
    split_labels = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
    if len(split_labels) != len(groups):
        raise RuntimeError("split label length mismatch")

    for group, split in zip(groups, split_labels):
        group["split"] = split

    return groups


def derive_v1_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_idx = 0
    for group in groups:
        for branch in group["branches"]:
            row = {
                "episode_id": f"r3v1_{episode_idx:06d}",
                "sample_id": group["sample_id"],
                "episode_duration_seconds": group["episode_duration_seconds"],
                "switch_time_seconds": group["switch_time_seconds"],
                "prefix_prompt": group["shared_prefix_prompt"],
                "switch_prompt": branch["branch_prompt"],
                "segments": [
                    {
                        "start_seconds": 0.0,
                        "duration_seconds": group["switch_time_seconds"],
                        "prompt": group["shared_prefix_prompt"],
                        "category": group["category"],
                    },
                    {
                        "start_seconds": group["switch_time_seconds"],
                        "duration_seconds": group["episode_duration_seconds"] - group["switch_time_seconds"],
                        "prompt": branch["branch_prompt"],
                        "category": group["category"],
                    },
                ],
                "keep_factors": group["shared_keep_factors"],
                "edit_factors": branch["edit_factors"],
                "modality_scope": group["modality_scope"],
                "category": group["category"],
                "family_id": group["family_id"],
                "family_name": group["family_name"],
                "counterfactual_axis": group["counterfactual_axis"],
                "parent_group_id": group["group_id"],
                "parent_branch_id": branch["branch_id"],
                "factor_values_old": group["shared_factor_values"],
                "factor_values_new": {**group["shared_factor_values"], **branch["edited_factor_values"]},
                "factor_ops": branch["factor_ops"],
                "split": group["split"],
                "clean_edit": True,
            }
            rows.append(row)
            episode_idx += 1

    return rows


def validate_groups(groups: list[dict[str, Any]]) -> None:
    seen_group_ids: set[str] = set()
    for group in groups:
        gid = group["group_id"]
        if gid in seen_group_ids:
            raise ValueError(f"duplicate group_id: {gid}")
        seen_group_ids.add(gid)

        if group["modality_scope"] not in VALID_MODALITY_SCOPE:
            raise ValueError(f"invalid modality_scope in {gid}")

        if len(group["branches"]) < 2:
            raise ValueError(f"group {gid} has <2 branches")

        keep = set(group["shared_keep_factors"])
        if not keep:
            raise ValueError(f"group {gid} has empty shared_keep_factors")

        for branch in group["branches"]:
            edits = set(branch["edit_factors"])
            if not edits:
                raise ValueError(f"group {gid} branch {branch['branch_id']} has empty edit_factors")
            if keep & edits:
                raise ValueError(f"group {gid} branch {branch['branch_id']} has keep/edit overlap")


def validate_v1_rows(rows: list[dict[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"v1 rows {len(rows)} != expected {expected}")

    seen_episode_ids: set[str] = set()
    for row in rows:
        eid = row["episode_id"]
        if eid in seen_episode_ids:
            raise ValueError(f"duplicate episode_id: {eid}")
        seen_episode_ids.add(eid)

        if row["modality_scope"] not in VALID_MODALITY_SCOPE:
            raise ValueError(f"invalid modality_scope in {eid}")

        if len(row["segments"]) != 2:
            raise ValueError(f"episode {eid} segments != 2")

        duration_sum = sum(float(seg["duration_seconds"]) for seg in row["segments"])
        if abs(duration_sum - float(row["episode_duration_seconds"])) > 1e-6:
            raise ValueError(f"episode {eid} duration mismatch")

        if set(row["keep_factors"]) & set(row["edit_factors"]):
            raise ValueError(f"episode {eid} keep/edit overlap")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_v1_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "episode_id", "sample_id", "split", "modality_scope", "category", "family_id", "counterfactual_axis",
        "prefix_prompt", "switch_prompt", "keep_factors", "edit_factors", "parent_group_id", "parent_branch_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            payload = {k: row.get(k) for k in keys}
            payload["keep_factors"] = "|".join(row["keep_factors"])
            payload["edit_factors"] = "|".join(row["edit_factors"])
            writer.writerow(payload)


def write_v2_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    keys = [
        "group_id", "sample_id", "split", "modality_scope", "category", "family_id", "counterfactual_axis",
        "shared_prefix_prompt", "branch_a_prompt", "branch_b_prompt", "shared_keep_factors", "branch_edit_factors",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for g in groups:
            b0, b1 = g["branches"][0], g["branches"][1]
            writer.writerow(
                {
                    "group_id": g["group_id"],
                    "sample_id": g["sample_id"],
                    "split": g["split"],
                    "modality_scope": g["modality_scope"],
                    "category": g["category"],
                    "family_id": g["family_id"],
                    "counterfactual_axis": g["counterfactual_axis"],
                    "shared_prefix_prompt": g["shared_prefix_prompt"],
                    "branch_a_prompt": b0["branch_prompt"],
                    "branch_b_prompt": b1["branch_prompt"],
                    "shared_keep_factors": "|".join(g["shared_keep_factors"]),
                    "branch_edit_factors": "|".join(b0["edit_factors"]),
                }
            )


def inspect_canonical_csv(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"canonical_categorized_csv": None, "rows": 0}
    if not path.exists():
        return {"canonical_categorized_csv": str(path), "rows": 0, "exists": False}

    rows = 0
    categories: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            c = (row.get("category") or "").strip()
            if c:
                categories[c] += 1
    return {
        "canonical_categorized_csv": str(path),
        "rows": rows,
        "exists": True,
        "category_counts": dict(categories),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Route3 prompt pilot artifacts")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/"),
    )
    parser.add_argument("--run-name", type=str, default="route35_prompt_pilot_smoke_v1")
    parser.add_argument("--target-v2-groups", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--split-counts", type=str, default=None, help="train,val,test counts")
    parser.add_argument(
        "--canonical-script",
        type=Path,
        default=Path("/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/datagen/gen_prompts_v1.py"),
    )
    parser.add_argument("--canonical-categorized-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = (args.output_root / args.run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_counts = parse_split_counts(args.split_counts, args.target_v2_groups)
    groups = build_groups(args.target_v2_groups, args.seed, split_counts)
    validate_groups(groups)

    v1_rows = derive_v1_rows(groups)
    validate_v1_rows(v1_rows, expected=2 * len(groups))

    v2_jsonl = out_dir / "route35_v2_counterfactual_groups.jsonl"
    v1_jsonl = out_dir / "route3_v1_switch_episodes.jsonl"
    v2_csv = out_dir / "route35_v2_counterfactual_groups.csv"
    v1_csv = out_dir / "route3_v1_switch_episodes.csv"

    write_jsonl(v2_jsonl, groups)
    write_jsonl(v1_jsonl, v1_rows)
    write_v2_csv(v2_csv, groups)
    write_v1_csv(v1_csv, v1_rows)

    modality_counts = Counter(g["modality_scope"] for g in groups)
    axis_counts = Counter(g["counterfactual_axis"] for g in groups)
    split_group_counts = Counter(g["split"] for g in groups)
    split_v1_counts = Counter(r["split"] for r in v1_rows)
    family_counts = Counter(g["family_id"] for g in groups)

    summary = {
        "run_name": args.run_name,
        "target_v2_groups": args.target_v2_groups,
        "actual_v2_groups": len(groups),
        "actual_v1_rows": len(v1_rows),
        "expected_v1_rows": 2 * len(groups),
        "modality_counts_v2": dict(modality_counts),
        "axis_counts_v2": dict(axis_counts),
        "split_counts_v2": dict(split_group_counts),
        "split_counts_v1": dict(split_v1_counts),
        "family_counts_v2": dict(family_counts),
        "canonical_reference": {
            "canonical_script": str(args.canonical_script),
            **inspect_canonical_csv(args.canonical_categorized_csv),
        },
        "files": {
            "v2_jsonl": str(v2_jsonl),
            "v1_jsonl": str(v1_jsonl),
            "v2_csv": str(v2_csv),
            "v1_csv": str(v1_csv),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
