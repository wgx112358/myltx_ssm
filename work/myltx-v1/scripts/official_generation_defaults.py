#!/usr/bin/env python3
"""Shared official LTX generation presets used by local streaming wrappers.

The values here intentionally mirror the two 2-stage output resolutions exposed by
the official LTX-2.3 pipelines in `ltx_pipelines.utils.constants` /
`ltx_pipelines.utils.args`:

- standard: 1024x1536
- hq: 1088x1920
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialResolutionPreset:
    name: str
    height: int
    width: int


OFFICIAL_2_STAGE_RESOLUTION_PRESETS: dict[str, OfficialResolutionPreset] = {
    "small": OfficialResolutionPreset(name="small", height=1024, width=1536),
    "hq": OfficialResolutionPreset(name="hq", height=1088, width=1920),
}

DEFAULT_OFFICIAL_2_STAGE_PRESET = "small"


def get_official_2_stage_resolution(preset: str = DEFAULT_OFFICIAL_2_STAGE_PRESET) -> OfficialResolutionPreset:
    try:
        return OFFICIAL_2_STAGE_RESOLUTION_PRESETS[preset]
    except KeyError as exc:
        supported = ", ".join(sorted(OFFICIAL_2_STAGE_RESOLUTION_PRESETS))
        raise ValueError(f"Unsupported official resolution preset: {preset}. Expected one of: {supported}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print shared official LTX generation resolution presets")
    parser.add_argument("--preset", choices=sorted(OFFICIAL_2_STAGE_RESOLUTION_PRESETS), default=DEFAULT_OFFICIAL_2_STAGE_PRESET)
    parser.add_argument("--format", choices=["values", "json"], default="values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_official_2_stage_resolution(args.preset)
    if args.format == "json":
        print(json.dumps({"name": preset.name, "height": preset.height, "width": preset.width}))
        return
    print(f"{preset.height} {preset.width}")


if __name__ == "__main__":
    main()
