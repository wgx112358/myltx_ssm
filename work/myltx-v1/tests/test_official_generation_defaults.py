from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from official_generation_defaults import (  # noqa: E402
    OFFICIAL_2_STAGE_RESOLUTION_PRESETS,
    get_official_2_stage_resolution,
)
from streaming_inference import StreamingConfig  # noqa: E402


def test_official_supported_resolutions_include_small_and_hq() -> None:
    assert OFFICIAL_2_STAGE_RESOLUTION_PRESETS["small"].height == 1024
    assert OFFICIAL_2_STAGE_RESOLUTION_PRESETS["small"].width == 1536
    assert OFFICIAL_2_STAGE_RESOLUTION_PRESETS["hq"].height == 1088
    assert OFFICIAL_2_STAGE_RESOLUTION_PRESETS["hq"].width == 1920


def test_streaming_config_defaults_to_smaller_official_resolution() -> None:
    config = StreamingConfig()

    official_small = get_official_2_stage_resolution("small")
    assert config.height == official_small.height
    assert config.width == official_small.width
