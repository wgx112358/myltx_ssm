from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_streaming_backbone_smoke_routes_persistent_ssm_to_official_preview() -> None:
    script = (REPO_ROOT / "qz" / "run_streaming_backbone_smoke.sh").read_text(encoding="utf-8")

    assert "official_2stage_backbone_orchestrator.py" in script
    assert "official_2stage_persistent_ssm_backbone_smoke_preview" in script
    assert "--distilled-checkpoint-path" in script
    assert "--gemma-root" in script
    assert "--spatial-upsampler-path" in script
    assert "--ssm-checkpoint" in script


def test_streaming_backbone_smoke_defaults_to_official_frame_rate() -> None:
    script = (REPO_ROOT / "qz" / "run_streaming_backbone_smoke.sh").read_text(encoding="utf-8")

    assert 'FRAME_RATE="${FRAME_RATE:-24}"' in script


def test_streaming_backbone_smoke_defaults_to_official_num_frames() -> None:
    script = (REPO_ROOT / "qz" / "run_streaming_backbone_smoke.sh").read_text(encoding="utf-8")

    assert 'CHUNK_NUM_FRAMES="${CHUNK_NUM_FRAMES:-121}"' in script


def test_streaming_backbone_smoke_enables_expandable_segments_allocator() -> None:
    script = (REPO_ROOT / "qz" / "run_streaming_backbone_smoke.sh").read_text(encoding="utf-8")

    assert 'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in script


def test_step300_demo_wrapper_defaults_to_official_frame_rate() -> None:
    script = (REPO_ROOT / "qz" / "run_official_2stage_backbone_demo_step300.sh").read_text(encoding="utf-8")

    assert 'FRAME_RATE="${FRAME_RATE:-24}"' in script


def test_step300_demo_wrapper_defaults_to_official_num_frames() -> None:
    script = (REPO_ROOT / "qz" / "run_official_2stage_backbone_demo_step300.sh").read_text(encoding="utf-8")

    assert 'CHUNK_NUM_FRAMES="${CHUNK_NUM_FRAMES:-121}"' in script


def test_streaming_backbone_smoke_keeps_legacy_branches_on_old_runtime() -> None:
    script = (REPO_ROOT / "qz" / "run_streaming_backbone_smoke.sh").read_text(encoding="utf-8")

    assert 'elif [[ "$REFERENCE_WINDOW_CHUNKS" == "0" ]]' in script
    assert "python scripts/streaming_inference.py" in script
    assert "streaming_backbone_no_memory_smoke_v2" in script
    assert "streaming_backbone_short_context_smoke_v2" in script


def test_streaming_backbone_audit_defaults_to_preview_output_dir() -> None:
    script = (REPO_ROOT / "qz" / "run_streaming_backbone_audit_smoke.sh").read_text(encoding="utf-8")

    assert "official_2stage_persistent_ssm_backbone_smoke_preview" in script
    assert "streaming_backbone_persistent_ssm_smoke_v2" not in script


def test_submit_wrapper_uses_preview_experiment_naming() -> None:
    script = (REPO_ROOT / "qz" / "submit_streaming_backbone_smoke_p3.sh").read_text(encoding="utf-8")

    assert '--version "preview-v1"' in script
    assert '--experiment "official-2stage-persistent-ssm-backbone-preview"' in script
