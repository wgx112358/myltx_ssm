from __future__ import annotations

import json
import sys
import types
import weakref
import gc
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-pipelines/src"))

import official_2stage_backbone_orchestrator as orchestrator  # noqa: E402
from official_2stage_backbone_orchestrator import BackbonePreviewConfig, run_backbone_preview  # noqa: E402


def test_parse_args_defaults_to_official_frame_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "official_2stage_backbone_orchestrator.py",
            "--manifest",
            "/tmp/manifest.jsonl",
            "--output",
            "/tmp/out.mp4",
            "--distilled-checkpoint-path",
            "/tmp/model.safetensors",
            "--gemma-root",
            "/tmp/gemma",
            "--spatial-upsampler-path",
            "/tmp/upscaler.safetensors",
        ],
    )

    args = orchestrator.parse_args()

    assert args.frame_rate == 24.0


def test_parse_args_defaults_to_official_num_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "official_2stage_backbone_orchestrator.py",
            "--manifest",
            "/tmp/manifest.jsonl",
            "--output",
            "/tmp/out.mp4",
            "--distilled-checkpoint-path",
            "/tmp/model.safetensors",
            "--gemma-root",
            "/tmp/gemma",
            "--spatial-upsampler-path",
            "/tmp/upscaler.safetensors",
        ],
    )

    args = orchestrator.parse_args()

    assert args.chunk_num_frames == 121


def _write_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "streaming_backbone_smoke.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0000",
                "sample_id": "00000",
                "segments": [
                    {
                        "category": "Human_Dialogue",
                        "prompt": "A steady documentary-style bench conversation with consistent identity and voice.",
                        "start_seconds": 0.0,
                        "duration_seconds": 30.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _base_config(tmp_path: Path, **overrides: object) -> BackbonePreviewConfig:
    output_path = tmp_path / "outputs" / "episode_0000.mp4"
    values: dict[str, object] = {
        "manifest_path": str(_write_manifest(tmp_path)),
        "episode_id": "episode_0000",
        "output_path": str(output_path),
        "distilled_checkpoint_path": "/tmp/ltx-2.3-distilled.safetensors",
        "gemma_root": "/tmp/gemma",
        "spatial_upsampler_path": "/tmp/ltx-2.3-spatial-upscaler.safetensors",
        "ssm_checkpoint_path": "/tmp/ssm.pt",
        "preset": "small",
        "chunk_num_frames": 121,
        "frame_rate": 24.0,
        "chunks_per_segment": 3,
        "window_blocks": 2,
        "seed": 42,
        "overwrite": True,
    }
    values.update(overrides)
    return BackbonePreviewConfig(**values)


def test_orchestrator_updates_single_pending_queue_by_chunk_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    encode_calls: list[dict[str, object]] = []
    stitch_calls: list[tuple[list[Path], Path]] = []

    class FakeRunner:
        def __init__(self) -> None:
            self.run_calls: list[dict[str, object]] = []
            self.compress_calls: list[tuple[object, torch.Tensor, torch.Tensor]] = []
            self.shared_video = torch.zeros((1, 2, 128), dtype=torch.float32)
            self.shared_audio = torch.zeros((1, 2, 128), dtype=torch.float32)

        def run_chunk(self, config, ssm_state=None):
            chunk_idx = len(self.run_calls)
            value = float(chunk_idx + 1)
            self.shared_video.fill_(value)
            self.shared_audio.fill_(value * 10.0)
            self.run_calls.append({"ssm_state": ssm_state, "prompt": config.prompt, "seed": config.seed})
            return types.SimpleNamespace(
                final_chunk_video=f"video-{chunk_idx}",
                final_chunk_audio=f"audio-{chunk_idx}",
                next_ssm_state={"chunk": chunk_idx},
                evictable_video_tokens=self.shared_video,
                evictable_audio_tokens=self.shared_audio,
            )

        def compress_evicted_tokens(self, ssm_state, evicted_video, evicted_audio):
            self.compress_calls.append((ssm_state, evicted_video, evicted_audio))
            return {"compressed": len(self.compress_calls), "from": ssm_state}

    fake_runner = FakeRunner()

    monkeypatch.setattr(orchestrator, "OfficialDistilledChunkRunner", lambda **kwargs: fake_runner)
    monkeypatch.setattr(
        orchestrator,
        "encode_video",
        lambda *, video, fps, audio, output_path, video_chunks_number: (
            Path(output_path).parent.mkdir(parents=True, exist_ok=True),
            Path(output_path).write_bytes(b"mp4"),
            encode_calls.append({"video": video, "audio": audio, "output_path": output_path, "fps": fps}),
        ),
    )

    def fake_stitch(chunk_paths: list[Path], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"stitched")
        stitch_calls.append((chunk_paths, output_path))

    monkeypatch.setattr(orchestrator, "_stitch_chunk_files", fake_stitch, raising=False)

    result = run_backbone_preview(_base_config(tmp_path))

    assert result["window_blocks"] == 2
    assert len(fake_runner.run_calls) == 3
    assert fake_runner.run_calls[0]["ssm_state"] is None
    assert fake_runner.run_calls[1]["ssm_state"] == {"chunk": 0}
    assert fake_runner.run_calls[2]["ssm_state"] == {"chunk": 1}
    assert len(fake_runner.compress_calls) == 1
    assert torch.all(fake_runner.compress_calls[0][1] == 1.0)
    assert torch.all(fake_runner.compress_calls[0][2] == 10.0)
    assert [chunk["pending_queue_length"] for chunk in result["chunk_outputs"]] == [1, 2, 2]
    assert [chunk["compression_applied"] for chunk in result["chunk_outputs"]] == [False, False, True]
    assert len(encode_calls) == 3
    assert len(stitch_calls) == 1


def test_orchestrator_queues_detached_chunk_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.shared_video = torch.zeros((1, 2, 128), dtype=torch.float32)
            self.shared_audio = torch.zeros((1, 2, 128), dtype=torch.float32)

        def run_chunk(self, config, ssm_state=None):
            chunk_idx = int((config.seed - 42))
            value = float(chunk_idx + 1)
            self.shared_video.fill_(value)
            self.shared_audio.fill_(value * 10.0)
            return types.SimpleNamespace(
                final_chunk_video=f"video-{chunk_idx}",
                final_chunk_audio=f"audio-{chunk_idx}",
                next_ssm_state={"chunk": chunk_idx},
                evictable_video_tokens=self.shared_video,
                evictable_audio_tokens=self.shared_audio,
            )

        def compress_evicted_tokens(self, ssm_state, evicted_video, evicted_audio):
            return {"compressed_from": ssm_state}

    fake_runner = FakeRunner()

    monkeypatch.setattr(orchestrator, "OfficialDistilledChunkRunner", lambda **kwargs: fake_runner)
    monkeypatch.setattr(
        orchestrator,
        "encode_video",
        lambda *, video, fps, audio, output_path, video_chunks_number: (
            Path(output_path).parent.mkdir(parents=True, exist_ok=True),
            Path(output_path).write_bytes(b"mp4"),
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_stitch_chunk_files",
        lambda chunk_paths, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True),
            output_path.write_bytes(b"stitched"),
        ),
        raising=False,
    )

    output = run_backbone_preview(_base_config(tmp_path, chunks_per_segment=2))
    output_json = Path(output["output_path"]).with_suffix(".json")
    metadata = json.loads(output_json.read_text(encoding="utf-8"))

    assert metadata["memory_mode"] == "persistent_ssm"
    assert metadata["preset"] == "small"
    assert metadata["height"] == 1024
    assert metadata["width"] == 1536
    assert metadata["chunk_num_frames"] == 121
    assert metadata["frame_rate"] == 24.0
    assert len(metadata["chunk_outputs"]) == 2
    assert Path(metadata["chunk_outputs"][0]["output_path"]).name == "chunk_000.mp4"
    assert Path(metadata["chunk_outputs"][1]["output_path"]).name == "chunk_001.mp4"
    assert Path(metadata["stitched_output_path"]).name == "episode_0000.mp4"
    assert output_json.exists()
    assert (output_json.parent / "episode_0000" / "chunk_000.mp4").exists()
    assert (output_json.parent / "episode_0000" / "chunk_001.mp4").exists()


def test_snapshot_tokens_moves_snapshots_off_gpu_queue() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.detach_called = 0
            self.cpu_called = 0
            self.clone_called = 0

        def detach(self):
            self.detach_called += 1
            return self

        def cpu(self):
            self.cpu_called += 1
            return self

        def clone(self):
            self.clone_called += 1
            return "snapshot-on-cpu"

    fake = FakeTensor()

    snapshot = orchestrator._snapshot_tokens(fake)

    assert snapshot == "snapshot-on-cpu"
    assert fake.detach_called == 1
    assert fake.cpu_called == 1
    assert fake.clone_called == 1


def test_orchestrator_releases_eviction_tensors_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_refs: dict[str, weakref.ReferenceType[object]] = {}

    class FakeTokens:
        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return "snapshot"

    class FakeRunner:
        def run_chunk(self, config, ssm_state=None):
            video_tokens = FakeTokens()
            audio_tokens = FakeTokens()
            token_refs["video"] = weakref.ref(video_tokens)
            token_refs["audio"] = weakref.ref(audio_tokens)
            return types.SimpleNamespace(
                final_chunk_video="video-0",
                final_chunk_audio="audio-0",
                next_ssm_state={"chunk": 0},
                evictable_video_tokens=video_tokens,
                evictable_audio_tokens=audio_tokens,
            )

        def compress_evicted_tokens(self, ssm_state, evicted_video, evicted_audio):
            return {"compressed_from": ssm_state}

    monkeypatch.setattr(orchestrator, "OfficialDistilledChunkRunner", lambda **kwargs: FakeRunner())

    def fake_encode_video(*, video, fps, audio, output_path, video_chunks_number):
        gc.collect()
        assert token_refs["video"]() is None
        assert token_refs["audio"]() is None
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"mp4")

    monkeypatch.setattr(orchestrator, "encode_video", fake_encode_video)
    monkeypatch.setattr(
        orchestrator,
        "_stitch_chunk_files",
        lambda chunk_paths, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True),
            output_path.write_bytes(b"stitched"),
        ),
        raising=False,
    )

    run_backbone_preview(_base_config(tmp_path, chunks_per_segment=1))


def test_orchestrator_creates_chunk_output_directory_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_chunk_dirs: list[Path] = []

    class FakeRunner:
        def run_chunk(self, config, ssm_state=None):
            return types.SimpleNamespace(
                final_chunk_video="video-0",
                final_chunk_audio="audio-0",
                next_ssm_state={"chunk": 0},
                evictable_video_tokens=torch.zeros((1, 2, 128), dtype=torch.float32),
                evictable_audio_tokens=torch.zeros((1, 2, 128), dtype=torch.float32),
            )

        def compress_evicted_tokens(self, ssm_state, evicted_video, evicted_audio):
            return ssm_state

    monkeypatch.setattr(orchestrator, "OfficialDistilledChunkRunner", lambda **kwargs: FakeRunner())

    def fake_encode_video(*, output_path, **kwargs):
        chunk_parent = Path(output_path).parent
        observed_chunk_dirs.append(chunk_parent)
        assert chunk_parent.exists()
        Path(output_path).write_bytes(b"mp4")

    monkeypatch.setattr(orchestrator, "encode_video", fake_encode_video)
    monkeypatch.setattr(
        orchestrator,
        "_stitch_chunk_files",
        lambda chunk_paths, output_path: output_path.write_bytes(b"stitched"),
        raising=False,
    )

    run_backbone_preview(_base_config(tmp_path, chunks_per_segment=1))

    assert observed_chunk_dirs == [tmp_path / "outputs" / "episode_0000"]


def test_stitch_chunk_files_falls_back_to_imageio_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunk_a = tmp_path / "chunk_a.mp4"
    chunk_b = tmp_path / "chunk_b.mp4"
    chunk_a.write_bytes(b"a")
    chunk_b.write_bytes(b"b")
    output_path = tmp_path / "stitched.mp4"
    recorded_commands: list[list[str]] = []

    monkeypatch.setattr(orchestrator.shutil, "which", lambda name: None)
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: "/tmp/fake-ffmpeg"),
    )

    def fake_run(command, check, stdout, stderr, text):
        recorded_commands.append(command)
        output_path.write_bytes(b"stitched")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    orchestrator._stitch_chunk_files([chunk_a, chunk_b], output_path)

    assert output_path.read_bytes() == b"stitched"
    assert recorded_commands[0][0] == "/tmp/fake-ffmpeg"
