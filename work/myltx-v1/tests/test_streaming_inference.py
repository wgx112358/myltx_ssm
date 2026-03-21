from __future__ import annotations

import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-core/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-trainer/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-pipelines/src"))

from self_forcing_data import SwitchEpisode, SwitchSegment, build_episode_chunk_plan  # noqa: E402
from streaming_inference import (  # noqa: E402
    StreamingConfig,
    apply_ssm_switch_state_decay,
    build_switch_generation_plan,
    run_switch_mode,
    select_reference_video,
    select_switch_recache_video,
    summarize_stream_state,
    stitch_generated_chunks,
)


def _episode() -> SwitchEpisode:
    return SwitchEpisode(
        episode_id="episode_0007",
        segments=(
            SwitchSegment(category="A", prompt="prompt-a", start_seconds=0.0, duration_seconds=2.0),
            SwitchSegment(category="B", prompt="prompt-b", start_seconds=2.0, duration_seconds=1.0),
            SwitchSegment(category="C", prompt="prompt-c", start_seconds=3.0, duration_seconds=1.0),
        ),
    )


def test_build_episode_chunk_plan_uses_segment_durations() -> None:
    plan = build_episode_chunk_plan(_episode(), num_chunks=4)

    assert plan["episode_id"] == "episode_0007"
    assert plan["segment_indices"] == [0, 0, 1, 2]
    assert plan["prompts"] == ["prompt-a", "prompt-a", "prompt-b", "prompt-c"]
    assert plan["switch_flags"] == [False, False, True, True]


def test_select_reference_video_keeps_recent_chunks_and_frame_cap() -> None:
    chunk_a = torch.full((3, 2, 2, 2), 1.0)
    chunk_b = torch.full((3, 2, 2, 2), 2.0)
    chunk_c = torch.full((3, 2, 2, 2), 3.0)

    reference = select_reference_video(
        generated_videos=[chunk_a, chunk_b, chunk_c],
        window_chunks=2,
        max_frames=3,
    )

    assert reference is not None
    assert tuple(reference.shape) == (3, 3, 2, 2)
    assert torch.all(reference[0] == 2.0)
    assert torch.all(reference[1] == 3.0)
    assert torch.all(reference[2] == 3.0)


def test_stitch_generated_chunks_concatenates_video_and_audio() -> None:
    video_a = torch.full((3, 2, 2, 2), 1.0)
    video_b = torch.full((3, 1, 2, 2), 2.0)
    audio_a = torch.full((2, 4), 10.0)
    audio_b = torch.full((2, 2), 20.0)

    stitched_video, stitched_audio = stitch_generated_chunks(
        videos=[video_a, video_b],
        audios=[audio_a, audio_b],
    )

    assert tuple(stitched_video.shape) == (3, 3, 2, 2)
    assert tuple(stitched_audio.shape) == (2, 6)
    assert torch.all(stitched_video[:, :2] == 1.0)
    assert torch.all(stitched_video[:, 2:] == 2.0)
    assert torch.all(stitched_audio[:, :4] == 10.0)
    assert torch.all(stitched_audio[:, 4:] == 20.0)


def test_build_switch_generation_plan_expands_episode_into_runtime_chunks() -> None:
    plan = build_switch_generation_plan(
        episode=_episode(),
        chunks_per_segment=2,
        chunk_num_frames=17,
        frame_rate=8.0,
        reference_window_chunks=2,
        reference_max_frames=17,
    )

    assert plan["episode_id"] == "episode_0007"
    assert plan["num_chunks"] == 6
    assert [chunk["prompt"] for chunk in plan["chunks"]] == [
        "prompt-a",
        "prompt-a",
        "prompt-b",
        "prompt-b",
        "prompt-c",
        "prompt-c",
    ]
    assert [chunk["prompt_switch"] for chunk in plan["chunks"]] == [False, False, True, False, True, False]
    assert plan["chunks"][0]["seed"] == 42
    assert plan["chunks"][5]["seed"] == 47


def test_run_switch_mode_plan_only_writes_metadata(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch.mp4"

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=2,
        chunk_num_frames=17,
        reference_window_chunks=2,
        reference_max_frames=17,
        plan_only=True,
    )

    run_switch_mode(config)

    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "switch"
    assert metadata["plan_only"] is True
    assert metadata["plan"]["episode_id"] == "episode_0007"
    assert metadata["plan"]["num_chunks"] == 4
    assert metadata["height"] == config.height
    assert metadata["width"] == config.width
    assert metadata["chunk_num_frames"] == config.chunk_num_frames
    assert metadata["frame_rate"] == config.frame_rate


def test_select_switch_recache_video_trims_to_valid_recent_frame_count() -> None:
    chunk_a = torch.full((3, 5, 2, 2), 1.0)
    chunk_b = torch.full((3, 5, 2, 2), 2.0)

    recache_video = select_switch_recache_video(
        generated_videos=[chunk_a, chunk_b],
        window_chunks=2,
        max_frames=10,
    )

    assert recache_video is not None
    assert tuple(recache_video.shape) == (9, 3, 2, 2)
    assert torch.all(recache_video[:4] == 1.0)
    assert torch.all(recache_video[4:] == 2.0)


def test_run_switch_mode_replaces_switch_reference_with_recached_history(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "switch.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch.mp4"

    calls: list[dict[str, object]] = []

    class DummyComponents:
        transformer = object()
        video_vae_encoder = object()
        video_vae_decoder = object()
        audio_vae_decoder = None
        vocoder = None

    @dataclass
    class FakeGenerationConfig:
        prompt: str
        height: int
        width: int
        num_frames: int
        frame_rate: float
        num_inference_steps: int
        guidance_scale: float
        seed: int
        reference_video: torch.Tensor | None = None
        reference_downscale_factor: int = 1
        generate_audio: bool = True
        cached_embeddings: object | None = None
        stg_scale: float = 0.0
        stg_blocks: list[int] | None = None
        condition_image: torch.Tensor | None = None
        disable_post_chunk_ssm_compression: bool = False
        ssm_window_blocks: int = 0
        ssm_window_blocks: int = 0
        disable_post_chunk_ssm_compression: bool = False

    class FakeSampler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(self, config, device: str = "cpu"):
            reference_video = config.reference_video.clone() if config.reference_video is not None else None
            condition_image = config.condition_image.clone() if config.condition_image is not None else None
            calls.append(
                {
                    "prompt": config.prompt,
                    "num_frames": config.num_frames,
                    "reference_video": reference_video,
                    "condition_image": condition_image,
                }
            )
            fill_value = 1.0
            if condition_image is not None and reference_video is None:
                fill_value = 9.0
            elif reference_video is not None:
                fill_value = 2.0
            video = torch.full((3, config.num_frames, 2, 2), fill_value)
            return video, None

    fake_baseline_audit = types.ModuleType("baseline_audit")
    fake_baseline_audit.load_prompt_cache = lambda **kwargs: {"prompt-a": object(), "prompt-b": object()}
    fake_baseline_audit.resolve_device = lambda device: device

    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = lambda **kwargs: DummyComponents()

    fake_validation_sampler = types.ModuleType("ltx_trainer.validation_sampler")
    fake_validation_sampler.GenerationConfig = FakeGenerationConfig
    fake_validation_sampler.ValidationSampler = FakeSampler

    fake_video_utils = types.ModuleType("ltx_trainer.video_utils")
    fake_video_utils.save_video = lambda **kwargs: None

    monkeypatch.setitem(sys.modules, "baseline_audit", fake_baseline_audit)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    monkeypatch.setitem(sys.modules, "ltx_trainer.validation_sampler", fake_validation_sampler)
    monkeypatch.setitem(sys.modules, "ltx_trainer.video_utils", fake_video_utils)

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=1,
        chunk_num_frames=17,
        reference_window_chunks=1,
        reference_max_frames=17,
        switch_recache_enabled=True,
        switch_recache_window_chunks=1,
        switch_recache_max_frames=17,
        skip_audio=True,
    )

    run_switch_mode(config)

    assert len(calls) == 3
    assert calls[0]["reference_video"] is None
    assert calls[1]["condition_image"] is not None
    assert calls[1]["reference_video"] is None
    assert calls[2]["reference_video"] is not None
    assert torch.all(calls[2]["reference_video"] == 9.0)

    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["chunk_outputs"][1]["reference_source"] == "switch_recache"
    assert metadata["chunk_outputs"][1]["switch_recache_source_frames"] == 17
    assert metadata["chunk_outputs"][1]["switch_recache_frames"] == 17


def test_run_switch_mode_ssm_streaming_uses_stream_state_and_skips_recache(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "switch.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch_ssm.mp4"
    calls: list[dict[str, object]] = []

    class DummyComponents:
        transformer = object()
        video_vae_encoder = object()
        video_vae_decoder = object()
        audio_vae_decoder = None
        vocoder = None

    @dataclass
    class FakeGenerationConfig:
        prompt: str
        height: int
        width: int
        num_frames: int
        frame_rate: float
        num_inference_steps: int
        guidance_scale: float
        seed: int
        reference_video: torch.Tensor | None = None
        reference_downscale_factor: int = 1
        generate_audio: bool = True
        cached_embeddings: object | None = None
        stg_scale: float = 0.0
        stg_blocks: list[int] | None = None
        condition_image: torch.Tensor | None = None
        disable_post_chunk_ssm_compression: bool = False
        ssm_window_blocks: int = 0

    class FakeSampler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(
            self,
            config,
            device: str = "cpu",
            stream_state=None,
            return_stream_state: bool = False,
        ):
            calls.append(
                {
                    "prompt": config.prompt,
                    "num_frames": config.num_frames,
                    "stream_state": stream_state,
                    "return_stream_state": return_stream_state,
                    "condition_image": config.condition_image,
                }
            )
            video = torch.full((3, config.num_frames, 2, 2), 1.0)
            next_stream_state = {"last_prompt": config.prompt}
            if return_stream_state:
                return video, None, next_stream_state
            return video, None

    @dataclass
    class FakeSSMConfig:
        enabled: bool = False
        d_state: int = 64
        gate_bias: float = -2.0

    class FakeSSMLayers:
        def load_state_dict(self, state_dict, strict: bool = False):
            return [], []

    class FakeAugmentedLTXModel:
        def __init__(self, base_model, ssm_config) -> None:
            self.base_model = base_model
            self.ssm_config = ssm_config
            self.ssm_layers = FakeSSMLayers()

        @classmethod
        def from_base(cls, base_model, ssm_config=None):
            return cls(base_model, ssm_config)

    fake_baseline_audit = types.ModuleType("baseline_audit")
    fake_baseline_audit.load_prompt_cache = lambda **kwargs: {"prompt-a": object(), "prompt-b": object()}
    fake_baseline_audit.resolve_device = lambda device: device

    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = lambda **kwargs: DummyComponents()

    fake_validation_sampler = types.ModuleType("ltx_trainer.validation_sampler")
    fake_validation_sampler.GenerationConfig = FakeGenerationConfig
    fake_validation_sampler.ValidationSampler = FakeSampler

    fake_video_utils = types.ModuleType("ltx_trainer.video_utils")
    fake_video_utils.save_video = lambda **kwargs: None

    fake_ssm_memory = types.ModuleType("ltx_core.model.transformer.ssm_memory")
    fake_ssm_memory.SSMConfig = FakeSSMConfig

    fake_ssm_integration = types.ModuleType("ltx_core.model.transformer.ssm_integration")
    fake_ssm_integration.SSMAugmentedLTXModel = FakeAugmentedLTXModel

    monkeypatch.setitem(sys.modules, "baseline_audit", fake_baseline_audit)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    monkeypatch.setitem(sys.modules, "ltx_trainer.validation_sampler", fake_validation_sampler)
    monkeypatch.setitem(sys.modules, "ltx_trainer.video_utils", fake_video_utils)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_memory", fake_ssm_memory)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_integration", fake_ssm_integration)

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=1,
        chunk_num_frames=17,
        reference_window_chunks=1,
        reference_max_frames=17,
        switch_recache_enabled=True,
        switch_recache_window_chunks=1,
        switch_recache_max_frames=17,
        ssm_streaming_enabled=True,
        skip_audio=True,
    )

    run_switch_mode(config)

    assert len(calls) == 2
    assert calls[0]["condition_image"] is None
    assert calls[0]["return_stream_state"] is True
    assert calls[0]["stream_state"] is None
    assert calls[1]["return_stream_state"] is True
    assert calls[1]["stream_state"] == {"last_prompt": "prompt-a"}

    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["memory_mode"] == "ssm_streaming"
    assert metadata["chunk_outputs"][1]["reference_source"] == "none"
    assert metadata["chunk_outputs"][1]["stream_state_enabled"] is True


def test_run_switch_mode_snapshots_chunk_tensors_before_stitching(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "switch_snapshot.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                    {"category": "C", "prompt": "prompt-c", "start_seconds": 2.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch_snapshot.mp4"
    saved_videos: list[tuple[str, torch.Tensor]] = []

    class DummyComponents:
        transformer = object()
        video_vae_encoder = object()
        video_vae_decoder = object()
        audio_vae_decoder = None
        vocoder = None

    @dataclass
    class FakeGenerationConfig:
        prompt: str
        height: int
        width: int
        num_frames: int
        frame_rate: float
        num_inference_steps: int
        guidance_scale: float
        seed: int
        reference_video: torch.Tensor | None = None
        reference_downscale_factor: int = 1
        generate_audio: bool = True
        cached_embeddings: object | None = None
        stg_scale: float = 0.0
        stg_blocks: list[int] | None = None
        condition_image: torch.Tensor | None = None
        disable_post_chunk_ssm_compression: bool = False

    class FakeSampler:
        def __init__(self, *args, **kwargs) -> None:
            self.shared_video = torch.zeros((3, 17, 2, 2))
            self.calls = 0

        def generate(self, config, device: str = "cpu"):
            self.calls += 1
            self.shared_video.fill_(float(self.calls))
            return self.shared_video, None

    def fake_save_video(*, video_tensor, output_path, **kwargs) -> None:
        saved_videos.append((Path(output_path).name, video_tensor.clone()))

    fake_baseline_audit = types.ModuleType("baseline_audit")
    fake_baseline_audit.load_prompt_cache = lambda **kwargs: {
        "prompt-a": object(),
        "prompt-b": object(),
        "prompt-c": object(),
    }
    fake_baseline_audit.resolve_device = lambda device: device

    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = lambda **kwargs: DummyComponents()

    fake_validation_sampler = types.ModuleType("ltx_trainer.validation_sampler")
    fake_validation_sampler.GenerationConfig = FakeGenerationConfig
    fake_validation_sampler.ValidationSampler = FakeSampler

    fake_video_utils = types.ModuleType("ltx_trainer.video_utils")
    fake_video_utils.save_video = fake_save_video

    monkeypatch.setitem(sys.modules, "baseline_audit", fake_baseline_audit)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    monkeypatch.setitem(sys.modules, "ltx_trainer.validation_sampler", fake_validation_sampler)
    monkeypatch.setitem(sys.modules, "ltx_trainer.video_utils", fake_video_utils)

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=1,
        chunk_num_frames=17,
        reference_window_chunks=0,
        reference_max_frames=17,
        switch_recache_enabled=False,
        skip_audio=True,
    )

    run_switch_mode(config)

    assert [name for name, _ in saved_videos] == [
        "chunk_000.mp4",
        "chunk_001.mp4",
        "chunk_002.mp4",
        "switch_snapshot.mp4",
    ]
    stitched_video = saved_videos[-1][1]
    assert torch.all(stitched_video[:, :17] == 1.0)
    assert torch.all(stitched_video[:, 17:34] == 2.0)
    assert torch.all(stitched_video[:, 34:] == 3.0)


def test_apply_ssm_switch_state_decay_scales_nested_ssm_state() -> None:
    class FakeSSMState:
        def __init__(self) -> None:
            self.states = {(0, "video"): torch.full((1, 1, 1), 8.0)}

        def scale_(self, factor: float):
            self.states[(0, "video")] = self.states[(0, "video")] * factor
            return self

    class FakeStreamState:
        def __init__(self) -> None:
            self.ssm_state = FakeSSMState()

    stream_state = FakeStreamState()
    applied = apply_ssm_switch_state_decay(stream_state, 0.25)

    assert applied is True
    assert torch.all(stream_state.ssm_state.states[(0, "video")] == 2.0)


def test_summarize_stream_state_reports_tensor_stats() -> None:
    class FakeSSMState:
        def __init__(self) -> None:
            self.states = {
                (0, "video"): torch.tensor([[[1.0, -3.0]]]),
                (1, "audio"): torch.tensor([[[2.0]]]),
            }

    class FakeStreamState:
        def __init__(self) -> None:
            self.ssm_state = FakeSSMState()

    stats = summarize_stream_state(FakeStreamState())

    assert stats["num_tensors"] == 2
    assert stats["total_numel"] == 3
    assert stats["max_abs"] == 3.0
    assert stats["mean_abs"] == 2.0


def test_summarize_stream_state_reports_pending_chunk_counts_without_ssm_state() -> None:
    class FakeStreamState:
        def __init__(self) -> None:
            self.pending_video_chunks = [torch.zeros((1, 1, 1)), torch.zeros((1, 1, 1))]
            self.pending_audio_chunks = [None, None]

    stats = summarize_stream_state(FakeStreamState())

    assert stats["num_tensors"] == 0
    assert stats["pending_video_chunks"] == 2
    assert stats["pending_audio_chunks"] == 2


def test_run_switch_mode_ssm_streaming_passes_disable_post_compress_and_records_state_stats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "switch_disable_compress.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch_disable_compress.mp4"
    calls: list[dict[str, object]] = []

    class DummyComponents:
        transformer = object()
        video_vae_encoder = object()
        video_vae_decoder = object()
        audio_vae_decoder = None
        vocoder = None

    class FakeInnerSSMState:
        def __init__(self, value: float) -> None:
            self.states = {(0, "video"): torch.full((1, 1, 1), value)}

    class FakeStreamState:
        def __init__(self, value: float) -> None:
            self.ssm_state = FakeInnerSSMState(value)

    @dataclass
    class FakeGenerationConfig:
        prompt: str
        height: int
        width: int
        num_frames: int
        frame_rate: float
        num_inference_steps: int
        guidance_scale: float
        seed: int
        reference_video: torch.Tensor | None = None
        reference_downscale_factor: int = 1
        generate_audio: bool = True
        cached_embeddings: object | None = None
        stg_scale: float = 0.0
        stg_blocks: list[int] | None = None
        condition_image: torch.Tensor | None = None
        disable_post_chunk_ssm_compression: bool = False
        disable_post_chunk_ssm_compression: bool = False
        ssm_window_blocks: int = 0

    class FakeSampler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(
            self,
            config,
            device: str = "cpu",
            stream_state=None,
            return_stream_state: bool = False,
        ):
            calls.append(
                {
                    "prompt": config.prompt,
                    "disable_post_chunk_ssm_compression": config.disable_post_chunk_ssm_compression,
                    "return_stream_state": return_stream_state,
                }
            )
            video = torch.full((3, config.num_frames, 2, 2), 1.0)
            next_stream_state = FakeStreamState(4.0)
            if return_stream_state:
                return video, None, next_stream_state
            return video, None

    @dataclass
    class FakeSSMConfig:
        enabled: bool = False
        d_state: int = 64
        gate_bias: float = -2.0

    class FakeSSMLayers:
        def load_state_dict(self, state_dict, strict: bool = False):
            return [], []

    class FakeAugmentedLTXModel:
        def __init__(self, base_model, ssm_config) -> None:
            self.base_model = base_model
            self.ssm_config = ssm_config
            self.ssm_layers = FakeSSMLayers()

        @classmethod
        def from_base(cls, base_model, ssm_config=None):
            return cls(base_model, ssm_config)

    fake_baseline_audit = types.ModuleType("baseline_audit")
    fake_baseline_audit.load_prompt_cache = lambda **kwargs: {"prompt-a": object(), "prompt-b": object()}
    fake_baseline_audit.resolve_device = lambda device: device

    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = lambda **kwargs: DummyComponents()

    fake_validation_sampler = types.ModuleType("ltx_trainer.validation_sampler")
    fake_validation_sampler.GenerationConfig = FakeGenerationConfig
    fake_validation_sampler.ValidationSampler = FakeSampler

    fake_video_utils = types.ModuleType("ltx_trainer.video_utils")
    fake_video_utils.save_video = lambda **kwargs: None

    fake_ssm_memory = types.ModuleType("ltx_core.model.transformer.ssm_memory")
    fake_ssm_memory.SSMConfig = FakeSSMConfig

    fake_ssm_integration = types.ModuleType("ltx_core.model.transformer.ssm_integration")
    fake_ssm_integration.SSMAugmentedLTXModel = FakeAugmentedLTXModel

    monkeypatch.setitem(sys.modules, "baseline_audit", fake_baseline_audit)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    monkeypatch.setitem(sys.modules, "ltx_trainer.validation_sampler", fake_validation_sampler)
    monkeypatch.setitem(sys.modules, "ltx_trainer.video_utils", fake_video_utils)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_memory", fake_ssm_memory)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_integration", fake_ssm_integration)

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=1,
        chunk_num_frames=17,
        reference_window_chunks=1,
        reference_max_frames=17,
        switch_recache_enabled=True,
        switch_recache_window_chunks=1,
        switch_recache_max_frames=17,
        ssm_streaming_enabled=True,
        ssm_disable_post_chunk_compression=True,
        skip_audio=True,
    )

    run_switch_mode(config)

    assert len(calls) == 2
    assert calls[0]["disable_post_chunk_ssm_compression"] is True
    assert calls[1]["disable_post_chunk_ssm_compression"] is True

    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["ssm_disable_post_chunk_compression"] is True
    assert metadata["chunk_outputs"][0]["stream_state_stats"]["num_tensors"] == 1
    assert metadata["chunk_outputs"][0]["stream_state_stats"]["max_abs"] == 4.0


def test_run_switch_mode_ssm_streaming_applies_switch_state_decay(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "switch_decay.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch_ssm_decay.mp4"
    calls: list[dict[str, object]] = []

    class DummyComponents:
        transformer = object()
        video_vae_encoder = object()
        video_vae_decoder = object()
        audio_vae_decoder = None
        vocoder = None

    class FakeInnerSSMState:
        def __init__(self, value: float) -> None:
            self.states = {(0, "video"): torch.full((1, 1, 1), value)}

        def scale_(self, factor: float):
            self.states[(0, "video")] = self.states[(0, "video")] * factor
            return self

    class FakeStreamState:
        def __init__(self, value: float) -> None:
            self.ssm_state = FakeInnerSSMState(value)

    @dataclass
    class FakeGenerationConfig:
        prompt: str
        height: int
        width: int
        num_frames: int
        frame_rate: float
        num_inference_steps: int
        guidance_scale: float
        seed: int
        reference_video: torch.Tensor | None = None
        reference_downscale_factor: int = 1
        generate_audio: bool = True
        cached_embeddings: object | None = None
        stg_scale: float = 0.0
        stg_blocks: list[int] | None = None
        condition_image: torch.Tensor | None = None
        disable_post_chunk_ssm_compression: bool = False
        ssm_window_blocks: int = 0

    class FakeSampler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(
            self,
            config,
            device: str = "cpu",
            stream_state=None,
            return_stream_state: bool = False,
        ):
            incoming = None
            if stream_state is not None:
                incoming = stream_state.ssm_state.states[(0, "video")].clone()
            calls.append(
                {
                    "prompt": config.prompt,
                    "incoming_state": incoming,
                    "return_stream_state": return_stream_state,
                }
            )
            video = torch.full((3, config.num_frames, 2, 2), 1.0)
            next_stream_state = FakeStreamState(8.0)
            if return_stream_state:
                return video, None, next_stream_state
            return video, None

    @dataclass
    class FakeSSMConfig:
        enabled: bool = False
        d_state: int = 64
        gate_bias: float = -2.0

    class FakeSSMLayers:
        def load_state_dict(self, state_dict, strict: bool = False):
            return [], []

    class FakeAugmentedLTXModel:
        def __init__(self, base_model, ssm_config) -> None:
            self.base_model = base_model
            self.ssm_config = ssm_config
            self.ssm_layers = FakeSSMLayers()

        @classmethod
        def from_base(cls, base_model, ssm_config=None):
            return cls(base_model, ssm_config)

    fake_baseline_audit = types.ModuleType("baseline_audit")
    fake_baseline_audit.load_prompt_cache = lambda **kwargs: {"prompt-a": object(), "prompt-b": object()}
    fake_baseline_audit.resolve_device = lambda device: device

    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = lambda **kwargs: DummyComponents()

    fake_validation_sampler = types.ModuleType("ltx_trainer.validation_sampler")
    fake_validation_sampler.GenerationConfig = FakeGenerationConfig
    fake_validation_sampler.ValidationSampler = FakeSampler

    fake_video_utils = types.ModuleType("ltx_trainer.video_utils")
    fake_video_utils.save_video = lambda **kwargs: None

    fake_ssm_memory = types.ModuleType("ltx_core.model.transformer.ssm_memory")
    fake_ssm_memory.SSMConfig = FakeSSMConfig

    fake_ssm_integration = types.ModuleType("ltx_core.model.transformer.ssm_integration")
    fake_ssm_integration.SSMAugmentedLTXModel = FakeAugmentedLTXModel

    monkeypatch.setitem(sys.modules, "baseline_audit", fake_baseline_audit)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    monkeypatch.setitem(sys.modules, "ltx_trainer.validation_sampler", fake_validation_sampler)
    monkeypatch.setitem(sys.modules, "ltx_trainer.video_utils", fake_video_utils)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_memory", fake_ssm_memory)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_integration", fake_ssm_integration)

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=1,
        chunk_num_frames=17,
        reference_window_chunks=1,
        reference_max_frames=17,
        switch_recache_enabled=True,
        switch_recache_window_chunks=1,
        switch_recache_max_frames=17,
        ssm_streaming_enabled=True,
        ssm_switch_state_decay=0.25,
        skip_audio=True,
    )

    run_switch_mode(config)

    assert len(calls) == 2
    assert calls[0]["incoming_state"] is None
    assert calls[1]["incoming_state"] is not None
    assert torch.all(calls[1]["incoming_state"] == 2.0)

    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["ssm_switch_state_decay"] == 0.25
    assert metadata["chunk_outputs"][1]["ssm_switch_state_decay_applied"] is True


def test_run_switch_mode_ssm_streaming_passes_window_blocks_to_generation_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "switch_window_blocks.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch_window_blocks.mp4"
    calls: list[dict[str, object]] = []

    class DummyComponents:
        transformer = object()
        video_vae_encoder = object()
        video_vae_decoder = object()
        audio_vae_decoder = None
        vocoder = None

    class FakeStreamState:
        def __init__(self) -> None:
            self.ssm_state = object()

    @dataclass
    class FakeGenerationConfig:
        prompt: str
        height: int
        width: int
        num_frames: int
        frame_rate: float
        num_inference_steps: int
        guidance_scale: float
        seed: int
        reference_video: torch.Tensor | None = None
        reference_downscale_factor: int = 1
        generate_audio: bool = True
        cached_embeddings: object | None = None
        stg_scale: float = 0.0
        stg_blocks: list[int] | None = None
        condition_image: torch.Tensor | None = None
        disable_post_chunk_ssm_compression: bool = False
        ssm_window_blocks: int = 0

    class FakeSampler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(
            self,
            config,
            device: str = "cpu",
            stream_state=None,
            return_stream_state: bool = False,
        ):
            calls.append(
                {
                    "prompt": config.prompt,
                    "ssm_window_blocks": config.ssm_window_blocks,
                    "return_stream_state": return_stream_state,
                }
            )
            video = torch.full((3, config.num_frames, 2, 2), 1.0)
            next_stream_state = FakeStreamState()
            if return_stream_state:
                return video, None, next_stream_state
            return video, None

    @dataclass
    class FakeSSMConfig:
        enabled: bool = False
        d_state: int = 64
        gate_bias: float = -2.0

    class FakeSSMLayers:
        def load_state_dict(self, state_dict, strict: bool = False):
            return [], []

    class FakeAugmentedLTXModel:
        def __init__(self, base_model, ssm_config) -> None:
            self.base_model = base_model
            self.ssm_config = ssm_config
            self.ssm_layers = FakeSSMLayers()

        @classmethod
        def from_base(cls, base_model, ssm_config=None):
            return cls(base_model, ssm_config)

    fake_baseline_audit = types.ModuleType("baseline_audit")
    fake_baseline_audit.load_prompt_cache = lambda **kwargs: {"prompt-a": object(), "prompt-b": object()}
    fake_baseline_audit.resolve_device = lambda device: device

    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = lambda **kwargs: DummyComponents()

    fake_validation_sampler = types.ModuleType("ltx_trainer.validation_sampler")
    fake_validation_sampler.GenerationConfig = FakeGenerationConfig
    fake_validation_sampler.ValidationSampler = FakeSampler

    fake_video_utils = types.ModuleType("ltx_trainer.video_utils")
    fake_video_utils.save_video = lambda **kwargs: None

    fake_ssm_memory = types.ModuleType("ltx_core.model.transformer.ssm_memory")
    fake_ssm_memory.SSMConfig = FakeSSMConfig

    fake_ssm_integration = types.ModuleType("ltx_core.model.transformer.ssm_integration")
    fake_ssm_integration.SSMAugmentedLTXModel = FakeAugmentedLTXModel

    monkeypatch.setitem(sys.modules, "baseline_audit", fake_baseline_audit)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    monkeypatch.setitem(sys.modules, "ltx_trainer.validation_sampler", fake_validation_sampler)
    monkeypatch.setitem(sys.modules, "ltx_trainer.video_utils", fake_video_utils)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_memory", fake_ssm_memory)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_integration", fake_ssm_integration)

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=1,
        chunk_num_frames=17,
        window_blocks=2,
        reference_window_chunks=1,
        reference_max_frames=17,
        switch_recache_enabled=True,
        switch_recache_window_chunks=1,
        switch_recache_max_frames=17,
        ssm_streaming_enabled=True,
        skip_audio=True,
    )

    run_switch_mode(config)

    assert len(calls) == 2
    assert calls[0]["return_stream_state"] is True
    assert calls[0]["ssm_window_blocks"] == 2
    assert calls[1]["ssm_window_blocks"] == 2

    metadata = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["window_blocks"] == 2


def test_run_switch_mode_ssm_streaming_logs_pending_chunk_counts(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(logging.INFO)
    manifest_path = tmp_path / "switch_pending_counts.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "segments": [
                    {"category": "A", "prompt": "prompt-a", "start_seconds": 0.0, "duration_seconds": 1.0},
                    {"category": "B", "prompt": "prompt-b", "start_seconds": 1.0, "duration_seconds": 1.0},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "switch_pending_counts.mp4"

    class DummyComponents:
        transformer = object()
        video_vae_encoder = object()
        video_vae_decoder = object()
        audio_vae_decoder = None
        vocoder = None

    class FakeStreamState:
        def __init__(self) -> None:
            self.pending_video_chunks = [torch.zeros((1, 1, 1)), torch.zeros((1, 1, 1))]
            self.pending_audio_chunks = [None, None]

    @dataclass
    class FakeGenerationConfig:
        prompt: str
        height: int
        width: int
        num_frames: int
        frame_rate: float
        num_inference_steps: int
        guidance_scale: float
        seed: int
        reference_video: torch.Tensor | None = None
        reference_downscale_factor: int = 1
        generate_audio: bool = True
        cached_embeddings: object | None = None
        stg_scale: float = 0.0
        stg_blocks: list[int] | None = None
        condition_image: torch.Tensor | None = None
        disable_post_chunk_ssm_compression: bool = False
        ssm_window_blocks: int = 0

    class FakeSampler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def generate(
            self,
            config,
            device: str = "cpu",
            stream_state=None,
            return_stream_state: bool = False,
        ):
            video = torch.full((3, config.num_frames, 2, 2), 1.0)
            next_stream_state = FakeStreamState()
            if return_stream_state:
                return video, None, next_stream_state
            return video, None

    @dataclass
    class FakeSSMConfig:
        enabled: bool = False
        d_state: int = 64
        gate_bias: float = -2.0

    class FakeSSMLayers:
        def load_state_dict(self, state_dict, strict: bool = False):
            return [], []

    class FakeAugmentedLTXModel:
        def __init__(self, base_model, ssm_config) -> None:
            self.base_model = base_model
            self.ssm_config = ssm_config
            self.ssm_layers = FakeSSMLayers()

        @classmethod
        def from_base(cls, base_model, ssm_config=None):
            return cls(base_model, ssm_config)

    fake_baseline_audit = types.ModuleType("baseline_audit")
    fake_baseline_audit.load_prompt_cache = lambda **kwargs: {"prompt-a": object(), "prompt-b": object()}
    fake_baseline_audit.resolve_device = lambda device: device

    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = lambda **kwargs: DummyComponents()

    fake_validation_sampler = types.ModuleType("ltx_trainer.validation_sampler")
    fake_validation_sampler.GenerationConfig = FakeGenerationConfig
    fake_validation_sampler.ValidationSampler = FakeSampler

    fake_video_utils = types.ModuleType("ltx_trainer.video_utils")
    fake_video_utils.save_video = lambda **kwargs: None

    fake_ssm_memory = types.ModuleType("ltx_core.model.transformer.ssm_memory")
    fake_ssm_memory.SSMConfig = FakeSSMConfig

    fake_ssm_integration = types.ModuleType("ltx_core.model.transformer.ssm_integration")
    fake_ssm_integration.SSMAugmentedLTXModel = FakeAugmentedLTXModel

    monkeypatch.setitem(sys.modules, "baseline_audit", fake_baseline_audit)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    monkeypatch.setitem(sys.modules, "ltx_trainer.validation_sampler", fake_validation_sampler)
    monkeypatch.setitem(sys.modules, "ltx_trainer.video_utils", fake_video_utils)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_memory", fake_ssm_memory)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_integration", fake_ssm_integration)

    config = StreamingConfig(
        mode="switch",
        output_path=str(output_path),
        frame_rate=8.0,
        manifest_path=str(manifest_path),
        episode_id="episode_0007",
        chunks_per_segment=1,
        chunk_num_frames=17,
        window_blocks=2,
        reference_window_chunks=0,
        reference_max_frames=17,
        switch_recache_enabled=False,
        ssm_streaming_enabled=True,
        skip_audio=True,
    )

    run_switch_mode(config)

    assert "pending_video_chunks=2" in caplog.text
    assert "pending_audio_chunks=2" in caplog.text
