from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-core/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-pipelines/src"))

import ltx_pipelines.distilled_streaming as distilled_streaming  # noqa: E402
from ltx_core.types import LatentState  # noqa: E402
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES  # noqa: E402
from ltx_pipelines.distilled_streaming import (  # noqa: E402
    OfficialDistilledChunkConfig,
    OfficialDistilledChunkRunner,
)


def _base_config(**overrides: object) -> OfficialDistilledChunkConfig:
    values: dict[str, object] = {
        "preset": "small",
        "distilled_checkpoint_path": "/tmp/ltx-2.3-distilled.safetensors",
        "gemma_root": "/tmp/gemma",
        "spatial_upsampler_path": "/tmp/ltx-2.3-spatial-upscaler.safetensors",
        "num_frames": 121,
        "frame_rate": 24.0,
        "prompt": "test prompt",
        "seed": 42,
    }
    values.update(overrides)
    return OfficialDistilledChunkConfig(
        **values,
    )


def test_chunk_config_accepts_only_phase1_small_geometry() -> None:
    config = _base_config()

    assert config.preset == "small"
    assert config.height == 1024
    assert config.width == 1536


def test_chunk_config_rejects_non_small_preset() -> None:
    with pytest.raises(ValueError, match="small"):
        _base_config(preset="hq")


def test_chunk_config_rejects_non_8k_plus_1_frames() -> None:
    with pytest.raises(ValueError, match="8\\*K\\+1"):
        _base_config(num_frames=40)


def test_runner_exposes_single_run_chunk_boundary() -> None:
    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )

    assert callable(runner.run_chunk)


def _latent_state(latent: torch.Tensor) -> LatentState:
    return LatentState(
        latent=latent,
        denoise_mask=torch.ones_like(latent),
        positions=torch.zeros((latent.shape[0], 1, 1), dtype=latent.dtype),
        clean_latent=latent.clone(),
        attention_mask=None,
    )


def _install_fake_runner_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    counts: dict[str, int] = {
        "model_ledger_init": 0,
        "transformer": 0,
        "video_encoder": 0,
        "spatial_upsampler": 0,
        "video_decoder": 0,
        "audio_decoder": 0,
        "vocoder": 0,
        "encode_prompts": 0,
    }
    calls: list[dict[str, object]] = []
    stage1_video_latent = torch.full((1, 128, 2, 1, 1), 3.0)
    stage1_audio_latent = torch.full((1, 1, 2, 128), 5.0)
    stage2_video_latent = torch.full((1, 128, 2, 1, 1), 7.0)
    stage2_audio_latent = torch.full((1, 1, 2, 128), 11.0)
    upscaled_video_latent = torch.full((1, 128, 2, 1, 1), 13.0)

    class FakePatchifier:
        def patchify(self, latent: torch.Tensor) -> torch.Tensor:
            if latent.ndim == 5:
                return latent.permute(0, 2, 3, 4, 1).reshape(latent.shape[0], -1, latent.shape[1])
            return latent.permute(0, 2, 1, 3).reshape(latent.shape[0], -1, latent.shape[1] * latent.shape[3])

    class FakeStreamingBackend:
        def __init__(self) -> None:
            self.compress_calls: list[tuple[object, object | None, object | None]] = []

        def compress_evicted_tokens(
            self,
            ssm_state: object,
            evicted_video: object | None,
            evicted_audio: object | None,
        ) -> object:
            self.compress_calls.append((ssm_state, evicted_video, evicted_audio))
            return {"compressed_from": ssm_state}

    class FakeTransformer:
        def __init__(self) -> None:
            self.velocity_model = FakeStreamingBackend()

    class FakeModelLedger:
        def __init__(self, **kwargs) -> None:
            counts["model_ledger_init"] += 1
            self.kwargs = kwargs

        def transformer(self) -> FakeTransformer:
            counts["transformer"] += 1
            return FakeTransformer()

        def video_encoder(self) -> str:
            counts["video_encoder"] += 1
            return "video-encoder"

        def spatial_upsampler(self) -> str:
            counts["spatial_upsampler"] += 1
            return "spatial-upsampler"

        def video_decoder(self) -> str:
            counts["video_decoder"] += 1
            return "video-decoder"

        def audio_decoder(self) -> str:
            counts["audio_decoder"] += 1
            return "audio-decoder"

        def vocoder(self) -> str:
            counts["vocoder"] += 1
            return "vocoder"

    class FakePipelineComponents:
        def __init__(self, **kwargs) -> None:
            self.dtype = kwargs["dtype"]
            self.device = kwargs["device"]
            self.video_patchifier = FakePatchifier()
            self.audio_patchifier = FakePatchifier()

    def fake_encode_prompts(*args, **kwargs):
        counts["encode_prompts"] += 1
        return [types.SimpleNamespace(video_encoding=torch.tensor([1.0]), audio_encoding=torch.tensor([2.0]))]

    def fake_combined_image_conditionings(**kwargs):
        return []

    def fake_denoise_audio_video(**kwargs):
        calls.append(kwargs)
        if len(calls) % 2 == 1:
            return _latent_state(stage1_video_latent), _latent_state(stage1_audio_latent)
        return _latent_state(stage2_video_latent), _latent_state(stage2_audio_latent)

    def fake_upsample_video(*, latent, video_encoder, upsampler):
        assert video_encoder == "video-encoder"
        assert upsampler == "spatial-upsampler"
        assert torch.equal(latent, stage1_video_latent[:1])
        return upscaled_video_latent

    monkeypatch.setattr(distilled_streaming, "ModelLedger", FakeModelLedger)
    monkeypatch.setattr(distilled_streaming, "PipelineComponents", FakePipelineComponents)
    monkeypatch.setattr(distilled_streaming, "encode_prompts", fake_encode_prompts)
    monkeypatch.setattr(distilled_streaming, "combined_image_conditionings", fake_combined_image_conditionings)
    monkeypatch.setattr(distilled_streaming, "denoise_audio_video", fake_denoise_audio_video)
    monkeypatch.setattr(distilled_streaming, "upsample_video", fake_upsample_video)
    monkeypatch.setattr(distilled_streaming, "vae_decode_video", lambda *args, **kwargs: "decoded-video")
    monkeypatch.setattr(distilled_streaming, "vae_decode_audio", lambda *args, **kwargs: "decoded-audio")
    monkeypatch.setattr(distilled_streaming, "cleanup_memory", lambda: None)
    monkeypatch.setattr(distilled_streaming.torch.cuda, "synchronize", lambda: None, raising=False)

    return {
        "counts": counts,
        "calls": calls,
        "stage1_audio_latent": stage1_audio_latent,
        "stage2_video_latent": stage2_video_latent,
        "stage2_audio_latent": stage2_audio_latent,
        "upscaled_video_latent": upscaled_video_latent,
    }


def test_run_chunk_uses_exact_distilled_stage2_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = _install_fake_runner_dependencies(monkeypatch)
    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )

    result = runner.run_chunk(_base_config())

    calls = fake_env["calls"]
    assert len(calls) == 2
    assert torch.equal(calls[0]["sigmas"], torch.tensor(DISTILLED_SIGMA_VALUES))
    assert torch.equal(calls[1]["sigmas"], torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES))
    assert calls[1]["initial_video_latent"] is fake_env["upscaled_video_latent"]
    assert calls[1]["initial_audio_latent"] is fake_env["stage1_audio_latent"]
    assert calls[1]["noise_scale"] == pytest.approx(float(STAGE_2_DISTILLED_SIGMA_VALUES[0]))
    assert result.final_chunk_video == "decoded-video"
    assert result.final_chunk_audio == "decoded-audio"
    assert tuple(result.evictable_video_tokens.shape) == (1, 2, 128)
    assert tuple(result.evictable_audio_tokens.shape) == (1, 2, 128)
    assert torch.all(result.evictable_video_tokens == 7.0)
    assert torch.all(result.evictable_audio_tokens == 11.0)


def test_run_chunk_reuses_heavy_components_across_multiple_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = _install_fake_runner_dependencies(monkeypatch)
    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )
    config = _base_config()

    runner.run_chunk(config)
    runner.run_chunk(config)

    counts = fake_env["counts"]
    assert counts["model_ledger_init"] == 1
    assert counts["transformer"] == 1
    assert counts["video_encoder"] == 1
    assert counts["spatial_upsampler"] == 1
    assert counts["video_decoder"] == 1
    assert counts["audio_decoder"] == 1
    assert counts["vocoder"] == 1
    assert counts["encode_prompts"] == 2


def test_run_chunk_threads_ssm_state_only_through_stage2(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_env = _install_fake_runner_dependencies(monkeypatch)
    stage2_calls: list[dict[str, object]] = []

    def fake_stateful_stage2(**kwargs):
        stage2_calls.append(kwargs)
        return (
            _latent_state(fake_env["stage2_video_latent"]),
            _latent_state(fake_env["stage2_audio_latent"]),
            "state1",
        )

    monkeypatch.setattr(distilled_streaming, "_run_stage2_chunk_with_ssm", fake_stateful_stage2, raising=False)
    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )

    result = runner.run_chunk(_base_config(ssm_streaming_enabled=True), ssm_state="state0")

    assert result.next_ssm_state == "state1"
    assert len(fake_env["calls"]) == 1
    assert len(stage2_calls) == 1
    assert stage2_calls[0]["ssm_state"] == "state0"
    assert stage2_calls[0]["initial_audio_latent"] is fake_env["stage1_audio_latent"]


def test_runner_exposes_compress_evicted_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runner_dependencies(monkeypatch)
    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )

    compressed = runner.compress_evicted_tokens(
        "state0",
        torch.ones((1, 2, 128)),
        torch.ones((1, 2, 128)),
    )

    assert compressed == {"compressed_from": "state0"}


def test_runner_compress_evicted_tokens_normalizes_snapshot_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runner_dependencies(monkeypatch)
    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )

    class FakeTokens:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.device, torch.dtype]] = []

        def to(self, *, device: torch.device, dtype: torch.dtype):
            self.calls.append((device, dtype))
            return {"device": device, "dtype": dtype}

    fake_video = FakeTokens()
    fake_audio = FakeTokens()

    runner.compress_evicted_tokens("state0", fake_video, fake_audio)

    assert fake_video.calls == [(runner.device, runner.dtype)]
    assert fake_audio.calls == [(runner.device, runner.dtype)]


def test_stage2_ssm_wrapper_moves_new_modules_to_runner_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBaseModel:
        pass

    class FakeSSMLayers:
        def load_state_dict(self, state_dict, strict: bool = False):
            return [], []

    class FakeAugmentedModel:
        def __init__(self, base_model, ssm_config) -> None:
            self.base_model = base_model
            self.ssm_config = ssm_config
            self.ssm_layers = FakeSSMLayers()

        @classmethod
        def from_base(cls, base_model, ssm_config=None):
            return cls(base_model, ssm_config)

    class FakeSSMConfig:
        def __init__(self, enabled: bool, d_state: int, gate_bias: float) -> None:
            self.enabled = enabled
            self.d_state = d_state
            self.gate_bias = gate_bias

    class FakeX0Model:
        def __init__(self, velocity_model) -> None:
            self.velocity_model = velocity_model
            self.to_calls: list[tuple[torch.device, torch.dtype]] = []

        def to(self, *, device: torch.device, dtype: torch.dtype):
            self.to_calls.append((device, dtype))
            return self

    fake_ssm_integration = types.ModuleType("ltx_core.model.transformer.ssm_integration")
    fake_ssm_integration.SSMAugmentedLTXModel = FakeAugmentedModel
    fake_ssm_memory = types.ModuleType("ltx_core.model.transformer.ssm_memory")
    fake_ssm_memory.SSMConfig = FakeSSMConfig

    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_integration", fake_ssm_integration)
    monkeypatch.setitem(sys.modules, "ltx_core.model.transformer.ssm_memory", fake_ssm_memory)
    monkeypatch.setattr(distilled_streaming, "X0Model", FakeX0Model)

    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )
    runner.model_ledger = types.SimpleNamespace(
        video_encoder=lambda: object(),
        spatial_upsampler=lambda: object(),
        video_decoder=lambda: object(),
        audio_decoder=lambda: object(),
        vocoder=lambda: object(),
    )
    runner._stage1_transformer = FakeX0Model(FakeBaseModel())

    stage2_transformer = runner._get_stage2_transformer(_base_config(ssm_streaming_enabled=True))

    assert stage2_transformer.to_calls == [(runner.device, runner.dtype)]


def test_run_chunk_does_not_pass_inference_tensors_into_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runner_dependencies(monkeypatch)
    video_decode_is_inference: list[bool] = []
    audio_decode_is_inference: list[bool] = []

    def fake_denoise_audio_video(**kwargs):
        latent = kwargs.get("initial_video_latent")
        if latent is None:
            return (
                _latent_state(torch.full((1, 128, 2, 1, 1), 3.0) + 0),
                _latent_state(torch.full((1, 1, 2, 128), 5.0) + 0),
            )
        return (
            _latent_state(torch.full((1, 128, 2, 1, 1), 7.0) + 0),
            _latent_state(torch.full((1, 1, 2, 128), 11.0) + 0),
        )

    def fake_upsample_video(*, latent, video_encoder, upsampler):
        return latent + 10.0

    def fake_decode_video(*args, **kwargs):
        latent = args[0] if args else kwargs["latent"]
        video_decode_is_inference.append(latent.is_inference())
        return "decoded-video"

    def fake_decode_audio(*args, **kwargs):
        latent = args[0] if args else kwargs["latent"]
        audio_decode_is_inference.append(latent.is_inference())
        return "decoded-audio"

    monkeypatch.setattr(distilled_streaming, "denoise_audio_video", fake_denoise_audio_video)
    monkeypatch.setattr(distilled_streaming, "upsample_video", fake_upsample_video)
    monkeypatch.setattr(distilled_streaming, "vae_decode_video", fake_decode_video)
    monkeypatch.setattr(distilled_streaming, "vae_decode_audio", fake_decode_audio)

    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )

    result = runner.run_chunk(_base_config())

    assert result.final_chunk_video == "decoded-video"
    assert result.final_chunk_audio == "decoded-audio"
    assert video_decode_is_inference == [False]
    assert audio_decode_is_inference == [False]


def test_offload_generation_components_moves_cached_modules_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModule:
        def __init__(self, name: str) -> None:
            self.name = name
            self.to_calls: list[tuple[torch.device, torch.dtype | None]] = []

        def to(self, *, device: torch.device, dtype: torch.dtype | None = None):
            self.to_calls.append((device, dtype))
            return self

    stage1_transformer = FakeModule("stage1")
    stage2_transformer = FakeModule("stage2")
    video_encoder = FakeModule("video_encoder")
    spatial_upsampler = FakeModule("spatial_upsampler")

    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )
    runner.device = torch.device("cuda")
    runner._stage1_transformer = stage1_transformer
    runner._video_encoder = video_encoder
    runner._spatial_upsampler = spatial_upsampler
    runner._stage2_transformers = {(True, 64, -2.0, ""): stage2_transformer}
    runner._active_stage2_transformer = stage2_transformer
    monkeypatch.setattr(distilled_streaming, "cleanup_memory", lambda: None)
    monkeypatch.setattr(distilled_streaming.torch.cuda, "synchronize", lambda: None, raising=False)

    runner._offload_generation_components()

    assert stage1_transformer.to_calls[-1][0] == torch.device("cpu")
    assert stage2_transformer.to_calls[-1][0] == torch.device("cpu")
    assert video_encoder.to_calls[-1][0] == torch.device("cpu")
    assert spatial_upsampler.to_calls[-1][0] == torch.device("cpu")


def test_prepare_video_decode_components_offloads_audio_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModule:
        def __init__(self, name: str) -> None:
            self.name = name
            self.to_calls: list[tuple[torch.device, torch.dtype | None]] = []

        def to(self, *, device: torch.device, dtype: torch.dtype | None = None):
            self.to_calls.append((device, dtype))
            return self

    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )
    runner.device = torch.device("cuda")
    runner._video_decoder = FakeModule("video_decoder")
    runner._audio_decoder = FakeModule("audio_decoder")
    runner._vocoder = FakeModule("vocoder")
    monkeypatch.setattr(distilled_streaming, "cleanup_memory", lambda: None)
    monkeypatch.setattr(distilled_streaming.torch.cuda, "synchronize", lambda: None, raising=False)

    runner._prepare_video_decode_components()

    assert runner._video_decoder.to_calls[-1][0] == torch.device("cuda")
    assert runner._audio_decoder.to_calls[-1][0] == torch.device("cpu")
    assert runner._vocoder.to_calls[-1][0] == torch.device("cpu")


def test_run_chunk_offloads_generation_before_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_runner_dependencies(monkeypatch)
    offload_events: list[str] = []
    video_decode_prep_events: list[str] = []

    def fake_offload_generation_components() -> None:
        offload_events.append("offloaded")

    def fake_prepare_video_decode_components() -> None:
        video_decode_prep_events.append("prepared")

    def fake_decode_video(*args, **kwargs):
        assert offload_events == ["offloaded"]
        assert video_decode_prep_events == ["prepared"]
        return "decoded-video"

    monkeypatch.setattr(distilled_streaming, "vae_decode_video", fake_decode_video)
    monkeypatch.setattr(distilled_streaming, "vae_decode_audio", lambda *args, **kwargs: "decoded-audio")

    runner = OfficialDistilledChunkRunner(
        distilled_checkpoint_path="/tmp/ltx-2.3-distilled.safetensors",
        gemma_root="/tmp/gemma",
        spatial_upsampler_path="/tmp/ltx-2.3-spatial-upscaler.safetensors",
    )
    monkeypatch.setattr(runner, "_offload_generation_components", fake_offload_generation_components)
    monkeypatch.setattr(runner, "_prepare_video_decode_components", fake_prepare_video_decode_components)

    result = runner.run_chunk(_base_config())

    assert result.final_chunk_video == "decoded-video"
    assert result.final_chunk_audio == "decoded-audio"
