from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-core/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-trainer/src"))
sys.modules.setdefault("safetensors", types.ModuleType("safetensors"))

from ltx_trainer.validation_sampler import ValidationStreamState, _update_delayed_stream_state  # noqa: E402


class FakeSSMState:
    def __init__(self) -> None:
        self.compressed: list[tuple[torch.Tensor, torch.Tensor | None]] = []


def test_update_delayed_stream_state_waits_until_window_is_exceeded() -> None:
    stream_state = ValidationStreamState(ssm_state=FakeSSMState())

    def fake_compress(ssm_state: FakeSSMState, evicted_video: torch.Tensor, evicted_audio: torch.Tensor | None) -> FakeSSMState:
        ssm_state.compressed.append((evicted_video.clone(), None if evicted_audio is None else evicted_audio.clone()))
        return ssm_state

    first_video = torch.full((1, 2, 3), 1.0)
    second_video = torch.full((1, 2, 3), 2.0)

    _update_delayed_stream_state(
        stream_state=stream_state,
        next_ssm_state=stream_state.ssm_state,
        chunk_video_tokens=first_video,
        chunk_audio_tokens=None,
        window_blocks=2,
        disable_compression=False,
        compress_evicted_tokens=fake_compress,
    )
    _update_delayed_stream_state(
        stream_state=stream_state,
        next_ssm_state=stream_state.ssm_state,
        chunk_video_tokens=second_video,
        chunk_audio_tokens=None,
        window_blocks=2,
        disable_compression=False,
        compress_evicted_tokens=fake_compress,
    )

    first_video.fill_(9.0)

    assert stream_state.ssm_state is not None
    assert stream_state.ssm_state.compressed == []
    assert len(stream_state.pending_video_chunks) == 2
    assert torch.all(stream_state.pending_video_chunks[0] == 1.0)
    assert torch.all(stream_state.pending_video_chunks[1] == 2.0)


def test_update_delayed_stream_state_compresses_the_evicted_chunk() -> None:
    stream_state = ValidationStreamState(ssm_state=FakeSSMState())

    def fake_compress(ssm_state: FakeSSMState, evicted_video: torch.Tensor, evicted_audio: torch.Tensor | None) -> FakeSSMState:
        ssm_state.compressed.append((evicted_video.clone(), None if evicted_audio is None else evicted_audio.clone()))
        return ssm_state

    for fill_value in (1.0, 2.0, 3.0):
        _update_delayed_stream_state(
            stream_state=stream_state,
            next_ssm_state=stream_state.ssm_state,
            chunk_video_tokens=torch.full((1, 2, 3), fill_value),
            chunk_audio_tokens=torch.full((1, 2, 3), fill_value + 10.0),
            window_blocks=2,
            disable_compression=False,
            compress_evicted_tokens=fake_compress,
        )

    assert stream_state.ssm_state is not None
    assert len(stream_state.ssm_state.compressed) == 1
    evicted_video, evicted_audio = stream_state.ssm_state.compressed[0]
    assert torch.all(evicted_video == 1.0)
    assert evicted_audio is not None
    assert torch.all(evicted_audio == 11.0)
    assert len(stream_state.pending_video_chunks) == 2
    assert len(stream_state.pending_audio_chunks) == 2
    assert torch.all(stream_state.pending_video_chunks[0] == 2.0)
    assert torch.all(stream_state.pending_video_chunks[1] == 3.0)
