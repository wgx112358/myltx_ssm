from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-core/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-trainer/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-pipelines/src"))

from self_forcing_data import load_switch_episodes  # noqa: E402
from train_self_forcing import (  # noqa: E402
    SelfForcingConfig,
    build_chunk_condition_schedule,
    chunked_forward_with_ssm,
)


def _write_manifest(path: Path) -> None:
    episodes = [{
        "episode_id": "episode_0000",
        "segments": [
            {"category": "A", "prompt": "prompt-a0", "start_seconds": 0.0, "duration_seconds": 2.0},
            {"category": "B", "prompt": "prompt-a1", "start_seconds": 2.0, "duration_seconds": 1.0},
            {"category": "C", "prompt": "prompt-a2", "start_seconds": 3.0, "duration_seconds": 1.0},
        ],
    }]
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode) + "\n")


def _condition_payload(fill_value: float) -> dict[str, torch.Tensor]:
    return {
        "video_prompt_embeds": torch.full((2, 4096), fill_value, dtype=torch.float32),
        "audio_prompt_embeds": torch.full((2, 2048), fill_value, dtype=torch.float32),
        "prompt_attention_mask": torch.ones(2, dtype=torch.int64),
    }


def test_build_chunk_condition_schedule_uses_manifest_prompt_cache(tmp_path: Path) -> None:
    manifest_path = tmp_path / "switch.jsonl"
    _write_manifest(manifest_path)
    episodes = load_switch_episodes(manifest_path)
    config = SelfForcingConfig()

    video_contexts, audio_contexts, context_masks, metadata = build_chunk_condition_schedule(
        sample_id="00000__step_000",
        num_chunks=4,
        base_conditions=_condition_payload(0.0),
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        switch_episodes=episodes,
        prompt_cache={
            "prompt-a0": _condition_payload(10.0),
            "prompt-a1": _condition_payload(20.0),
            "prompt-a2": _condition_payload(30.0),
        },
    )

    assert metadata["prompt_switch_flags"] == [False, False, True, True]
    assert metadata["segment_indices"] == [0, 0, 1, 2]
    assert [float(context[0, 0, 0].item()) for context in video_contexts] == [10.0, 10.0, 20.0, 30.0]
    assert [float(context[0, 0, 0].item()) for context in audio_contexts] == [10.0, 10.0, 20.0, 30.0]
    assert all(mask.shape == (1, 2) for mask in context_masks)


class _FakeModel:
    def __call__(self, *, video, audio, ssm_state):
        return torch.zeros_like(video.latent), torch.zeros_like(audio.latent), ssm_state

    def compress_evicted_tokens(self, ssm_state, evicted_v, evicted_a):
        return ssm_state


def test_chunked_forward_with_ssm_applies_switch_weights() -> None:
    config = SelfForcingConfig(keep_loss_weight=1.0, edit_loss_weight=2.0, switch_loss_weight=3.0, window_blocks=8)
    sigma = torch.tensor([0.5], dtype=torch.float32)

    loss, metrics = chunked_forward_with_ssm(
        model=_FakeModel(),
        video_chunks=[torch.ones(1, 1, 1), torch.ones(1, 1, 1)],
        audio_chunks=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        video_target_chunks=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        audio_target_chunks=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        video_sigma=sigma,
        audio_sigma=sigma,
        video_positions_chunks=[torch.zeros(1, 3, 1, 2), torch.zeros(1, 3, 1, 2)],
        audio_positions_chunks=[torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)],
        video_contexts=[torch.zeros(1, 1, 4096), torch.zeros(1, 1, 4096)],
        audio_contexts=[torch.zeros(1, 1, 2048), torch.zeros(1, 1, 2048)],
        context_masks=[torch.ones(1, 1, dtype=torch.int64), torch.ones(1, 1, dtype=torch.int64)],
        prompt_switch_flags=[False, True],
        config=config,
        step=0,
    )

    assert loss.item() == 14.0
    assert metrics["video_loss"] == 4.0
    assert metrics["audio_loss"] == 0.0
    assert metrics["keep_loss"] == 4.0
    assert metrics["edit_loss"] == 4.0
    assert metrics["num_switch_chunks"] == 1.0
    assert metrics["switch_state_decay_chunks"] == 0.0



class _StateDecayModel:
    def __init__(self) -> None:
        self.incoming_states: list[torch.Tensor | None] = []

    def __call__(self, *, video, audio, ssm_state):
        incoming = ssm_state.get(0, "video")
        self.incoming_states.append(incoming.clone() if incoming is not None else None)
        if incoming is None:
            ssm_state.set(0, "video", torch.full((1, 1, 1), 8.0))
        return torch.zeros_like(video.latent), torch.zeros_like(audio.latent), ssm_state

    def compress_evicted_tokens(self, ssm_state, evicted_v, evicted_a):
        return ssm_state


def test_chunked_forward_with_ssm_applies_switch_state_decay() -> None:
    model = _StateDecayModel()
    config = SelfForcingConfig(ssm_switch_state_decay=0.25, window_blocks=8)
    sigma = torch.tensor([0.5], dtype=torch.float32)

    _, metrics = chunked_forward_with_ssm(
        model=model,
        video_chunks=[torch.ones(1, 1, 1), torch.ones(1, 1, 1)],
        audio_chunks=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        video_target_chunks=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        audio_target_chunks=[torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
        video_sigma=sigma,
        audio_sigma=sigma,
        video_positions_chunks=[torch.zeros(1, 3, 1, 2), torch.zeros(1, 3, 1, 2)],
        audio_positions_chunks=[torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)],
        video_contexts=[torch.zeros(1, 1, 4096), torch.zeros(1, 1, 4096)],
        audio_contexts=[torch.zeros(1, 1, 2048), torch.zeros(1, 1, 2048)],
        context_masks=[torch.ones(1, 1, dtype=torch.int64), torch.ones(1, 1, dtype=torch.int64)],
        prompt_switch_flags=[False, True],
        config=config,
        step=0,
    )

    assert model.incoming_states[0] is None
    assert model.incoming_states[1] is not None
    assert torch.all(model.incoming_states[1] == 2.0)
    assert metrics["switch_state_decay_chunks"] == 1.0
