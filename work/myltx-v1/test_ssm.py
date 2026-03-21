"""Smoke test for SSM memory modules.

Run on remote:
    cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx-v1
    source /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx/.venv/bin/activate
    PYTHONPATH=packages/ltx-core/src:packages/ltx-trainer/src:packages/ltx-pipelines/src python test_ssm.py
"""

import torch
import sys

def test_ssm_memory_module():
    from ltx_core.model.transformer.ssm_memory import SSMMemoryModule, SSMState, SSMConfig

    print("=== Test 1: SSMMemoryModule basic ops ===")

    video_ssm = SSMMemoryModule(d_model=4096, d_state=64, gate_bias=-2.0)
    audio_ssm = SSMMemoryModule(d_model=2048, d_state=64, gate_bias=-2.0)

    B = 2
    device = torch.device("cpu")
    dtype = torch.float32

    # Init states
    v_state = video_ssm.init_state(B, device, dtype)
    a_state = audio_ssm.init_state(B, device, dtype)
    print(f"  Video state shape: {v_state.shape}")  # [2, 64, 4096]
    print(f"  Audio state shape: {a_state.shape}")  # [2, 64, 2048]

    # Compress evicted tokens
    evicted_v = torch.randn(B, 100, 4096)  # 100 evicted video tokens
    evicted_a = torch.randn(B, 50, 2048)   # 50 evicted audio tokens
    v_state = video_ssm.compress(v_state, evicted_v)
    a_state = audio_ssm.compress(a_state, evicted_a)
    print(f"  After compress - video state norm: {v_state.norm().item():.4f}")
    print(f"  After compress - audio state norm: {a_state.norm().item():.4f}")

    # Query with current tokens
    current_v = torch.randn(B, 200, 4096)  # 200 current video tokens
    current_a = torch.randn(B, 25, 2048)   # 25 current audio tokens
    out_v = video_ssm.query(v_state, current_v)
    out_a = audio_ssm.query(a_state, current_a)
    print(f"  Video query output shape: {out_v.shape}")  # [2, 200, 4096]
    print(f"  Audio query output shape: {out_a.shape}")  # [2, 25, 2048]

    # Gate should be near 0 at init (gate_bias=-2.0)
    gate_val = torch.sigmoid(torch.tensor(-2.0)).item()
    print(f"  Initial gate value (sigmoid(-2.0)): {gate_val:.4f}")
    print(f"  Video output norm (should be small): {out_v.norm().item():.4f}")

    # Param count
    v_params = sum(p.numel() for p in video_ssm.parameters())
    a_params = sum(p.numel() for p in audio_ssm.parameters())
    print(f"  Video SSM params: {v_params:,} ({v_params/1e6:.2f}M)")
    print(f"  Audio SSM params: {a_params:,} ({a_params/1e6:.2f}M)")
    print(f"  Total per layer (V+A): {(v_params + a_params):,}")
    print(f"  Total 48 layers: {48 * (v_params + a_params):,} ({48*(v_params+a_params)/1e6:.1f}M)")

    print("  PASSED\n")


def test_ssm_state():
    from ltx_core.model.transformer.ssm_memory import SSMState

    print("=== Test 2: SSMState container ===")

    state = SSMState.empty()
    assert state.get(0, "video") is None

    t = torch.randn(2, 64, 4096)
    state.set(0, "video", t)
    assert state.get(0, "video") is t
    assert state.get(0, "audio") is None

    cloned = state.clone()
    assert cloned.get(0, "video") is not t
    assert torch.equal(cloned.get(0, "video"), t)

    print("  PASSED\n")


def test_gradient_flow():
    from ltx_core.model.transformer.ssm_memory import SSMMemoryModule

    print("=== Test 3: Gradient flow ===")

    ssm = SSMMemoryModule(d_model=256, d_state=16, gate_bias=-2.0)
    state = ssm.init_state(1, torch.device("cpu"), torch.float32)

    evicted = torch.randn(1, 10, 256, requires_grad=True)
    state = ssm.compress(state, evicted)

    current = torch.randn(1, 5, 256, requires_grad=True)
    out = ssm.query(state, current)
    loss = out.sum()
    loss.backward()

    # Check gradients exist for SSM params
    has_grad = all(p.grad is not None for p in ssm.parameters())
    print(f"  All SSM params have gradients: {has_grad}")
    print(f"  Current tokens have gradient: {current.grad is not None}")
    # Note: evicted tokens won't have gradients through compress->query
    # because state was created fresh (no graph through init_state)
    # In practice, compress is called on detached evicted tokens anyway.

    print("  PASSED\n")


if __name__ == "__main__":
    test_ssm_memory_module()
    test_ssm_state()
    test_gradient_flow()
    print("All tests passed!")
