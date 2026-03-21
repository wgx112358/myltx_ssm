"""SSM memory integration for LTX model.

This module wraps the existing LTXModel to add per-layer, per-modality
SSM memory modules.  The wrapper loads an ODE-trained checkpoint unchanged
and adds only the new SSM parameters on top.

Usage:
    base_model = load_ltx_model(checkpoint_path)
    model = SSMAugmentedLTXModel.from_base(base_model, ssm_config)
    # base weights frozen, SSM modules trainable
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.ssm_memory import SSMConfig, SSMMemoryModule, SSMState
from ltx_core.model.transformer.transformer_args import TransformerArgs


class SSMBlockPair(nn.Module):
    """A pair of SSM modules (video + audio) for one transformer layer."""

    def __init__(
        self,
        video_dim: int | None,
        audio_dim: int | None,
        ssm_config: SSMConfig,
    ):
        super().__init__()
        self.has_video = video_dim is not None
        self.has_audio = audio_dim is not None

        if self.has_video:
            self.video_ssm = SSMMemoryModule(
                d_model=video_dim,
                d_state=ssm_config.d_state,
                gate_bias=ssm_config.gate_bias,
            )
        if self.has_audio:
            self.audio_ssm = SSMMemoryModule(
                d_model=audio_dim,
                d_state=ssm_config.d_state,
                gate_bias=ssm_config.gate_bias,
            )


class SSMAugmentedLTXModel(nn.Module):
    """LTXModel wrapped with per-layer SSM memory.

    Architecture:
        For each transformer block, after the original block forward:
        1. Query video SSM with current video hidden states -> add to output
        2. Query audio SSM with current audio hidden states -> add to output
        SSM state compression (eviction) is handled externally by the
        streaming inference / training loop.

    The base LTXModel weights can be frozen while only SSM modules are
    trained, or both can be trained jointly.
    """

    def __init__(self, base_model: LTXModel, ssm_config: SSMConfig):
        super().__init__()
        self.base_model = base_model
        self.ssm_config = ssm_config

        num_layers = len(base_model.transformer_blocks)
        video_dim = getattr(base_model, "inner_dim", None)
        audio_dim = getattr(base_model, "audio_inner_dim", None)

        self.ssm_layers = nn.ModuleList([
            SSMBlockPair(video_dim, audio_dim, ssm_config)
            for _ in range(num_layers)
        ])

        self._num_layers = num_layers

    @classmethod
    def from_base(
        cls,
        base_model: LTXModel,
        ssm_config: SSMConfig | None = None,
    ) -> "SSMAugmentedLTXModel":
        if ssm_config is None:
            ssm_config = SSMConfig(enabled=True)
        return cls(base_model, ssm_config)

    # ------------------------------------------------------------------
    # Delegate properties / methods to base model
    # ------------------------------------------------------------------

    @property
    def model_type(self):
        return self.base_model.model_type

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self.base_model.set_gradient_checkpointing(enable)

    # ------------------------------------------------------------------
    # Forward with SSM
    # ------------------------------------------------------------------

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None = None,
        ssm_state: SSMState | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, SSMState]:
        """Forward pass with SSM memory.

        Args:
            video: Video modality input.
            audio: Audio modality input.
            perturbations: Optional perturbation config.
            ssm_state: SSM state from previous chunk.  ``None`` initialises
                       fresh zero states.

        Returns:
            (video_output, audio_output, updated_ssm_state)
        """
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(
                (video or audio).latent.shape[0]
            )

        if ssm_state is None:
            ssm_state = SSMState.empty()

        # Preprocess (same as base model)
        base = self.base_model
        video_args = (
            base.video_args_preprocessor.prepare(video, audio)
            if video is not None
            else None
        )
        audio_args = (
            base.audio_args_preprocessor.prepare(audio, video)
            if audio is not None
            else None
        )

        # Process transformer blocks with SSM injection
        video_out, audio_out, ssm_state = self._process_blocks_with_ssm(
            video_args, audio_args, perturbations, ssm_state,
        )

        # Output projection (same as base model)
        vx = (
            base._process_output(
                base.scale_shift_table, base.norm_out, base.proj_out,
                video_out.x, video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        ax = (
            base._process_output(
                base.audio_scale_shift_table, base.audio_norm_out,
                base.audio_proj_out, audio_out.x, audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )

        return vx, ax, ssm_state

    def _process_blocks_with_ssm(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
        ssm_state: SSMState,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None, SSMState]:
        base = self.base_model

        for idx, (block, ssm_pair) in enumerate(
            zip(base.transformer_blocks, self.ssm_layers)
        ):
            # --- Original block forward ---
            if base._enable_gradient_checkpointing and self.training:
                video, audio = torch.utils.checkpoint.checkpoint(
                    block, video, audio, perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video, audio=audio, perturbations=perturbations,
                )

            # --- SSM memory query (additive residual after each block) ---
            if video is not None and video.x.numel() > 0 and ssm_pair.has_video:
                v_state = ssm_state.get(idx, "video")
                if v_state is None:
                    v_state = ssm_pair.video_ssm.init_state(
                        video.x.shape[0], video.x.device, video.x.dtype,
                    )
                ssm_out = ssm_pair.video_ssm.query(v_state, video.x)
                video = replace(video, x=video.x + ssm_out)

            if audio is not None and audio.x.numel() > 0 and ssm_pair.has_audio:
                a_state = ssm_state.get(idx, "audio")
                if a_state is None:
                    a_state = ssm_pair.audio_ssm.init_state(
                        audio.x.shape[0], audio.x.device, audio.x.dtype,
                    )
                ssm_out = ssm_pair.audio_ssm.query(a_state, audio.x)
                audio = replace(audio, x=audio.x + ssm_out)

        return video, audio, ssm_state

    def _project_evicted_tokens(
        self,
        evicted_tokens: torch.Tensor | None,
        *,
        modality: str,
    ) -> torch.Tensor | None:
        if evicted_tokens is None or evicted_tokens.numel() == 0:
            return evicted_tokens

        if modality == "video":
            target_dim = getattr(self.base_model, "inner_dim", evicted_tokens.shape[-1])
            if evicted_tokens.shape[-1] == target_dim:
                return evicted_tokens
            preprocessor = self.base_model.video_args_preprocessor
            projector = getattr(preprocessor, "patchify_proj", None)
            if projector is None:
                projector = preprocessor.simple_preprocessor.patchify_proj
            return projector(evicted_tokens)

        if modality == "audio":
            target_dim = getattr(self.base_model, "audio_inner_dim", evicted_tokens.shape[-1])
            if evicted_tokens.shape[-1] == target_dim:
                return evicted_tokens
            preprocessor = self.base_model.audio_args_preprocessor
            projector = getattr(preprocessor, "patchify_proj", None)
            if projector is None:
                projector = preprocessor.simple_preprocessor.patchify_proj
            return projector(evicted_tokens)

        raise ValueError(f"Unsupported modality for eviction projection: {modality}")

    def compress_evicted_tokens(
        self,
        ssm_state: SSMState,
        evicted_video: torch.Tensor | None,
        evicted_audio: torch.Tensor | None,
    ) -> SSMState:
        """Compress evicted tokens into SSM state for all layers.

        Called by the streaming loop when tokens exit the local KV window.

        Args:
            ssm_state: Current SSM state.
            evicted_video: [B, n_evicted, video_dim] or None.
            evicted_audio: [B, n_evicted, audio_dim] or None.

        Returns:
            Updated SSM state.
        """
        # The streaming loop currently evicts patchified latent tokens (128-d for video,
        # 128-d for audio). Compress them in transformer hidden space so the SSM state
        # matches the post-block representation queried during forward.
        projected_video = self._project_evicted_tokens(evicted_video, modality="video")
        projected_audio = self._project_evicted_tokens(evicted_audio, modality="audio")

        for idx, ssm_pair in enumerate(self.ssm_layers):
            if projected_video is not None and ssm_pair.has_video:
                v_state = ssm_state.get(idx, "video")
                if v_state is None:
                    v_state = ssm_pair.video_ssm.init_state(
                        projected_video.shape[0],
                        projected_video.device,
                        projected_video.dtype,
                    )
                ssm_state.set(
                    idx, "video",
                    ssm_pair.video_ssm.compress(v_state, projected_video),
                )

            if projected_audio is not None and ssm_pair.has_audio:
                a_state = ssm_state.get(idx, "audio")
                if a_state is None:
                    a_state = ssm_pair.audio_ssm.init_state(
                        projected_audio.shape[0],
                        projected_audio.device,
                        projected_audio.dtype,
                    )
                ssm_state.set(
                    idx, "audio",
                    ssm_pair.audio_ssm.compress(a_state, projected_audio),
                )

        return ssm_state

    def ssm_param_count(self) -> int:
        return sum(p.numel() for p in self.ssm_layers.parameters())

    def freeze_base(self) -> None:
        """Freeze all base model parameters, keep SSM trainable."""
        for p in self.base_model.parameters():
            p.requires_grad = False

    def unfreeze_base(self) -> None:
        """Unfreeze base model parameters for joint training."""
        for p in self.base_model.parameters():
            p.requires_grad = True
