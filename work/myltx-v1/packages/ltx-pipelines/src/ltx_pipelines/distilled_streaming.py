from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger, euler_denoising_loop
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    get_device,
    modality_from_latent_state,
    post_process_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.types import PipelineComponents


_PHASE1_PRESET_TO_GEOMETRY: dict[str, tuple[int, int]] = {
    "small": (1024, 1536),
}


def _cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class OfficialDistilledChunkConfig:
    preset: str
    distilled_checkpoint_path: str
    gemma_root: str
    spatial_upsampler_path: str
    num_frames: int
    frame_rate: float
    prompt: str
    seed: int
    ssm_streaming_enabled: bool = False
    ssm_d_state: int = 64
    ssm_gate_bias: float = -2.0
    ssm_checkpoint_path: str = ""

    def __post_init__(self) -> None:
        if self.preset not in _PHASE1_PRESET_TO_GEOMETRY:
            supported = ", ".join(sorted(_PHASE1_PRESET_TO_GEOMETRY))
            raise ValueError(f"Unsupported preset: {self.preset}. Expected one of: {supported}")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError("OfficialDistilledChunkConfig requires num_frames to satisfy 8*K+1.")
        assert_resolution(height=self.height, width=self.width, is_two_stage=True)

    @property
    def height(self) -> int:
        return _PHASE1_PRESET_TO_GEOMETRY[self.preset][0]

    @property
    def width(self) -> int:
        return _PHASE1_PRESET_TO_GEOMETRY[self.preset][1]


@dataclass
class OfficialDistilledChunkResult:
    final_chunk_video: Any
    final_chunk_audio: Audio | Any | None
    next_ssm_state: object | None
    evictable_video_tokens: torch.Tensor
    evictable_audio_tokens: torch.Tensor


def _extract_ssm_state_dict(checkpoint_payload: object) -> dict[str, torch.Tensor]:
    payload = checkpoint_payload
    if isinstance(payload, dict):
        for candidate_key in ("state_dict", "model_state_dict", "model"):
            nested = payload.get(candidate_key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict):
        raise ValueError("SSM checkpoint must resolve to a state_dict dictionary")

    ssm_layers_state_dict: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("ssm_layers."):
            ssm_layers_state_dict[key[len("ssm_layers."):]] = value
        elif ".ssm_layers." in key:
            ssm_layers_state_dict[key.split(".ssm_layers.", 1)[1]] = value

    if not ssm_layers_state_dict:
        tensor_only = {key: value for key, value in payload.items() if torch.is_tensor(value)}
        if any(key and key[0].isdigit() for key in tensor_only):
            ssm_layers_state_dict = tensor_only

    if not ssm_layers_state_dict:
        raise ValueError("No ssm_layers weights found in checkpoint")
    return ssm_layers_state_dict


def _run_stage2_chunk_with_ssm(  # noqa: PLR0913
    *,
    output_shape: VideoPixelShape,
    conditionings: list[object],
    noiser: GaussianNoiser,
    sigmas: torch.Tensor,
    stepper: EulerDiffusionStep,
    transformer: X0Model,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float,
    initial_video_latent: torch.Tensor,
    initial_audio_latent: torch.Tensor,
    ssm_state: object | None,
) -> tuple[LatentState, LatentState, object | None]:
    current_ssm_state = ssm_state

    def denoising_loop(
        loop_sigmas: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        loop_stepper: EulerDiffusionStep,
    ) -> tuple[LatentState, LatentState]:
        nonlocal current_ssm_state

        def denoise_fn(
            video_state: LatentState,
            audio_state: LatentState,
            denoise_sigmas: torch.Tensor,
            step_index: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal current_ssm_state
            sigma = denoise_sigmas[step_index]
            denoised_video, denoised_audio, current_ssm_state = transformer(
                video=modality_from_latent_state(video_state, video_context, sigma),
                audio=modality_from_latent_state(audio_state, audio_context, sigma),
                perturbations=None,
                ssm_state=current_ssm_state,
                return_ssm_state=True,
            )
            return denoised_video, denoised_audio

        return euler_denoising_loop(
            sigmas=loop_sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=loop_stepper,
            denoise_fn=denoise_fn,
        )

    video_state, audio_state = denoise_audio_video(
        output_shape=output_shape,
        conditionings=conditionings,
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=denoising_loop,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_video_latent=initial_video_latent,
        initial_audio_latent=initial_audio_latent,
    )
    return video_state, audio_state, current_ssm_state


class OfficialDistilledChunkRunner:
    def __init__(
        self,
        *,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.distilled_checkpoint_path = distilled_checkpoint_path
        self.gemma_root = gemma_root
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras
        self.quantization = quantization

        self.model_ledger: ModelLedger | None = None
        self.pipeline_components: PipelineComponents | None = None
        self._stage1_transformer: object | None = None
        self._video_encoder: object | None = None
        self._spatial_upsampler: object | None = None
        self._video_decoder: object | None = None
        self._audio_decoder: object | None = None
        self._vocoder: object | None = None
        self._stage2_transformers: dict[tuple[bool, int, float, str], object] = {}
        self._active_stage2_transformer: object | None = None

    def _ensure_components(self) -> None:
        if self.model_ledger is None:
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=self.distilled_checkpoint_path,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root_path=self.gemma_root,
                loras=self.loras,
                quantization=self.quantization,
            )
        if self.pipeline_components is None:
            self.pipeline_components = PipelineComponents(dtype=self.dtype, device=self.device)
        if self._stage1_transformer is None:
            self._stage1_transformer = self.model_ledger.transformer()
        if self._video_encoder is None:
            self._video_encoder = self.model_ledger.video_encoder()
        if self._spatial_upsampler is None:
            self._spatial_upsampler = self.model_ledger.spatial_upsampler()
        if self._video_decoder is None:
            self._video_decoder = self.model_ledger.video_decoder()
        if self._audio_decoder is None:
            self._audio_decoder = self.model_ledger.audio_decoder()
        if self._vocoder is None:
            self._vocoder = self.model_ledger.vocoder()

    def _move_module(
        self,
        module: object | None,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> object | None:
        if module is None or not hasattr(module, "to"):
            return module
        kwargs: dict[str, object] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return module.to(**kwargs)

    def _activate_generation_components(self) -> None:
        self._ensure_components()
        self._stage1_transformer = self._move_module(self._stage1_transformer, device=self.device, dtype=self.dtype)
        self._video_encoder = self._move_module(self._video_encoder, device=self.device, dtype=self.dtype)
        self._spatial_upsampler = self._move_module(self._spatial_upsampler, device=self.device, dtype=self.dtype)

    def _offload_generation_components(self) -> None:
        if self.device.type != "cuda":
            return
        cpu = torch.device("cpu")
        self._stage1_transformer = self._move_module(self._stage1_transformer, device=cpu, dtype=self.dtype)
        self._video_encoder = self._move_module(self._video_encoder, device=cpu, dtype=self.dtype)
        self._spatial_upsampler = self._move_module(self._spatial_upsampler, device=cpu, dtype=self.dtype)
        for key, transformer in list(self._stage2_transformers.items()):
            self._stage2_transformers[key] = self._move_module(transformer, device=cpu, dtype=self.dtype)
        self._active_stage2_transformer = self._move_module(self._active_stage2_transformer, device=cpu, dtype=self.dtype)
        _cuda_synchronize()
        cleanup_memory()

    def _prepare_video_decode_components(self) -> None:
        if self.device.type != "cuda":
            return
        cpu = torch.device("cpu")
        self._audio_decoder = self._move_module(self._audio_decoder, device=cpu, dtype=self.dtype)
        self._vocoder = self._move_module(self._vocoder, device=cpu, dtype=self.dtype)
        self._video_decoder = self._move_module(self._video_decoder, device=self.device, dtype=self.dtype)
        _cuda_synchronize()
        cleanup_memory()

    def _prepare_audio_decode_components(self) -> None:
        if self.device.type != "cuda":
            return
        self._audio_decoder = self._move_module(self._audio_decoder, device=self.device, dtype=self.dtype)
        self._vocoder = self._move_module(self._vocoder, device=self.device, dtype=self.dtype)
        _cuda_synchronize()
        cleanup_memory()

    def _validate_asset_paths(self, config: OfficialDistilledChunkConfig) -> None:
        mismatches = []
        if config.distilled_checkpoint_path != self.distilled_checkpoint_path:
            mismatches.append("distilled_checkpoint_path")
        if config.gemma_root != self.gemma_root:
            mismatches.append("gemma_root")
        if config.spatial_upsampler_path != self.spatial_upsampler_path:
            mismatches.append("spatial_upsampler_path")
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(f"Chunk config asset paths do not match runner initialization for: {joined}")

    def _stage1_output_shape(self, config: OfficialDistilledChunkConfig) -> VideoPixelShape:
        return VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            width=config.width // 2,
            height=config.height // 2,
            fps=config.frame_rate,
        )

    def _stage2_output_shape(self, config: OfficialDistilledChunkConfig) -> VideoPixelShape:
        return VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            width=config.width,
            height=config.height,
            fps=config.frame_rate,
        )

    def _get_stage2_transformer(self, config: OfficialDistilledChunkConfig) -> object:
        self._ensure_components()
        cache_key = (
            config.ssm_streaming_enabled,
            config.ssm_d_state,
            float(config.ssm_gate_bias),
            config.ssm_checkpoint_path,
        )
        if cache_key in self._stage2_transformers:
            transformer = self._move_module(self._stage2_transformers[cache_key], device=self.device, dtype=self.dtype)
            self._stage2_transformers[cache_key] = transformer
            self._active_stage2_transformer = transformer
            return transformer
        if not config.ssm_streaming_enabled:
            transformer = self._stage1_transformer
        elif not isinstance(self._stage1_transformer, X0Model):
            transformer = self._stage1_transformer
        else:
            from ltx_core.model.transformer.ssm_integration import SSMAugmentedLTXModel
            from ltx_core.model.transformer.ssm_memory import SSMConfig

            ssm_config = SSMConfig(enabled=True, d_state=config.ssm_d_state, gate_bias=config.ssm_gate_bias)
            base_model = self._stage1_transformer.velocity_model
            transformer = X0Model(SSMAugmentedLTXModel.from_base(base_model, ssm_config))
            if config.ssm_checkpoint_path:
                checkpoint_payload = torch.load(Path(config.ssm_checkpoint_path).resolve(), map_location="cpu")
                ssm_layers_state_dict = _extract_ssm_state_dict(checkpoint_payload)
                transformer.velocity_model.ssm_layers.load_state_dict(ssm_layers_state_dict, strict=False)
            transformer = transformer.to(device=self.device, dtype=self.dtype)
        self._stage2_transformers[cache_key] = transformer
        self._active_stage2_transformer = transformer
        return transformer

    def _patchify_evictable_tokens(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_components()
        return (
            self.pipeline_components.video_patchifier.patchify(video_latent),
            self.pipeline_components.audio_patchifier.patchify(audio_latent),
        )

    @torch.no_grad()
    def run_chunk(
        self,
        config: OfficialDistilledChunkConfig,
        ssm_state: object | None = None,
    ) -> OfficialDistilledChunkResult:
        self._ensure_components()
        self._activate_generation_components()
        self._validate_asset_paths(config)
        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        (ctx_p,) = encode_prompts([config.prompt], self.model_ledger)
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        stage_1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=self.device)
        stage_1_output_shape = self._stage1_output_shape(config)
        stage_1_conditionings = combined_image_conditionings(
            images=[],
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=self._video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        stage_1_video_state, stage_1_audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=lambda sigmas, video_state, audio_state, loop_stepper: euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=loop_stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=self._stage1_transformer,
                ),
            ),
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
        )

        upscaled_video_latent = upsample_video(
            latent=stage_1_video_state.latent[:1],
            video_encoder=self._video_encoder,
            upsampler=self._spatial_upsampler,
        )

        _cuda_synchronize()
        cleanup_memory()

        stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)
        stage_2_output_shape = self._stage2_output_shape(config)
        stage_2_conditionings = combined_image_conditionings(
            images=[],
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=self._video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        stage_2_transformer = self._get_stage2_transformer(config)
        if config.ssm_streaming_enabled:
            final_video_state, final_audio_state, next_ssm_state = _run_stage2_chunk_with_ssm(
                output_shape=stage_2_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                transformer=stage_2_transformer,
                video_context=video_context,
                audio_context=audio_context,
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
                noise_scale=float(stage_2_sigmas[0]),
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=stage_1_audio_state.latent,
                ssm_state=ssm_state,
            )
        else:
            final_video_state, final_audio_state = denoise_audio_video(
                output_shape=stage_2_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=lambda sigmas, video_state, audio_state, loop_stepper: euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=loop_stepper,
                    denoise_fn=simple_denoising_func(
                        video_context=video_context,
                        audio_context=audio_context,
                        transformer=stage_2_transformer,
                    ),
                ),
                components=self.pipeline_components,
                dtype=self.dtype,
                device=self.device,
                noise_scale=float(stage_2_sigmas[0]),
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=stage_1_audio_state.latent,
            )
            next_ssm_state = ssm_state

        evictable_video_tokens, evictable_audio_tokens = self._patchify_evictable_tokens(
            final_video_state.latent,
            final_audio_state.latent,
        )

        del stage_1_video_state
        del stage_1_audio_state
        del upscaled_video_latent
        del stage_1_conditionings
        del stage_2_conditionings
        del stage_1_sigmas
        del stage_2_sigmas
        del video_context
        del audio_context
        del noiser
        del stepper
        del stage_2_transformer
        self._offload_generation_components()
        _cuda_synchronize()
        cleanup_memory()

        self._prepare_video_decode_components()
        decoded_video = vae_decode_video(
            final_video_state.latent,
            self._video_decoder,
            TilingConfig.default(),
            generator,
        )
        self._prepare_audio_decode_components()
        decoded_audio = vae_decode_audio(final_audio_state.latent, self._audio_decoder, self._vocoder)

        return OfficialDistilledChunkResult(
            final_chunk_video=decoded_video,
            final_chunk_audio=decoded_audio,
            next_ssm_state=next_ssm_state,
            evictable_video_tokens=evictable_video_tokens,
            evictable_audio_tokens=evictable_audio_tokens,
        )

    def compress_evicted_tokens(
        self,
        ssm_state: object,
        evicted_video: torch.Tensor | None,
        evicted_audio: torch.Tensor | None,
    ) -> object:
        self._ensure_components()
        transformer = self._active_stage2_transformer or self._stage1_transformer
        compressor = getattr(transformer, "compress_evicted_tokens", None)
        if compressor is None:
            compressor = getattr(getattr(transformer, "velocity_model", None), "compress_evicted_tokens", None)
        if compressor is None:
            raise RuntimeError("Stage-2 transformer does not expose compress_evicted_tokens")
        normalized_video = evicted_video
        normalized_audio = evicted_audio
        if normalized_video is not None and hasattr(normalized_video, "to"):
            normalized_video = normalized_video.to(device=self.device, dtype=self.dtype)
        if normalized_audio is not None and hasattr(normalized_audio, "to"):
            normalized_audio = normalized_audio.to(device=self.device, dtype=self.dtype)
        return compressor(ssm_state, normalized_video, normalized_audio)
