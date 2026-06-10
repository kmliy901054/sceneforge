"""ForgePipeline: depth-ControlNet SDXL with the Lightning ladder (ARCHITECTURE.md §7).

Implements:
  - §7.1 pipeline assembly (SDXL-base fp16 + controlnet-depth fp16 + vae-fp16-fix,
    Lightning 4-step full-UNet swap with Euler *trailing* for L0/L1; fused 4-step
    LoRA for L2; plain base SDXL for L3). NO attention slicing / VAE tiling by
    default — re-enabled only at degrade level V2 (§10.2).
  - §7.2 generation call defaults (per-level params from the §7.4 ladder).
  - §7.5 OOM recovery, exact sequence: pipe.to("cpu") -> gc.collect() ->
    torch.cuda.empty_cache() -> enable_model_cpu_offload() -> retry ONCE ->
    still OOM -> drop to 640 px for the remainder of the burst. Side effect:
    ``force_sequential`` is latched True for the session (cfg.vram.mode).
  - peak_vram() reporting ALL THREE numbers: max_memory_allocated,
    max_memory_reserved, and min ``mem_get_info`` free seen during generation
    (device-wide; the only number that sees the CUDA context and Ollama, §10.1).
"""
from __future__ import annotations

import gc
import logging
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_GB = float(1 << 30)

#: §7.4 conditioning ladder. L4 (canny MultiControlNet) is a download-gated
#: contingency and intentionally NOT implemented here.
LEVEL_PARAMS: dict[int, dict[str, Any]] = {
    0: dict(steps=4, guidance_scale=0.0, cond_scale=0.85, control_guidance_end=0.9),
    1: dict(steps=4, guidance_scale=0.0, cond_scale=1.0, control_guidance_end=1.0),
    2: dict(steps=8, guidance_scale=1.5, cond_scale=0.9, control_guidance_end=1.0),
    3: dict(steps=20, guidance_scale=7.5, cond_scale=0.8, control_guidance_end=1.0),
}

def aspect_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale (width, height) so the long side is ``max_side``, rounded to
    multiples of 64 (SDXL latent/UNet alignment), each side ≥ 64.

    e.g. 640×480 @ 768 → (768, 576) — used by the augment restyler to generate
    at the source frame's aspect ratio instead of distorting to a square.
    """
    scale = max_side / max(width, height)
    return (
        max(64, int(round(width * scale / 64.0)) * 64),
        max(64, int(round(height * scale / 64.0)) * 64),
    )


SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_UNET = "sdxl_lightning_4step_unet.safetensors"
LIGHTNING_LORA = "sdxl_lightning_4step_lora.safetensors"


class ForgePipeline:
    """Load/generate/unload wrapper around StableDiffusionXLControlNetPipeline.

    State attributes (read by orchestrator/UI):
        level: currently loaded ladder level (None until load()).
        resolution: current output size; drops to 640 at OOM stage V4 (§10.2).
        offloaded: True once enable_model_cpu_offload() has run (V3).
        force_sequential: latched True after any caught OOM — the session must
            run cfg.vram.mode="sequential" from then on (§7.5 side effects).
    """

    def __init__(self, device: str = "cuda", resolution: int = 768) -> None:
        self.device = device
        self.resolution = resolution
        self.pipe: Any = None
        self.level: int | None = None
        self.offloaded = False
        self.force_sequential = False
        self._min_free_bytes: int | None = None

    # ------------------------------------------------------------------ load
    def load(self, level: int = 0) -> None:
        """Assemble the §7.1 pipeline for ladder level L0–L3.

        L0 <-> L1 share identical weights (Lightning 4-step UNet) and differ only
        in call params, so switching between them is free. Any other transition
        rebuilds the pipeline.
        """
        if level not in LEVEL_PARAMS:
            raise NotImplementedError(
                f"level L{level} not implemented; L4 (canny MultiControlNet) is a "
                "contingency requiring a ~2.5 GB download (§7.4)"
            )
        if self.pipe is not None and self.level is not None:
            if {self.level, level} <= {0, 1}:
                self.level = level  # same Lightning UNet — param-only switch
                return
            self.unload()

        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            EulerDiscreteScheduler,
            StableDiffusionXLControlNetPipeline,
            UNet2DConditionModel,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        logger.info("ForgePipeline.load(level=L%d)", level)
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_DEPTH, torch_dtype=torch.float16, variant="fp16")
        vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE, controlnet=controlnet, vae=vae,
            torch_dtype=torch.float16, variant="fp16")

        if level in (0, 1):
            # SDXL-Lightning 4-step UNet — FULL swap, not LoRA (§7.1).
            unet_cfg = UNet2DConditionModel.load_config(SDXL_BASE, subfolder="unet")
            unet = UNet2DConditionModel.from_config(unet_cfg).to(torch.float16)
            unet.load_state_dict(load_file(
                hf_hub_download(LIGHTNING_REPO, LIGHTNING_UNET)))
            pipe.unet = unet
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing")  # REQUIRED
        elif level == 2:
            # Plan-B: base UNet + fused 4-step LoRA tolerates 6–8 steps (§7.1).
            pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LIGHTNING_LORA)
            pipe.fuse_lora()
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing")
        else:  # level == 3: plain base SDXL, stock EulerDiscrete
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        # §7.1 review fix: NO enable_attention_slicing()/enable_vae_tiling() here.
        self.pipe = pipe
        self.level = level
        self.offloaded = False
        self.reset_peak()

    # -------------------------------------------------------------- generate
    def generate(
        self,
        control: Image.Image,
        prompt: str,
        negative: str = "",
        seed: int = 0,
        cond_scale: float | None = None,
        steps: int | None = None,
        size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """One image. Defaults come from the loaded level's §7.4 params.

        ``size``: optional explicit (width, height) — used by the augment
        restyler for non-square REAL robot frames (both must be multiples of
        8; ``aspect_size`` rounds to 64). None keeps the classic square
        ``resolution`` × ``resolution`` behavior.

        OOM recovery per §7.5 (unit-tested with a mocked OutOfMemoryError in
        tests/test_gpu.py): cpu -> gc -> empty_cache -> cpu_offload -> retry
        ONCE -> still OOM -> 640 px (explicit sizes are rescaled to a 640 max
        side) for the remainder of the burst.
        """
        if self.pipe is None or self.level is None:
            raise RuntimeError("ForgePipeline.generate() before load(level)")
        try:
            return self._call(control, prompt, negative, seed, cond_scale, steps, size)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM — running §7.5 recovery (cpu_offload)")
            self._recover_oom()
            try:
                return self._call(control, prompt, negative, seed, cond_scale, steps, size)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM persists after cpu_offload — dropping to 640 px (V4)")
                self.resolution = 640
                if size is not None:
                    size = aspect_size(size[0], size[1], 640)
                return self._call(control, prompt, negative, seed, cond_scale, steps, size)

    def _recover_oom(self) -> None:
        """Exact §7.5 sequence; latches force_sequential for the session."""
        self.pipe.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.pipe.enable_model_cpu_offload()
        self.offloaded = True
        self.force_sequential = True

    def _call(
        self,
        control: Image.Image,
        prompt: str,
        negative: str,
        seed: int,
        cond_scale: float | None,
        steps: int | None,
        size: tuple[int, int] | None = None,
    ) -> Image.Image:
        p = LEVEL_PARAMS[self.level]  # type: ignore[index]
        width, height = size if size is not None else (self.resolution, self.resolution)
        gen_device = self.device if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(gen_device).manual_seed(seed)
        self._sample_free()
        # NOTE: negative_prompt is a NO-OP when guidance_scale <= 1 (L0/L1):
        # diffusers 0.38 enables CFG iff guidance_scale > 1 (verified, §7.2).
        # If resolution dropped to 640 (V4), diffusers' image processor scales
        # the control internally; the orchestrator should re-render depth at
        # 640 for subsequent frames to honor the never-resize contract (§7.3).
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=control,
            num_inference_steps=steps if steps is not None else p["steps"],
            guidance_scale=p["guidance_scale"],
            controlnet_conditioning_scale=(
                cond_scale if cond_scale is not None else p["cond_scale"]),
            control_guidance_end=p["control_guidance_end"],
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=self._step_callback,
        )
        self._sample_free()
        return result.images[0]

    def _step_callback(self, pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
        self._sample_free()
        return kwargs

    # ----------------------------------------------------------------- vram
    def _sample_free(self) -> None:
        if not torch.cuda.is_available():
            return
        free, _total = torch.cuda.mem_get_info()
        if self._min_free_bytes is None or free < self._min_free_bytes:
            self._min_free_bytes = free

    def reset_peak(self) -> None:
        """Reset torch peak counters and the min-device-free tracker."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._min_free_bytes = None

    def peak_vram(self) -> dict[str, float]:
        """All three §7.5 numbers (GB): max allocated / max reserved / min free.

        min_free_gb is device-wide via torch.cuda.mem_get_info — the only number
        that accounts for the CUDA context, allocator slack and Ollama (§10.1).
        """
        if not torch.cuda.is_available():
            return {"max_allocated_gb": 0.0, "max_reserved_gb": 0.0, "min_free_gb": 0.0}
        free_now, _ = torch.cuda.mem_get_info()
        min_free = self._min_free_bytes if self._min_free_bytes is not None else free_now
        return {
            "max_allocated_gb": torch.cuda.max_memory_allocated() / _GB,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / _GB,
            "min_free_gb": min_free / _GB,
        }

    # --------------------------------------------------------------- unload
    def unload(self) -> None:
        """Drop all refs and return VRAM to the driver (§7.5)."""
        self.pipe = None
        self.level = None
        self.offloaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
