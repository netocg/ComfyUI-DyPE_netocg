import copy
import torch
import torch.nn.functional as F
import math
import comfy.samplers
import comfy.sample
import latent_preview
from .models import apply_dype_to_flux
from .sigmas import (
    compute_empirical_mu,
    generalized_time_snr_shift,
    lowstep_sigma,
    bong_tangent_scheduler
)


class FlowMatchingShiftUpscalerSampler:
    """
    A studio-grade monolithic meta-sampler that automates progressive latent-space
    iterative upscaling for Flux models. It handles latent resizing, dynamic resolution-aware
    metadata conditioning (add_dict), and dynamic DyPE model patching natively in a single loop.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": ("INT",
                               {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "latent_samples": ("LATENT",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "Classifier-Free Guidance scale."}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05,
                                             "tooltip": "Target total upscale multiplier (e.g., 2.0x)."}),
                "total_passes": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                                         "tooltip": "Number of progressive upscale steps."}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 40, "step": 1,
                                       "tooltip": "Starting step offset."}),
                "steps_per_pass": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                                           "tooltip": "Number of low-denoise sampling steps per pass."}),
                "step_mode": (["geometric", "simple"],
                              {"default": "simple", "tooltip": "Growth curve for upscale passes."}),
                "scheduler": (["flux", "lowstep", "bong_tangent"],
                              {"default": "bong_tangent", "tooltip": ""}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "The amount of noise to remove."}),
                "vae_compression": ("INT", {"default": 16, "min": 8, "max": 16, "step": 8,
                                            "tooltip": "VAE spatial compression factor (16 for Flux)."}),
                "dype_method": (["yarn", "ntk", "base"],
                                {"default": "ntk", "tooltip": "DyPE extrapolation method for RoPE."}),
                "enable_dype": ("BOOLEAN", {"default": True,
                                            "tooltip": "Enable dynamic position extrapolation for high-resolution stability."}),
                "dype_exponent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.1,
                                            "tooltip": "Controls DyPE scaling intensity over resolution."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute_progressive_upscale"
    CATEGORY = "Custom/MetaSampler"

    def execute_progressive_upscale(
            self, noise_seed, model, positive, negative, sampler, latent_samples, upscale_factor, total_passes,
            start_step, steps_per_pass, step_mode="geometric", vae_compression=16, dype_method="ntk", enable_dype=True,
            dype_exponent=2.0, scheduler="bong_tangent", cfg=1.0, denoise=1.0, unique_id=None
    ):
        current_latents = latent_samples['samples']
        b, c, orig_h, orig_w = current_latents.shape

        # Calculate base pixel dimensions from initial latent tensor
        base_h, base_w = orig_h * vae_compression, orig_w * vae_compression

        # Calculate scale increments per pass
        if step_mode == "geometric":
            scale_multiplier = pow(upscale_factor, 1.0 / total_passes)
        else:
            scale_multiplier = max(0, (upscale_factor - 1.0) / total_passes)

        current_scale = 1.0
        working_latent_dict = copy.deepcopy(latent_samples)

        for p in range(total_passes):
            # Compute cumulative scale for this pass
            if step_mode == "geometric":
                current_scale *= scale_multiplier
            else:
                current_scale += scale_multiplier

            # Ensure final pass precisely matches target scale factor
            if p == total_passes - 1:
                current_scale = upscale_factor

            # 1. Calculate latent dimensions directly in latent space, forced to even numbers
            target_latent_h = int((orig_h * current_scale) // 2 * 2)
            target_latent_w = int((orig_w * current_scale) // 2 * 2)

            # 2. Derive pixel dimensions cleanly from the guaranteed even latent dimensions
            target_h_px = target_latent_h * vae_compression
            target_w_px = target_latent_w * vae_compression

            # 3. Resize the latent tensor using the explicit even latent size
            current_latents = F.interpolate(
                working_latent_dict['samples'],
                size=(target_latent_h, target_latent_w),
                mode='bicubic',
                align_corners=False
            ).contiguous()

            working_latent_dict['samples'] = current_latents

            # Handle noise_mask and noise generation
            noise_mask = working_latent_dict.get("noise_mask", None)
            if noise_mask is not None:
                # Ensure mask is shaped (B, H, W) or (B, 1, H, W) for interpolation
                if noise_mask.dim() == 3:
                    noise_mask = noise_mask.unsqueeze(1)
                noise_mask = F.interpolate(
                    noise_mask.float(),
                    size=(target_latent_h, target_latent_w),
                    mode='bicubic',
                    align_corners=False
                ).squeeze(1).clamp(0.0, 1.0)
            generator = torch.manual_seed(noise_seed)
            samples_tensor = working_latent_dict['samples']
            if getattr(samples_tensor, "is_nested", False):
                # Handles nested/audio-video or complex tensors safely if present
                noises = [torch.randn_like(t, generator=generator) for t in samples_tensor.unbind()]
                noise = comfy.nested_tensor.NestedTensor(noises) if hasattr(comfy,
                                                                            "nested_tensor") else torch.randn_like(
                    samples_tensor)
            else:
                noise = torch.randn(
                    samples_tensor.shape,
                    dtype=samples_tensor.dtype,
                    layout=samples_tensor.layout,
                    device=samples_tensor.device,
                    generator=generator
                )

            # 2. Apply Dynamic DyPE Model Patching for this specific pass resolution
            patched_model = apply_dype_to_flux(
                model=model,
                width=target_w_px,
                height=target_h_px,
                method=dype_method,
                enable_dype=enable_dype,
                dype_exponent=dype_exponent
            )

            # 3. Update Resolution-Aware Metadata Conditioning for Flux (`add_dict`)
            updated_positive = self.update_conditioning_resolution(positive, target_w_px, target_h_px)
            updated_negative = self.update_conditioning_resolution(negative, target_w_px, target_h_px)

            # 4. Configure Guider
            cfg_guider = comfy.samplers.CFGGuider(patched_model)
            cfg_guider.set_conds(updated_positive, updated_negative)
            cfg_guider.set_cfg(cfg)

            # 5. Get execution sigmas via class method self.get_sigmas(...)
            sigmas = self.get_sigmas(patched_model, scheduler, start_step, steps_per_pass, 1.0, denoise, target_w_px,
                                     target_h_px)

            # 6. Run standard ComfyUI sampling over the tensor slice
            callback = latent_preview.prepare_callback(patched_model, steps_per_pass)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            # Pass working_latent_dict['samples'] directly as the latent_image tensor
            sampled_tensor = cfg_guider.sample(
                noise=noise,
                latent_image=working_latent_dict['samples'],
                sampler=sampler,
                sigmas=sigmas,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise_seed
            )

            # Store it back into the working latent dictionary for the next pass
            working_latent_dict['samples'] = sampled_tensor

        return (working_latent_dict,)

    def update_conditioning_resolution(self, conditioning, width, height):
        """Programmatically updates Flux's add_dict canvas dimensions per pass."""
        new_cond = []
        for t in conditioning:
            c_dict = t[1].copy()
            add_dict = c_dict.get("add_dict", {}).copy()
            add_dict["size"] = (width, height)
            c_dict["add_dict"] = add_dict
            new_cond.append([t[0], c_dict])
        return new_cond

    def get_sigmas(self, model, scheduler, start_step, steps, shift, denoise, width, height) -> torch.Tensor:
        if steps <= 0:
            return torch.FloatTensor([1.0, 0.0])

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return torch.FloatTensor([])
            total_steps = int(steps / denoise)

        model_sampling = model.get_model_object("model_sampling")

        # Determine active shift value and ensure 'mu' is always defined
        if shift == 1.0:
            seq_len = round((width * height) / (16 * 16))
            mu = compute_empirical_mu(seq_len, total_steps)
            effective_shift = math.exp(mu)
        else:
            effective_shift = shift
            mu = math.log(shift)

        # Apply transformation based on selected scheduler
        if scheduler == "flux":
            timesteps = torch.linspace(1, 0, total_steps + 1)
            sigmas = generalized_time_snr_shift(timesteps, mu, 1.0)
            sigmas[-1] = 0.0
        elif scheduler == "lowstep":
            sigmas = lowstep_sigma(total_steps, effective_shift)
        else:  # bong_tangent
            sigmas = bong_tangent_scheduler(model_sampling, total_steps)
            sigmas = (effective_shift * sigmas) / (1.0 + (effective_shift - 1.0) * sigmas)
            sigmas[-1] = 0.0

        # Handle step slicing, denoise windowing, and start steps
        sigmas = sigmas[-(steps + 1):]

        if total_steps < (len(sigmas) - 1):
            sigmas = sigmas[:total_steps + 1]
            if denoise == 1.0:
                sigmas[-1] = 0.0

        if start_step != 0:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]

        return sigmas