import comfy.model_sampling
import comfy.supported_models
import torch
import math


def time_snr_shift_exponential(alpha, t):
    return math.exp(alpha) / (math.exp(alpha) + (1 / t - 1) ** 1.0)

def time_snr_shift_linear(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class ModelSamplingAdvancedResolution:
    # this is used to set the "shift" using either exponential scaling (default for SD3.5M and Flux) or linear scaling (default for SD3.5L and SD3 2B beta)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "scaling": (["exponential", "linear"], {"default": 'exponential'}),
            "max_shift": ("FLOAT", {"default": 1.15, "min": -100.0, "max": 100.0, "step": 0.01, "round": False}),
            "base_shift": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01, "round": False}),
            "latent_image": ("LATENT",),
        }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "DyPE/model_shift"

    def sigma_exponential(self, timestep):
        return time_snr_shift_exponential(self.timestep_shift, timestep / self.multiplier)

    def sigma_linear(self, timestep):
        return time_snr_shift_linear(self.timestep_shift, timestep / self.multiplier)

    def main(self, model, scaling, max_shift, base_shift, latent_image):
        m = model.clone()

        height, width = latent_image['samples'].shape[-2:]
        # frames is not used in the shift calculation but kept for completeness
        frames = latent_image['samples'].shape[-3] if latent_image['samples'].ndim == 5 else 1

        # --- Shift Calculation ---
        x1 = 256  # Base latent area (e.g., 512*512 / 256)
        x2 = 4096  # Max latent area
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1

        # Calculate latent area equivalent (width * height / 256)
        latent_area_eq = width * height / (8 * 8 * 2 * 2)

        # Linear interpolation for shift
        shift = latent_area_eq * mm + b

        self.timestep_shift = shift

        # --- Model Configuration and Patching ---
        self.multiplier = 1000
        timesteps = 1000
        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        sampling_type = comfy.model_sampling.EPS

        # Simplified configuration check (original code was very long, keeping only the essential logic structure)
        if isinstance(m.model.model_config,
                      (comfy.supported_models.Flux, comfy.supported_models.FluxSchnell, comfy.supported_models.Chroma)):
            self.multiplier = 1
            timesteps = 10000
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST

        elif isinstance(m.model.model_config,
                        (comfy.supported_models.SD3, comfy.supported_models.AuraFlow, comfy.supported_models.HiDream,
                         comfy.supported_models.HunyuanVideo, comfy.supported_models.WAN21_T2V,
                         comfy.supported_models.WAN21_I2V, comfy.supported_models.LTXV)):
            self.multiplier = 1000
            timesteps = 1000
            # SD3, AuraFlow, etc., use a flow-based sampling structure
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST

        elif isinstance(m.model.model_config, (comfy.supported_models.CosmosT2V, comfy.supported_models.CosmosI2V)):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingContinuousEDM
            sampling_type = comfy.model_sampling.CONST

        # Create the new ModelSampling class dynamically
        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        m.object_patches['model_sampling'] = m.model.model_sampling = ModelSamplingAdvanced(m.model.model_config)

        # Apply the calculated shift and multiplier
        m.model.model_sampling.__dict__['shift'] = self.timestep_shift
        m.model.model_sampling.__dict__['multiplier'] = self.multiplier

        # --- Recalculate Sigmas ---
        s_range = torch.arange(1, timesteps + 1, 1).to(torch.float64)

        if scaling == "exponential":
            # This calls the sigma_exponential method which needs external helper function
            ts = self.sigma_exponential((s_range / timesteps) * self.multiplier)
        elif scaling == "linear":
            # This calls the sigma_linear method which needs external helper function
            ts = self.sigma_linear((s_range / timesteps) * self.multiplier)

        m.model.model_sampling.register_buffer('sigmas', ts)
        m.object_patches['model_sampling'].sigmas = m.model.model_sampling.sigmas

        return (m,)