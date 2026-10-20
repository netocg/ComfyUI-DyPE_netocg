import comfy.model_sampling
import comfy.supported_models
import torch
import torch.nn as nn
import math
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import ComfyExtension, io

def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    low = math.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = math.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def find_newbase_ntk(dim, base, scale):
    return base * (scale ** (dim / (dim - 2)))


def get_1d_rotary_pos_embed(
        dim: int,
        pos: torch.Tensor,
        theta: float = 10000.0,
        use_real=False,
        linear_factor=1.0,
        ntk_factor=1.0,
        repeat_interleave_real=True,
        freqs_dtype=torch.float32,
        yarn=False,
        max_pe_len=None,
        ori_max_pe_len=64,
        dype=False,
        current_timestep=1.0,
        dype_exponent=2.0,
):
    assert dim % 2 == 0
    device = pos.device

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=device)

        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        beta_0, beta_1 = 1.25, 0.75
        gamma_0, gamma_1 = 16, 2

        freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
        freqs_linear = 1.0 / torch.einsum('..., f -> ... f', scale,
                                          (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)))

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(new_base, (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        if dype:
            beta_0 = beta_0 ** (dype_exponent * (current_timestep ** dype_exponent))
            beta_1 = beta_1 ** (dype_exponent * (current_timestep ** dype_exponent))

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low, high = max(0, low), min(dim // 2, high)

        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        if dype:
            gamma_0 = gamma_0 ** (dype_exponent * (current_timestep ** dype_exponent))
            gamma_1 = gamma_1 ** (dype_exponent * (current_timestep ** dype_exponent))

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low, high = max(0, low), min(dim // 2, high)

        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask

    else:
        theta_ntk = theta * ntk_factor
        if dype and ntk_factor > 1.0:
            theta_ntk = theta * (ntk_factor ** (dype_exponent * (current_timestep ** dype_exponent)))

        freqs = 1.0 / (theta_ntk ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)) / linear_factor

    freqs = torch.einsum("...s,d->...sd", pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1).float()

        if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
            mscale = torch.where(scale <= 1., torch.tensor(1.0), 0.1 * torch.log(scale) + 1.0).to(scale)
            freqs_cos, freqs_sin = freqs_cos * mscale, freqs_sin * mscale
        return freqs_cos, freqs_sin
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', dype: bool = True,
                 dype_exponent: float = 2.0):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.dype = dype if method != 'base' else False
        self.dype_exponent = dype_exponent
        self.current_timestep = 1.0
        self.base_resolution = 1024
        self.base_patches = (self.base_resolution // 8) // 2

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb_parts = []
        pos = ids.float()
        freqs_dtype = torch.bfloat16

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]

            common_kwargs = {
                'dim': axis_dim,
                'pos': axis_pos,
                'theta': self.theta,
                'repeat_interleave_real': True,
                'use_real': True,
                'freqs_dtype': freqs_dtype
            }
            dype_kwargs = {
                'dype': self.dype,
                'current_timestep': self.current_timestep,
                'dype_exponent': self.dype_exponent
            }

            if i > 0:
                max_pos = axis_pos.max().item()
                current_patches = int(max_pos + 1)

                if self.method == 'yarn' and current_patches > self.base_patches:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, yarn=True, max_pe_len=max_pe_len,
                                                       ori_max_pe_len=self.base_patches, **dype_kwargs)
                elif self.method == 'ntk' and current_patches > self.base_patches:
                    base_ntk_scale = (current_patches / self.base_patches)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, ntk_factor=base_ntk_scale, **dype_kwargs)
                else:
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)


def apply_dype_to_flux(model: ModelPatcher, width: int, height: int, method: str, enable_dype: bool,
                       dype_exponent: float) -> ModelPatcher:
    m = model.clone()

    # Automatically scale the exponent based on resolution scale using width and height
    patch_size = 2  # Flux patch size
    latent_h, latent_w = height // 8, width // 8
    padded_h = math.ceil(latent_h / patch_size) * patch_size
    padded_w = math.ceil(latent_w / patch_size) * patch_size
    image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)

    base_seq_len = 4096  # 1024x1024 base tokens
    # Scale linearly with token sequence length (image_seq_len / base_seq_len)
    seq_ratio = image_seq_len / base_seq_len  # 1.0 at 1k, 4.0 at 2k, 16.0 at 4k
    effective_exponent = dype_exponent * (1.0 + (max(1.0, seq_ratio) - 1.0) / 15.0)

    try:
        orig_embedder = m.model.diffusion_model.pe_embedder
        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX model supporting pe_embedder.")

    new_pe_embedder = FluxPosEmbed(theta, axes_dim, method, enable_dype, effective_exponent)
    m.add_object_patch("diffusion_model.pe_embedder", new_pe_embedder)

    sigma_max = 1.0
    if hasattr(m.model.model_sampling, "sigma_max") and m.model.model_sampling.sigma_max is not None:
        sigma_max = m.model.model_sampling.sigma_max.item()

    def dype_wrapper_function(model_function, args_dict):
        if enable_dype:
            timestep_tensor = args_dict.get("timestep")
            if timestep_tensor is not None and timestep_tensor.numel() > 0:
                current_sigma = timestep_tensor[0].item()
                if sigma_max > 0:
                    normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                    new_pe_embedder.set_timestep(normalized_timestep)

        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)

    return m

class DyPE_FLUX(io.ComfyNode):
    """
    Applies DyPE (Dynamic Position Extrapolation) to a FLUX model.
    This allows generating images at resolutions far beyond the model's training scale
    by dynamically adjusting positional encodings and the noise schedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_FLUX",
            display_name="DyPE for FLUX",
            category="model_patches/unet",
            description="Applies DyPE (Dynamic Position Extrapolation) to a FLUX model for ultra-high-resolution generation.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The FLUX model to patch with DyPE.",
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image width. Must match the width of your empty latent."
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image height. Must match the height of your empty latent."
                ),
                io.Combo.Input(
                    "method",
                    options=["yarn", "ntk", "base"],
                    default="yarn",
                    tooltip="Position encoding extrapolation method (YARN recommended).",
                ),
                io.Boolean.Input(
                    "enable_dype",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable Dynamic Position Extrapolation for RoPE.",
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0, min=0.0, max=4.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE strength over time (λt). 2.0=Exponential (best for 4K+), 1.0=Linear, 0.5=Sub-linear (better for ~2K)."
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The FLUX model patched with DyPE.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, method: str, enable_dype: bool,
                dype_exponent: float = 2.0) -> io.NodeOutput:
        """
        Clones the model and applies the DyPE patch for both the noise schedule and positional embeddings.
        """
        if not hasattr(model.model, "diffusion_model") or not hasattr(model.model.diffusion_model, "pe_embedder"):
            raise ValueError("This node is only compatible with FLUX models.")

        patched_model = apply_dype_to_flux(model, width, height, method, enable_dype, dype_exponent)
        return io.NodeOutput(patched_model)

# deprecated ModelSamplingAdvancedResolution node/code below
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