import math
import torch
import nodes
from comfy_extras.nodes_custom_sampler import io
from math import atan, pi

# --- Bong Tangent Sigmas Generator ---
def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    """
    Generates a segment of the sigma schedule using the arctangent function,
    scaled and normalized to fit between 'start' and 'end' values.
    """
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2

    srange = smax - smin
    sscale = start - end

    if srange == 0.0:
        return [start] * steps

    sigmas = [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in range(steps)]

    return sigmas

def bong_tangent_scheduler(model_sampling, steps, start=1.0, middle=0.5, end=0.0,
                           pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    """
    Combines two arctangent sigma segments with an integrated flow-matching
    """
    total_steps = steps + 2

    pivot_1_idx = int(total_steps * pivot_1)
    pivot_2_idx = int(total_steps * pivot_2)

    midpoint = int((pivot_1_idx + pivot_2_idx) / 2)

    slope_factor = total_steps / 40.0
    slope_1_scaled = slope_1 / slope_factor
    slope_2_scaled = slope_2 / slope_factor

    stage_2_len = total_steps - midpoint
    stage_1_len = total_steps - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1_scaled, pivot_1_idx, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2_scaled, pivot_2_idx - stage_1_len, middle, end)

    tan_sigmas_1 = tan_sigmas_1[:-1]

    if pad:
        tan_sigmas_2.append(0.0)

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas

# lowstep sigma generator
def lowstep_sigma(steps: int, shift: float = 3.0) -> torch.Tensor:
    if steps <= 0:
        return torch.FloatTensor([1.0, 0.0])
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    sigmas = 1.0 - t
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    sigmas[-1] = 0.0
    return sigmas

# flux sigma generator
def generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


class AdvancedScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AdvancedScheduler",
            category="sampling/custom_sampling/schedulers",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("scheduler", options=["flux", "lowstep", "bong_tangent"]),
                io.Int.Input("start_step", default=0, min=0, max=10000),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("shift", default=1.0, min=0.1, max=100.0, step=0.1),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, model, scheduler, start_step, steps, shift, denoise, width, height) -> io.NodeOutput:
        if steps <= 0:
            return io.NodeOutput(torch.FloatTensor([1.0, 0.0]))

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return io.NodeOutput(torch.FloatTensor([]))
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
        else: # bong_tangent
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

        return io.NodeOutput(sigmas)

    get_sigmas = execute