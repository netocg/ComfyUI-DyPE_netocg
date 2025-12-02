import torch
from math import atan, pi


# --- Bong Tangent Sigmas Generator ---

def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    """
    Generates a segment of the sigma schedule using the arctangent function,
    scaled and normalized to fit between 'start' and 'end' values.
    """

    # 1. Calculate the S-curve value at the boundaries (x=0 and x=steps-1)
    # Raw Curve(x) = 0.5 * ((2/pi) * atan(-slope * (x - pivot)) + 1)
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2

    srange = smax - smin
    sscale = start - end

    # Avoid division by zero if smax == smin (e.g., if slope is near zero)
    if srange == 0.0:
        return [start] * steps

    # 2. Normalize and scale the curve values
    sigmas = [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1 / srange) * sscale + end for x in
              range(steps)]

    return sigmas


# --- Bong Tangent Scheduler ---

def bong_tangent_scheduler(model_sampling, steps, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2,
                           slope_2=0.2, pad=False):
    """
    Combines two arctangent sigma segments (start->middle and middle->end)
    to create a custom, adjustable noise schedule.
    """

    # Add steps for boundary conditions, typically for zero-padding or starting at step 1
    # Note: steps in ComfyUI are usually the number of sampler steps, not the total number of points in the sigma list
    total_steps = steps + 2

    # Scale pivots from percentage (0.0-1.0) to step index
    pivot_1_idx = int(total_steps * pivot_1)
    pivot_2_idx = int(total_steps * pivot_2)

    # Calculate the midpoint based on pivots for segment lengths
    midpoint = int((pivot_1_idx + pivot_2_idx) / 2)

    # Scale slopes based on total steps for consistent steepness
    slope_factor = total_steps / 40.0
    slope_1_scaled = slope_1 / slope_factor
    slope_2_scaled = slope_2 / slope_factor

    # Determine segment lengths
    stage_2_len = total_steps - midpoint
    stage_1_len = total_steps - stage_2_len

    # Stage 1: Start -> Middle
    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1_scaled, pivot_1_idx, start, middle)

    # Stage 2: Middle -> End
    # The pivot must be relative to the start of this stage (step 0 of this segment)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2_scaled, pivot_2_idx - stage_1_len, middle, end)

    # Remove the duplicate middle value (the last of Stage 1, first of Stage 2)
    tan_sigmas_1 = tan_sigmas_1[:-1]

    # Optional padding with a final zero sigma
    if pad:
        tan_sigmas_2.append(0.0)

    # Combine and convert to a PyTorch tensor
    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas