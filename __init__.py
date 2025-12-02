import torch # Required for the tensor creation in bong_tangent_scheduler
from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES

# Import your local modules
try:
    from . import models
    from . import sigmas
    from . import text_encode
except ImportError:
    # This handles cases where the module is imported outside of ComfyUI's environment
    # or the dependency files are missing.
    print("Warning: Failed to import models or sigmas for DyPE_netocg custom node. Check file structure.")
    models = None
    sigmas = None
    text_encode = None

# --- Custom Scheduler Registration ---
if sigmas is not None:
    new_scheduler_name = "bong_tangent"
    if new_scheduler_name not in SCHEDULER_HANDLERS:
        # Check if the scheduler function exists in the imported sigmas module
        if hasattr(sigmas, "bong_tangent_scheduler"):
            bong_tangent_handler = SchedulerHandler(handler=sigmas.bong_tangent_scheduler, use_ms=True)
            SCHEDULER_HANDLERS[new_scheduler_name] = bong_tangent_handler
            SCHEDULER_NAMES.append(new_scheduler_name)
            print(f"DyPE_netocg: Added new custom scheduler '{new_scheduler_name}'.")
        else:
            print(f"DyPE_netocg: Could not find '{new_tangent_scheduler}' function in sigmas.py.")

NODE_CLASS_MAPPINGS = {
    "DyPE_Encode": text_encode.DyPE_Encode,
    "ModelSamplingAdvancedResolution": models.ModelSamplingAdvancedResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DyPE_Encode": "DyPE_Encode",
    "ModelSamplingAdvancedResolution": "Model Sampling (Advanced Resolution)",
}

# Define exports for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]