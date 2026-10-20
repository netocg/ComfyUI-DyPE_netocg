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

WEB_DIRECTORY = "."

NODE_CLASS_MAPPINGS = {
    "DyPE_FLUX": models.DyPE_FLUX,
    "ModelSamplingAdvancedResolution": models.ModelSamplingAdvancedResolution,
    "DyPE_Encode": text_encode.DyPE_Encode,
    "MultiAttentionConditioning": text_encode.MultiAttentionConditioning,
    "IfConditioningsElseFallback": text_encode.IfConditioningsElseFallback,
    "AdvancedScheduler": sigmas.AdvancedScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DyPE_FLUX": "DyPE_FLUX",
    "ModelSamplingAdvancedResolution": "Model Sampling (Advanced Resolution)",
    "DyPE_Encode": "DyPE Dual Text Encode",
    "MultiAttentionConditioning": "Multi-Attention Conditioning",
    "IfConditioningsElseFallback": "If Conditionings Else Fallback",
    "AdvancedScheduler": "Advanced Scheduler",
}

# Define exports for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]