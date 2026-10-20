import nodes
import node_helpers
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import comfy.model_management
import comfy.utils
from nodes import ConditioningCombine


class DyPE_Encode:
    """
    A combined text encoding node for FLUX models that supports dual text encoders
    (CLIP L and T5-XXL tokens) and resolution conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_l": ("STRING", {"multiline": True, "default": "A beautiful landscape", "dynamic_prompts": True}),
                "t5xxl": ("STRING", {"multiline": True, "default": "detailed background with stunning atmosphere",
                                     "dynamic_prompts": True}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "display": "number"}),
                "width": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16, "display": "number"}),
            },
        }

    # We will return the conditioned prompt and the guidance value
    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("positive_conditioning", "guidance_out")
    FUNCTION = "encode"
    CATEGORY = "DyPE/Encoding"

    def encode(self, clip, clip_l, t5xxl, guidance, width, height, **kwargs):
        # 1. Tokenize and encode base prompts
        tokens = {}
        if clip_l:
            tokens = clip.tokenize(clip_l)
        if t5xxl:
            tokens["t5xxl"] = clip.tokenize(t5xxl).get("t5xxl")

        positive_cond = clip.encode_from_tokens_scheduled(tokens)

        # 2. Build a SINGLE dictionary for add_dict
        add_dict = {
            "size": (width, height),
            "guidance": guidance
        }

        # 3. Inject add_dict into conditioning via node_helpers
        positive_cond = node_helpers.conditioning_set_values(positive_cond, {"add_dict": add_dict})

        return (positive_cond, guidance)


class MultiAttentionConditioning:
    """
    A combined text encoding and regional masking node for FLUX models
    supporting dual text encoders (CLIP L and T5-XXL), resolution conditioning,
    reference latents, and advanced mask manipulation. Automatically adapts
    canvas dimensions from reference images or masks, scaling with inputcount.
    """

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP", {"tooltip": "Clip_l and t5xxl Dual clip for conditioning workflow using Flux model."}),
                "vae": ("VAE", {"tooltip": "VAE from Flux workflow."}),
                "reference_latents_method": (["offset", "index", "uxo/uno", "index_timestep_zero"],
                                             {"default": "uxo/uno", "tooltip": "Select how connected Reference images will be translated into conditioning."}),
                "width": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 8,
                                  "tooltip": "Set the width to be used by the DyPE conditioning method and Latent resolution when no images or masks are used."}),
                "height": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 8,
                                   "tooltip": "Set the height to be used by the DyPE conditioning method and Latent resolution when no images or masks are used."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1,
                                       "tooltip": "Number of active regional conditioning branches."}),
            },
            "optional": {}  # <-- FIXED: Initialized optional key to prevent KeyError
        }
        # Dynamically populate optional inputs up to the maximum inputcount (20)
        for i in range(1, 21):
            inputs["optional"].update({
                f"clip_l_{i}": ("STRING", {"multiline": False, "default": "", "dynamic_prompts": True, "forceInput": True,
                                           "tooltip": "Clip_l prompt for Dual clip conditioning workflow using Flux model."}),
                f"t5xxl_{i}": ("STRING", {"multiline": False, "default": "", "dynamic_prompts": True, "forceInput": True,
                                          "tooltip": "t5xxl prompt for Dual clip conditioning workflow using Flux model."}),
                f"guidance_{i}": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "precision": 1,
                                            "tooltip": "Sets the classifier-free guidance scale for this conditioning."}),
                f"reference_image_{i}": ("IMAGE", {"tooltip": "Optional Reference Image used for structural conditioning latents."}),
                f"mask_{i}": ("MASK", {"tooltip": "Optional Mask to set where in the canvas the conditioning will be applied."}),
                f"mask_{i}_blur": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                                           "tooltip": "Gaussian blur applied to the mask image."}),
                f"mask_{i}_resize": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1,
                                             "tooltip": "Expand or Shrink the mask size."}),
                f"mask_{i}_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "precision": 2,
                                                 "tooltip": "How much this mask blends with the others."}),
            })
        return inputs

    RETURN_TYPES = ("CONDITIONING", "IMAGE", "MASK", "LATENT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("conditioning_combined", "references_combined", "masks_combined", "empty_latent", "use_reference", "use_masks")
    FUNCTION = "combine"
    CATEGORY = "DyPE/Encoding/masking/conditioning"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."
    SEARCH_ALIASES = ["combine", "empty latent", "encoding", "masking", "cross-attention", "conditioning", "DyPE"]

    def combine(self, clip, vae, reference_latents_method, width, height, batch_size, inputcount, **kwargs):
        use_reference = False
        use_masks = False
        clip = clip.clone()
        for t5_type in ["t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl"]:
            clip.set_tokenizer_option(f"{t5_type}_min_padding", 0)
            clip.set_tokenizer_option(f"{t5_type}_min_length", 0)

        conditioning_combined = None
        device = comfy.model_management.intermediate_device()
        cond_combine_node = ConditioningCombine()

        # 1. Automatically determine canvas resolution from the first available reference image or mask
        canvas_h, canvas_w = height, width
        for i in range(1, inputcount + 1):
            ref_img = kwargs.get(f"reference_image_{i}", None)
            mask_tensor = kwargs.get(f"mask_{i}", None)

            if ref_img is not None:
                canvas_h, canvas_w = ref_img.shape[1], ref_img.shape[2]
                break
            elif mask_tensor is not None:
                canvas_h, canvas_w = mask_tensor.shape[-2], mask_tensor.shape[-1]
                break

        # Initialize canvas tensors
        masks_combined = torch.zeros((1, canvas_h, canvas_w), dtype=torch.float32, device=device)
        references_combined = torch.zeros((1, canvas_h, canvas_w, 3), dtype=torch.float32, device=device)

        # Helper function mirroring ConditioningSetMask behavior
        def append_helper(cond_list, mask_tensor, strength_val, set_area_to_bounds):
            new_cond = []
            for t in cond_list:
                n = [t[0], t[1].copy()]
                n[1]['mask'] = mask_tensor
                n[1]['set_area_to_bounds'] = set_area_to_bounds
                n[1]['mask_strength'] = strength_val
                new_cond.append(n)
            return new_cond

        # 2. Loop strictly through the active inputcount specified by the user
        for i in range(1, inputcount + 1):
            clip_l = kwargs.get(f"clip_l_{i}", "") or ""
            t5xxl = kwargs.get(f"t5xxl_{i}", "") or ""
            guidance = kwargs.get(f"guidance_{i}", 1.0)
            ref_image = kwargs.get(f"reference_image_{i}", None)
            mask = kwargs.get(f"mask_{i}", None)
            blur_amount = kwargs.get(f"mask_{i}_blur", 0)
            resize_amount = kwargs.get(f"mask_{i}_resize", 0)
            strength = kwargs.get(f"mask_{i}_strength", 1.0)

            # Tokenize and encode base prompts
            tokens = clip.tokenize(clip_l)
            t5_tokens = clip.tokenize(t5xxl)
            if "t5xxl" in t5_tokens:
                tokens["t5xxl"] = t5_tokens["t5xxl"]

            current_cond = clip.encode_from_tokens_scheduled(tokens)

            # Inject add_dict (Resolution & Guidance) using resolved canvas dimensions
            add_dict = {
                "size": (canvas_w, canvas_h),
                "guidance": guidance
            }
            current_cond = node_helpers.conditioning_set_values(current_cond, {"add_dict": add_dict})

            # Reset processed_mask each iteration to prevent scope leakage between branches
            processed_mask = None

            # Process Mask (Resize, Blur, and Region Application via Helper)
            if mask is not None:
                mask = mask.to(device)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)

                if mask.shape[1] != canvas_h or mask.shape[2] != canvas_w:
                    mask = comfy.utils.common_upscale(mask.unsqueeze(1), canvas_w, canvas_h, "bilinear",
                                                      "center").squeeze(1)

                if resize_amount != 0:
                    kernel_size = abs(resize_amount) * 2 + 1
                    if resize_amount > 0:  # Expand
                        mask = F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1,
                                            padding=abs(resize_amount)).squeeze(1)
                    else:  # Shrink
                        mask = -F.max_pool2d(-mask.unsqueeze(1), kernel_size=kernel_size, stride=1,
                                             padding=abs(resize_amount)).squeeze(1)

                if blur_amount > 0:
                    if blur_amount % 2 == 0:
                        blur_amount += 1
                    mask = TF.gaussian_blur(mask.unsqueeze(1), kernel_size=[blur_amount, blur_amount]).squeeze(1)

                processed_mask = torch.clamp(mask, 0.0, 1.0)
                masks_combined = torch.maximum(masks_combined, processed_mask)

                # Apply the mask via the user's requested helper pattern
                current_cond = append_helper(current_cond, processed_mask, strength, set_area_to_bounds=True)
                use_masks = True
            else:
                default_mask = torch.ones((1, canvas_h, canvas_w), dtype=torch.float32, device=device)
                masks_combined = torch.maximum(masks_combined, default_mask)

            # Handle Reference Image to Latent Conditioning & Compositing
            if ref_image is not None:
                ref_image = ref_image.to(device)
                if ref_image.shape[1] != canvas_h or ref_image.shape[2] != canvas_w:
                    ref_image_perm = ref_image.movedim(-1, 1)
                    ref_image_perm = comfy.utils.common_upscale(ref_image_perm, canvas_w, canvas_h, "bilinear",
                                                                "center")
                    ref_image = ref_image_perm.movedim(1, -1)

                # Encode and inject reference latents into current branch condition with append=True
                latents = vae.encode(ref_image[:, :, :, :3])
                current_cond = node_helpers.conditioning_set_values(
                    current_cond,
                    {"reference_latents": [latents]},
                    append=True
                )

                # Apply mask to reference image channels if a mask exists for this branch
                if processed_mask is not None:
                    masked_ref = ref_image * processed_mask.unsqueeze(-1)
                else:
                    masked_ref = ref_image

                references_combined = torch.maximum(references_combined, masked_ref[:, :, :, :3])
                use_reference = True

            # Merge conditionings
            if conditioning_combined is None:
                conditioning_combined = current_cond
            else:
                conditioning_combined = cond_combine_node.combine(conditioning_combined, current_cond)[0]

        if conditioning_combined is None:
            raise ValueError("MultiAttentionConditioning: At least one valid prompt branch must be provided.")

        # Apply reference latents method normalization
        if "uxo" in reference_latents_method or "uso" in reference_latents_method:
            reference_latents_method = "uxo"

        conditioning_combined = node_helpers.conditioning_set_values(
            conditioning_combined,
            {"reference_latents_method": reference_latents_method}
        )

        latent_samples = torch.zeros((batch_size, 4, canvas_h // 8, canvas_w // 8), dtype=torch.float32, device=device)
        empty_latent = {"samples": latent_samples}

        return (conditioning_combined, references_combined, masks_combined, empty_latent, use_reference, use_masks)

class IfConditioningsElseFallback:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Fallback": ("CONDITIONING",),
                "operation": (["combine", "concat"], {"default": "combine"}),
            },
            "optional": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_1_switch": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled",
                                               "tooltip": "Use or not conditioning_1"}),

                "conditioning_2": ("CONDITIONING",),
                "conditioning_2_switch": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                                      "tooltip": "Use or not conditioning_2"}),
                "conditioning_3": ("CONDITIONING",),
                "conditioning_3_switch": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                                      "tooltip": "Use or not conditioning_3"}),
                "conditioning_4": ("CONDITIONING",),
                "conditioning_4_switch": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                                      "tooltip": "Use or not conditioning_4"}),
                "conditioning_5": ("CONDITIONING",),
                "conditioning_5_switch": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                                      "tooltip": "Use or not conditioning_5"}),
                "conditioning_6": ("CONDITIONING",),
                "conditioning_6_switch": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                                      "tooltip": "Use or not conditioning_6"}),
                "conditioning_7": ("CONDITIONING",),
                "conditioning_7_switch": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                                      "tooltip": "Use or not conditioning_6"}),
                "conditioning_8": ("CONDITIONING",),
                "conditioning_8_switch": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled",
                                                      "tooltip": "Use or not conditioning_6"}),
            },
        }

    OUTPUT_TOOLTIPS = ("Concatenated or Combined conditionings, otherwise pass the Fallback.",)

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("combined", )
    FUNCTION = "combine"
    CATEGORY = "logic/conditioning"
    DESCRIPTION = """
    Combines multiple conditioning nodes into one
    """

    @staticmethod
    def combine(operation, **kwargs):
        from nodes import ConditioningCombine
        from nodes import ConditioningConcat
        cond_combine_node = ConditioningCombine()
        cond_concat_node = ConditioningConcat()
        cond = kwargs["Fallback"]
        for c in range(1, 8):
            new_cond = kwargs[f"conditioning_{c}"]
            if kwargs[f"conditioning_{c}_switch"]:
                if operation == "combine":
                    cond = cond_combine_node.combine(cond, new_cond)[0]
                elif operation == "concat":
                    cond = cond_concat_node.concat(cond, new_cond)[0]
        return (cond, )