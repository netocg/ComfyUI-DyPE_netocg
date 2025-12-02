import node_helpers


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
        ## 1. Multi-Token Encoding (from CLIPTextEncodeFlux)
        # Tokenize the short prompt (clip_l) for the main tokens
        tokens = clip.tokenize(clip_l)

        # Tokenize the long prompt (t5xxl) and inject its tokens into the main token dictionary
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        # Encode the tokens using the scheduled encoder, injecting the guidance value
        # This creates the multi-token conditioning data
        positive_cond = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})

        ## 2. Resolution Conditioning (from DyPE_Encode)
        # Create the add_dict containing the target size
        add_dict = {"size": (width, height)}

        # Inject the resolution conditioning into the prompt conditioning
        positive_cond = node_helpers.conditioning_set_values(positive_cond, {"add_dict": add_dict})

        # The function returns the final conditioned prompt and the guidance value
        return (positive_cond, guidance)