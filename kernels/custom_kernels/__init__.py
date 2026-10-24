"""vLLM plugin for the Gemma 4 inspection models, loaded through the vllm.general_plugins entry point."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model("Gemma4ForConditionalGeneration",
                                 "custom_kernels.vllm_gemma4:FastGemma4ForConditionalGeneration")
