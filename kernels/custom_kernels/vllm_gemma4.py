"""vLLM's Gemma4ForConditionalGeneration with custom_kernels' image encoding; everything else is inherited."""
import torch
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

from .gemma4_vision import encode_images, encoder_for_env


class FastGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    def _process_image_input(self, image_input):
        # LoRA on the vision tower needs vLLM's padded batch, which matches its adapter mapping.
        if self._enable_mm_lora:
            return super()._process_image_input(image_input)
        per_image = encode_images(self.vision_tower, image_input["pixel_values"], image_input["pixel_position_ids"],
                                  self.model_dtype, encoder_for_env())
        projected = self.embed_vision(inputs_embeds=torch.cat(per_image).unsqueeze(0)).squeeze(0)
        return list(projected.split([p.shape[0] for p in per_image]))
