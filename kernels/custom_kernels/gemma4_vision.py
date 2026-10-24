"""Faster Gemma 4 image encoding for vLLM: no padding, one host sync per batch, optional torch.compile.

vLLM 0.31 runs Hugging Face's vision tower on images padded to max_soft_tokens * 9 patches. The padding mask sends
SDPA to its memory-efficient kernel, and the mask builder and pooler sync with the host. Padding sits at the end of
every image, so the valid patches are a prefix: encoding only those needs no mask (SDPA picks FlashAttention), and a
k x k average pool over the patch grid reproduces the pooler without its one-hot matmul.

CUSTOM_VISION selects the encoder: "compile" (default) runs the layers under torch.compile, "eager" does not.
CUSTOM_ATTENTION=cudnn lets the encoder's attention use cuDNN, which vLLM switches off for every model; on Blackwell
its kernel is faster than FlashAttention 2.
"""
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MAX_IMAGES_PER_CALL = 16
CUDNN_FIRST = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


def encoder_layers(tower, x, position_ids):
    """Gemma4VisionEncoder.forward without its mask builder: without padding there is no mask and no host sync."""
    position_embeddings = tower.encoder.rotary_emb(x, position_ids)
    for layer in tower.encoder.layers:
        x = layer(x, attention_mask=None, position_embeddings=position_embeddings, position_ids=position_ids)
    return x


def cudnn_encoder_layers(tower, x, position_ids):
    """encoder_layers with cuDNN attention first"""
    with sdpa_kernel(CUDNN_FIRST, set_priority=True):
        return encoder_layers(tower, x, position_ids)


_compiled = {}


def compiled(layers):
    def run(tower, x, position_ids):
        if layers not in _compiled:
            _compiled[layers] = torch.compile(layers, dynamic=True)
            # Dynamo specializes batch size 1 and recompiles for it. vLLM's first call is its startup profiling run,
            # so compile the other batch-size variant there too instead of making a later request wait about a minute.
            if len(x) > 1:
                _compiled[layers](tower, x[:1], position_ids[:1])
            else:
                _compiled[layers](tower, x.repeat(2, 1, 1), position_ids.repeat(2, 1, 1))
        return _compiled[layers](tower, x, position_ids)
    return run


compiled_encoder_layers = compiled(encoder_layers)
compiled_cudnn_encoder_layers = compiled(cudnn_encoder_layers)


def encoder_for_env():
    layers = cudnn_encoder_layers if os.environ.get("CUSTOM_ATTENTION") == "cudnn" else encoder_layers
    return layers if os.environ.get("CUSTOM_VISION") == "eager" else compiled(layers)


def patch_grids(position_ids):
    """(valid patches, grid width, grid height) per image. This .tolist() is the only host sync of a batch."""
    per_image = [torch.stack([(p != -1).all(-1).sum(), p[:, 0].amax() + 1, p[:, 1].amax() + 1]) for p in position_ids]
    return [tuple(grid) for grid in torch.stack(per_image).tolist()]


def pool(tower, hidden, width, height):
    """Gemma4VisionPooler without its one-hot matmul: average each k x k block of the patch grid, then scale."""
    k = tower.config.pooling_kernel_size
    if width % k or height % k:
        raise ValueError(f"{width}x{height} patch grid is not a multiple of the {k}x{k} pooling kernel")
    # Patches are row-major (x fastest), so the (height, width) grid matches Gemma4VisionPooler's order.
    grid = hidden.float().view(len(hidden), height, width, -1).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, k).flatten(2).transpose(1, 2) * tower.pooler.root_hidden_size
    if tower.config.standardize:
        pooled = (pooled - tower.std_bias.float()) * tower.std_scale.float()
    return pooled


def encode_images(tower, pixel_values, position_ids, dtype, encode=encoder_layers):
    """Pooled soft tokens per image, from padded pixel values and patch positions (lists or stacked tensors)."""
    pixel_values, position_ids = list(pixel_values), list(position_ids)
    images_by_grid = {}
    for i, grid in enumerate(patch_grids(position_ids)):
        images_by_grid.setdefault(grid, []).append(i)
    tokens = [None] * len(pixel_values)
    for (n, width, height), images in images_by_grid.items():
        for start in range(0, len(images), MAX_IMAGES_PER_CALL):
            chunk = images[start:start + MAX_IMAGES_PER_CALL]
            pv = torch.stack([pixel_values[i][:n] for i in chunk])
            pos = torch.stack([position_ids[i][:n] for i in chunk])
            no_padding = torch.zeros(pos.shape[:2], dtype=torch.bool, device=pos.device)
            hidden = encode(tower, tower.patch_embedder(pv, pos, no_padding).to(dtype), pos)
            for i, image_tokens in zip(chunk, pool(tower, hidden, width, height)):
                tokens[i] = image_tokens.to(dtype)
    return tokens
