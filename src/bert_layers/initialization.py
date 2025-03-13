# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 OLMo Authors
# License: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# License: Apache-2.0

import math
from typing import Optional, Union

import torch
import torch.nn as nn

from src.utils import StrEnum

from .configuration_bert import FlexBertConfig
from .normalization import RMSNorm

__all__ = ["init_weights", "ModuleType", "InitFnType"]


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    default = "default"
    """
    All weights are initialized with the default HuggingFace Bert method. Set init_std=0.02 to match.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: FlexBertConfig,
    module: Union[nn.Linear, nn.Embedding],
    layer_dim: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param layer_dim: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    if config.init_method == InitFnType.full_megatron and config.init_small_embedding:
        raise ValueError("Cannot use 'small_embedding_init' with 'full_megatron' init.")

    layer_dim = layer_dim if layer_dim is not None else config.hidden_size
    if config.init_method == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_method == InitFnType.mitchell:
        std = std_factor / math.sqrt(layer_dim)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_method == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_method == InitFnType.fan_in:
        std = std_factor / math.sqrt(layer_dim)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_method == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.num_hidden_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.hidden_size**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")

        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    elif config.init_method == InitFnType.default:
        # default hugging face bert initialization
        # normalization layers already init to ones and zeros
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    else:
        raise NotImplementedError(config.init_method)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_method == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.num_hidden_layers))

    if isinstance(module, nn.Embedding) and config.init_small_embedding:
        nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)


class TileMode(StrEnum):
    center_weights = "center_weights"
    tile_weights_from_edge = "tile_weights_from_edge"
    tile_weights_from_middle = "tile_weights_from_middle"


def tile_weight(
    pretrained_weights: torch.Tensor,
    new_weights: torch.Tensor,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
) -> torch.Tensor:
    """
    Tile or center an input tensor to a larger desired size. Works for both 2D and 1D tensors.

    Args:
    pretrained_weights (torch.Tensor): The input tensor to be tiled or centered (1D or 2D).
    new_weights (torch.Tensor): The tensor with the desired size.
    mode (Union[str, TileMode]): 'center_weights', 'tile_weights_from_edge', or 'tile_weights_from_middle'

    Returns:
    torch.Tensor: The resulting tensor of the desired size.
    """
    assert pretrained_weights.dim() in (1, 2), "Input tensor must be 1-dimensional or 2-dimensional"
    if isinstance(mode, str):
        mode = TileMode(mode)

    pretrained_weights = pretrained_weights.clone()

    if pretrained_weights.dim() == 1:
        return _tile_1d(pretrained_weights, new_weights, mode)
    else:
        return _tile_2d(pretrained_weights, new_weights, mode)


def _tile_1d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 1, "Input tensor must be 1-dimensional"
    input_size = pretrained_weights.shape[0]
    new_size = new_weights.shape[0]
    assert new_size >= input_size, "Desired size must be greater than or equal to input size"

    if mode == TileMode.center_weights:
        offset = (new_size - input_size) // 2
        new_weights[offset : offset + input_size] = pretrained_weights
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_count = (new_size + input_size - 1) // input_size
        tiled_tensor = pretrained_weights.repeat(repeat_count)
        return tiled_tensor[:new_size].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        offset = (new_size - input_size) // 2

        # Create a new tensor with the desired size
        result = torch.zeros(new_size, dtype=pretrained_weights.dtype, device=pretrained_weights.device)

        # Place the original tensor in the center
        result[offset : offset + input_size] = pretrained_weights

        # Tile the left and right sides
        for i in range(offset):
            result[offset - 1 - i] = pretrained_weights[input_size - 1 - (i % input_size)]
        for i in range(offset + input_size, new_size):
            result[i] = pretrained_weights[(i - offset) % input_size]
        return result.clone()


def _tile_2d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 2, "Input tensor must be 2-dimensional"
    input_height, input_width = pretrained_weights.shape
    new_height, new_width = new_weights.shape
    assert new_height >= input_height, "Desired height must be greater than or equal to input height"
    assert new_width >= input_width, "Desired width must be greater than or equal to input width"

    if mode == TileMode.center_weights:
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2
        new_weights[height_offset : height_offset + input_height, width_offset : width_offset + input_width] = pretrained_weights  # fmt: skip
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_height = (new_height + input_height - 1) // input_height
        repeat_width = (new_width + input_width - 1) // input_width
        tiled_tensor = pretrained_weights.repeat(repeat_height, repeat_width)
        return tiled_tensor[:new_height, :new_width].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2

        # Create a new tensor with the desired width and input height
        horizontal_tiled = torch.zeros(
            input_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device
        )

        # Place the original tensor in the center horizontally
        horizontal_tiled[:, width_offset : width_offset + input_width] = pretrained_weights

        # Tile the left and right sides
        for i in range(width_offset):
            horizontal_tiled[:, i] = horizontal_tiled[
                :, width_offset + input_width - 1 - (width_offset - i - 1) % input_width
            ]
        for i in range(width_offset + input_width, new_width):
            horizontal_tiled[:, i] = horizontal_tiled[:, width_offset + (i - width_offset) % input_width]

        # Now tile vertically
        result = torch.zeros(new_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device)
        result[height_offset : height_offset + input_height, :] = horizontal_tiled

        # Tile top
        for i in range(height_offset):
            row_to_copy = (input_height - 1) - (i % input_height)
            result[height_offset - 1 - i, :] = horizontal_tiled[row_to_copy, :]

        # Tile bottom
        for i in range(height_offset + input_height, new_height):
            row_to_copy = (i - height_offset) % input_height
            result[i, :] = horizontal_tiled[row_to_copy, :]
        return result.clone()


def tile_fused_qkv(
    pretrained_qkv_weight: torch.Tensor,
    new_qkv_weight: torch.Tensor,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a fused pretrained QKV layer to a new, larger QKV dimension.

    Args:
        pretrained_qkv_weight (torch.Tensor): The original fused QKV layer
        new_qkv_weight (torch.Tensor): The new fused QKV layer with larger linear_dim
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused QKV layer with tiled weights
    """
    # Split QKV, assume new_q, new_k, new_v are the same shape
    pretrained_q, pretrained_k, pretrained_v = pretrained_qkv_weight.chunk(3, dim=0)
    new_q, new_k, new_v = new_qkv_weight.chunk(3, dim=0)

    # Tile Q, K, V separately
    new_q = tile_weight(pretrained_q, new_q, mode=mode)
    new_k = tile_weight(pretrained_k, new_k, mode=mode)
    new_v = tile_weight(pretrained_v, new_v, mode=mode)

    # Concatenate tiled Q, K, V
    return torch.cat([new_q, new_k, new_v], dim=0)


def tile_fused_glu(
    pretrained_glu_weight: torch.Tensor,
    new_glu_weight: torch.Tensor,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a fused pretrained GLU layer to a new, larger GLU dimension.

    Args:
        pretrained_glu_weight (torch.Tensor): The original fused GLU layer
        new_glu_weight (torch.Tensor): The new fused GLU layer with larger linear_dim
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused GLU layer with tiled weights
    """
    # Split GLU, assume new_glu_wi, new_glu_wg are the same shape
    pretrained_glu_wi, pretrained_glu_wg = pretrained_glu_weight.chunk(2, dim=0)
    new_glu_wi, new_glu_wg = new_glu_weight.chunk(2, dim=0)

    # Tile GLU separately
    new_glu_wi = tile_weight(pretrained_glu_wi, new_glu_wi, mode=mode)
    new_glu_wg = tile_weight(pretrained_glu_wg, new_glu_wg, mode=mode)

    # Concatenate tiled GLU
    return torch.cat([new_glu_wi, new_glu_wg], dim=0)


def tile_fused_qkvff(
    pretrained_qkvff_weight: torch.Tensor,
    new_qkvff_weight: torch.Tensor,
    pretrained_attn_size: int,
    pretrained_mlp_size: int,
    new_attn_size: int,
    new_mlp_size: int,
    is_glu: bool = False,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a fused pretrained QKVFF layer to a new, larger QKVFF dimension.

    Args:
        pretrained_qkvff_weight (torch.Tensor): The original fused QKVFF layer
        new_qkvff_weight (torch.Tensor): The new fused QKVFF layer with larger linear_dim
        pretrained_attn_size (int): The attention size of the pretrained fused QKVFF layer
        pretrained_mlp_size (int): The mlp size of the pretrained fused QKVFF layer
        new_attn_size (int): The attention size of the new fused QKVFF layer
        new_mlp_size (int): The mlp size of the new fused QKVFF layer
        is_glu (bool): Whether the QKVFF layer is a GLU layer
        mode (Union[str, TileMode]): The tiling mode to use
    Returns:
        torch.Tensor: The new fused QKVFF layer with tiled weights
    """
    # Split QKVFF
    pretrained_qkv, pretrained_ff = pretrained_qkvff_weight.split([pretrained_attn_size, pretrained_mlp_size], dim=0)
    new_qkv, new_ff = new_qkvff_weight.split([new_attn_size, new_mlp_size], dim=0)

    # Tile QKVFF separately
    new_qkv = tile_fused_qkv(pretrained_qkv, new_qkv, mode=mode)
    if is_glu:
        new_ff = tile_fused_glu(pretrained_ff, new_ff, mode=mode)
    else:
        new_ff = tile_weight(pretrained_ff, new_ff, mode=mode)

    # Concatenate tiled QKVFF
    return torch.cat([new_qkv, new_ff], dim=0)


class TileLinear(StrEnum):
    wqkv = "wqkv"
    glu = "glu"
    wqkvff = "wqkvff"
    default = "default"


def tile_linear(
    pretrained_linear: nn.Linear,
    new_linear: nn.Linear,
    linear_type: Union[str, TileLinear] = TileLinear.default,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
    pretrained_attn_size: Optional[int] = None,
    pretrained_mlp_size: Optional[int] = None,
    new_attn_size: Optional[int] = None,
    new_mlp_size: Optional[int] = None,
    wqkvff_is_glu: Optional[bool] = None,
    bias_only: Optional[bool] = False,
):
    """
    Tile the weights of a linear layer to a new, larger linear dimension.

    Args:
        pretrained_linear (nn.Linear): The original linear layer
        new_linear (nn.Linear): The new linear layer with larger linear_dim
        linear_type (Union[str, TileLinear]): The type of linear layer to tile
        mode (Union[str, TileMode]): The tiling mode to use
        pretrained_attn_size (int): The attention size of the pretrained linear layer. Only used if linear_type is wqkvff.
        pretrained_mlp_size (int): The mlp size of the pretrained linear layer. Only used if linear_type is wqkvff.
        new_attn_size (int): The attention size of the new linear layer. Only used if linear_type is wqkvff.
        new_mlp_size (int): The mlp size of the new linear layer. Only used if linear_type is wqkvff.
        wqkvff_is_glu (bool): Whether the wqkvff layer is a GLU layer. Only used if linear_type is wqkvff.
        bias_only (bool): Whether to only tile the bias. Only used if tiling weight tied decoder.
    """
    if isinstance(linear_type, str):
        linear_type = TileLinear(linear_type)
    if isinstance(mode, str):
        mode = TileMode(mode)

    with torch.no_grad():
        if linear_type == TileLinear.wqkv:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_qkv(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_qkv(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        elif linear_type == TileLinear.glu:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_glu(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_glu(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        elif linear_type == TileLinear.wqkvff:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_fused_qkvff(
                        pretrained_linear.weight,
                        new_linear.weight,
                        pretrained_attn_size,
                        pretrained_mlp_size,
                        new_attn_size,
                        new_mlp_size,
                        wqkvff_is_glu,
                        mode=mode,
                    ),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_fused_qkvff(
                        pretrained_linear.bias,
                        new_linear.bias,
                        pretrained_attn_size,
                        pretrained_mlp_size,
                        new_attn_size,
                        new_mlp_size,
                        wqkvff_is_glu,
                        mode=mode,
                    ),
                    requires_grad=new_linear.bias.requires_grad,
                )
        else:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    tile_weight(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    tile_weight(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )


def tile_norm(
    pretrained_norm: Union[nn.LayerNorm, RMSNorm, nn.Identity],
    new_norm: Union[nn.LayerNorm, RMSNorm, nn.Identity],
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
):
    """
    Tile the weights of a pretrained norm layer to a new, larger layer norm dimension.

    Args:
        pretrained_norm (Union[nn.LayerNorm, RMSNorm, nn.Identity]): The original norm layer
        new_norm (Union[nn.LayerNorm, RMSNorm, nn.Identity]): The new norm layer with larger layer norm dimension
        mode (Union[str, TileMode]): The Phi-style weight tiling mode to use
    """
    if isinstance(pretrained_norm, nn.Identity):
        return
    if isinstance(mode, str):
        mode = TileMode(mode)

    with torch.no_grad():
        new_norm.weight.data = nn.Parameter(
            tile_weight(pretrained_norm.weight, new_norm.weight, mode=mode),
            requires_grad=new_norm.weight.requires_grad,
        )
        if hasattr(pretrained_norm, "bias") and pretrained_norm.bias is not None:
            new_norm.bias.data = nn.Parameter(
                tile_weight(pretrained_norm.bias, new_norm.bias, mode=mode),
                requires_grad=new_norm.bias.requires_grad,
            )


def tile_embedding(
    pretrained_embedding: nn.Embedding,
    new_embedding: nn.Embedding,
    mode: Union[str, TileMode] = TileMode.tile_weights_from_middle,
) -> nn.Embedding:
    """
    Tile the weights of an embedding layer to a new, larger embedding dimension.

    Args:
    pretrained_embedding (nn.Embedding): The original embedding layer
    new_embedding (nn.Embedding): The new embedding layer with larger embedding_dim
    tile_mode (Union[str, TileMode]): The Phi-style weight tiling mode to use

    Returns:
    nn.Embedding: The new embedding layer with tiled weights
    """
    with torch.no_grad():
        # Ensure vocabulary size remains the same
        if pretrained_embedding.num_embeddings != new_embedding.num_embeddings:
            raise ValueError("Vocabulary size (num_embeddings) must remain constant")

        # Ensure new embedding dimension is larger
        if new_embedding.embedding_dim <= pretrained_embedding.embedding_dim:
            raise ValueError("New embedding_dim must be larger than the old embedding_dim")

        # Tile the weights
        new_embedding.weight.data = nn.Parameter(
            tile_weight(pretrained_embedding.weight, new_embedding.weight, mode=mode),
            requires_grad=new_embedding.weight.requires_grad,
        )

        # Handle padding_idx if it exists
        if pretrained_embedding.padding_idx is not None:
            if new_embedding.padding_idx is None:
                new_embedding.padding_idx = pretrained_embedding.padding_idx
            else:
                assert new_embedding.padding_idx == pretrained_embedding.padding_idx, "padding_idx must remain the same"


### Subselecting


class SubselectMode(StrEnum):
    subselect_middle = "subselect_middle"
    """Select the middle portion of the weights."""
    
    subselect_first = "subselect_first"
    """Select the first/beginning portion of the weights."""
    
    subselect_last = "subselect_last"
    """Select the last/end portion of the weights."""
    
    subselect_strided = "subselect_strided"
    """Select weights in a strided fashion based on the size difference."""


def subselect_weight(
    pretrained_weights: torch.Tensor,
    new_weights: torch.Tensor,
    mode: Union[str, SubselectMode] = SubselectMode.subselect_middle,
) -> torch.Tensor:
    """
    Subselect a portion of an input tensor to a smaller desired size. Works for both 2D and 1D tensors.

    Args:
    pretrained_weights (torch.Tensor): The larger input tensor to be subselected from (1D or 2D).
    new_weights (torch.Tensor): The tensor with the desired smaller size.
    mode (Union[str, SubselectMode]): 'subselect_middle', 'subselect_first', 'subselect_last', or 'subselect_strided'

    Returns:
    torch.Tensor: The resulting tensor of the desired smaller size.
    """
    assert pretrained_weights.dim() in (1, 2), "Input tensor must be 1-dimensional or 2-dimensional"
    if isinstance(mode, str):
        mode = SubselectMode(mode)

    # Ensure the pretrained weights are larger than the new weights
    if pretrained_weights.dim() == 1:
        assert pretrained_weights.shape[0] >= new_weights.shape[0], "Pretrained size must be greater than or equal to desired size"
        return _subselect_1d(pretrained_weights, new_weights, mode)
    else:
        assert pretrained_weights.shape[0] >= new_weights.shape[0] and pretrained_weights.shape[1] >= new_weights.shape[1], \
            "Pretrained dimensions must be greater than or equal to desired dimensions"
        return _subselect_2d(pretrained_weights, new_weights, mode)


def _subselect_1d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: SubselectMode) -> torch.Tensor:
    """Subselect from a 1D tensor to a smaller size."""
    assert pretrained_weights.dim() == 1, "Input tensor must be 1-dimensional"
    pretrained_size = pretrained_weights.shape[0]
    new_size = new_weights.shape[0]
    
    if mode == SubselectMode.subselect_middle:
        # Select the middle portion
        start_idx = (pretrained_size - new_size) // 2
        new_weights = pretrained_weights[start_idx:start_idx + new_size].clone()
    
    elif mode == SubselectMode.subselect_first:
        # Select the first portion
        new_weights = pretrained_weights[:new_size].clone()
    
    elif mode == SubselectMode.subselect_last:
        # Select the last portion
        new_weights = pretrained_weights[-new_size:].clone()
    
    elif mode == SubselectMode.subselect_strided:
        # Select with stride
        # Calculate the stride needed to evenly sample the pretrained weights
        stride = pretrained_size / new_size
        indices = torch.linspace(0, pretrained_size - 1, new_size).long()
        new_weights = pretrained_weights[indices].clone()
    
    return new_weights


def _subselect_2d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: SubselectMode) -> torch.Tensor:
    """Subselect from a 2D tensor to a smaller size."""
    assert pretrained_weights.dim() == 2, "Input tensor must be 2-dimensional"
    pretrained_height, pretrained_width = pretrained_weights.shape
    new_height, new_width = new_weights.shape
    
    if mode == SubselectMode.subselect_middle:
        # Select the middle portion
        height_start = (pretrained_height - new_height) // 2
        width_start = (pretrained_width - new_width) // 2
        new_weights = pretrained_weights[
            height_start:height_start + new_height, 
            width_start:width_start + new_width
        ].clone()
    
    elif mode == SubselectMode.subselect_first:
        # Select the first/top-left portion
        new_weights = pretrained_weights[:new_height, :new_width].clone()
    
    elif mode == SubselectMode.subselect_last:
        # Select the last/bottom-right portion
        new_weights = pretrained_weights[-new_height:, -new_width:].clone()
    
    elif mode == SubselectMode.subselect_strided:
        # Select with stride for both dimensions
        # Calculate the strides needed to evenly sample the pretrained weights
        height_stride = pretrained_height / new_height
        width_stride = pretrained_width / new_width
        
        # Create indices for rows and columns
        row_indices = [int(i * height_stride) for i in range(new_height)]
        col_indices = [int(j * width_stride) for j in range(new_width)]
        
        # Use advanced indexing to extract the strided values
        for i, row_idx in enumerate(row_indices):
            for j, col_idx in enumerate(col_indices):
                new_weights[i, j] = pretrained_weights[row_idx, col_idx].clone()
    
    return new_weights


def subselect_fused_qkv(
    pretrained_qkv_weight: torch.Tensor,
    new_qkv_weight: torch.Tensor,
    mode: Union[str, SubselectMode] = SubselectMode.subselect_middle,
):
    """
    Subselect portions of a fused pretrained QKV layer to a new, smaller QKV dimension.

    Args:
        pretrained_qkv_weight (torch.Tensor): The original fused QKV layer (larger)
        new_qkv_weight (torch.Tensor): The new fused QKV layer with smaller dimensions
        mode (Union[str, SubselectMode]): The subselection mode to use
    Returns:
        torch.Tensor: The new fused QKV layer with subselected weights
    """
    # Split QKV, assume new_q, new_k, new_v are the same shape
    pretrained_q, pretrained_k, pretrained_v = pretrained_qkv_weight.chunk(3, dim=0)
    new_q, new_k, new_v = new_qkv_weight.chunk(3, dim=0)

    # Subselect Q, K, V separately
    new_q = subselect_weight(pretrained_q, new_q, mode=mode)
    new_k = subselect_weight(pretrained_k, new_k, mode=mode)
    new_v = subselect_weight(pretrained_v, new_v, mode=mode)

    # Concatenate subselected Q, K, V
    return torch.cat([new_q, new_k, new_v], dim=0)


def subselect_fused_glu(
    pretrained_glu_weight: torch.Tensor,
    new_glu_weight: torch.Tensor,
    mode: Union[str, SubselectMode] = SubselectMode.subselect_middle,
):
    """
    Subselect portions of a fused pretrained GLU layer to a new, smaller GLU dimension.

    Args:
        pretrained_glu_weight (torch.Tensor): The original fused GLU layer (larger)
        new_glu_weight (torch.Tensor): The new fused GLU layer with smaller dimensions
        mode (Union[str, SubselectMode]): The subselection mode to use
    Returns:
        torch.Tensor: The new fused GLU layer with subselected weights
    """
    # Split GLU, assume new_glu_wi, new_glu_wg are the same shape
    pretrained_glu_wi, pretrained_glu_wg = pretrained_glu_weight.chunk(2, dim=0)
    new_glu_wi, new_glu_wg = new_glu_weight.chunk(2, dim=0)

    # Subselect GLU separately
    new_glu_wi = subselect_weight(pretrained_glu_wi, new_glu_wi, mode=mode)
    new_glu_wg = subselect_weight(pretrained_glu_wg, new_glu_wg, mode=mode)

    # Concatenate subselected GLU
    return torch.cat([new_glu_wi, new_glu_wg], dim=0)


def subselect_fused_qkvff(
    pretrained_qkvff_weight: torch.Tensor,
    new_qkvff_weight: torch.Tensor,
    pretrained_attn_size: int,
    pretrained_mlp_size: int,
    new_attn_size: int,
    new_mlp_size: int,
    is_glu: bool = False,
    mode: Union[str, SubselectMode] = SubselectMode.subselect_middle,
):
    """
    Subselect portions of a fused pretrained QKVFF layer to a new, smaller QKVFF dimension.

    Args:
        pretrained_qkvff_weight (torch.Tensor): The original fused QKVFF layer (larger)
        new_qkvff_weight (torch.Tensor): The new fused QKVFF layer with smaller dimensions
        pretrained_attn_size (int): The attention size of the pretrained fused QKVFF layer
        pretrained_mlp_size (int): The mlp size of the pretrained fused QKVFF layer
        new_attn_size (int): The attention size of the new fused QKVFF layer
        new_mlp_size (int): The mlp size of the new fused QKVFF layer
        is_glu (bool): Whether the QKVFF layer is a GLU layer
        mode (Union[str, SubselectMode]): The subselection mode to use
    Returns:
        torch.Tensor: The new fused QKVFF layer with subselected weights
    """
    # Split QKVFF
    pretrained_qkv, pretrained_ff = pretrained_qkvff_weight.split([pretrained_attn_size, pretrained_mlp_size], dim=0)
    new_qkv, new_ff = new_qkvff_weight.split([new_attn_size, new_mlp_size], dim=0)

    # Subselect QKVFF separately
    new_qkv = subselect_fused_qkv(pretrained_qkv, new_qkv, mode=mode)
    if is_glu:
        new_ff = subselect_fused_glu(pretrained_ff, new_ff, mode=mode)
    else:
        new_ff = subselect_weight(pretrained_ff, new_ff, mode=mode)

    # Concatenate subselected QKVFF
    return torch.cat([new_qkv, new_ff], dim=0)


class SubselectLinear(StrEnum):
    wqkv = "wqkv"
    glu = "glu"
    wqkvff = "wqkvff"
    default = "default"


def subselect_linear(
    pretrained_linear: nn.Linear,
    new_linear: nn.Linear,
    linear_type: Union[str, SubselectLinear] = SubselectLinear.default,
    mode: Union[str, SubselectMode] = SubselectMode.subselect_middle,
    pretrained_attn_size: Optional[int] = None,
    pretrained_mlp_size: Optional[int] = None,
    new_attn_size: Optional[int] = None,
    new_mlp_size: Optional[int] = None,
    wqkvff_is_glu: Optional[bool] = None,
    bias_only: Optional[bool] = False,
):
    """
    Subselect portions of a pretrained linear layer to a new, smaller linear dimension.

    Args:
        pretrained_linear (nn.Linear): The original linear layer (larger)
        new_linear (nn.Linear): The new linear layer with smaller dimensions
        linear_type (Union[str, SubselectLinear]): The type of linear layer to subselect
        mode (Union[str, SubselectMode]): The subselection mode to use
        pretrained_attn_size (int): The attention size of the pretrained linear layer. Only used if linear_type is wqkvff.
        pretrained_mlp_size (int): The mlp size of the pretrained linear layer. Only used if linear_type is wqkvff.
        new_attn_size (int): The attention size of the new linear layer. Only used if linear_type is wqkvff.
        new_mlp_size (int): The mlp size of the new linear layer. Only used if linear_type is wqkvff.
        wqkvff_is_glu (bool): Whether the wqkvff layer is a GLU layer. Only used if linear_type is wqkvff.
        bias_only (bool): Whether to only subselect the bias. Only used if weight tied decoder.
    """
    if isinstance(linear_type, str):
        linear_type = SubselectLinear(linear_type)
    if isinstance(mode, str):
        mode = SubselectMode(mode)

    with torch.no_grad():
        if linear_type == SubselectLinear.wqkv:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    subselect_fused_qkv(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    subselect_fused_qkv(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        elif linear_type == SubselectLinear.glu:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    subselect_fused_glu(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    subselect_fused_glu(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )
        elif linear_type == SubselectLinear.wqkvff:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    subselect_fused_qkvff(
                        pretrained_linear.weight,
                        new_linear.weight,
                        pretrained_attn_size,
                        pretrained_mlp_size,
                        new_attn_size,
                        new_mlp_size,
                        wqkvff_is_glu,
                        mode=mode,
                    ),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    subselect_fused_qkvff(
                        pretrained_linear.bias,
                        new_linear.bias,
                        pretrained_attn_size,
                        pretrained_mlp_size,
                        new_attn_size,
                        new_mlp_size,
                        wqkvff_is_glu,
                        mode=mode,
                    ),
                    requires_grad=new_linear.bias.requires_grad,
                )
        else:
            if not bias_only:
                new_linear.weight = nn.Parameter(
                    subselect_weight(pretrained_linear.weight, new_linear.weight, mode=mode),
                    requires_grad=new_linear.weight.requires_grad,
                )
            if pretrained_linear.bias is not None:
                new_linear.bias = nn.Parameter(
                    subselect_weight(pretrained_linear.bias, new_linear.bias, mode=mode),
                    requires_grad=new_linear.bias.requires_grad,
                )


def subselect_norm(
    pretrained_norm: Union[nn.LayerNorm, RMSNorm, nn.Identity],
    new_norm: Union[nn.LayerNorm, RMSNorm, nn.Identity],
    mode: Union[str, SubselectMode] = SubselectMode.subselect_middle,
):
    """
    Subselect portions of a pretrained norm layer to a new, smaller layer norm dimension.

    Args:
        pretrained_norm (Union[nn.LayerNorm, RMSNorm, nn.Identity]): The original norm layer (larger)
        new_norm (Union[nn.LayerNorm, RMSNorm, nn.Identity]): The new norm layer with smaller dimensions
        mode (Union[str, SubselectMode]): The subselection mode to use
    """
    if isinstance(pretrained_norm, nn.Identity):
        return
    if isinstance(mode, str):
        mode = SubselectMode(mode)

    with torch.no_grad():
        new_norm.weight.data = nn.Parameter(
            subselect_weight(pretrained_norm.weight, new_norm.weight, mode=mode),
            requires_grad=new_norm.weight.requires_grad,
        )
        if hasattr(pretrained_norm, "bias") and pretrained_norm.bias is not None:
            new_norm.bias.data = nn.Parameter(
                subselect_weight(pretrained_norm.bias, new_norm.bias, mode=mode),
                requires_grad=new_norm.bias.requires_grad,
            )


def subselect_embedding(
    pretrained_embedding: nn.Embedding,
    new_embedding: nn.Embedding,
    mode: Union[str, SubselectMode] = SubselectMode.subselect_middle,
) -> nn.Embedding:
    """
    Subselect portions of an embedding layer to a new, smaller embedding dimension.

    Args:
    pretrained_embedding (nn.Embedding): The original embedding layer (larger)
    new_embedding (nn.Embedding): The new embedding layer with smaller embedding_dim
    mode (Union[str, SubselectMode]): The subselection mode to use

    Returns:
    nn.Embedding: The new embedding layer with subselected weights
    """
    with torch.no_grad():
        # Ensure vocabulary size remains the same
        if pretrained_embedding.num_embeddings != new_embedding.num_embeddings:
            raise ValueError("Vocabulary size (num_embeddings) must remain constant")

        # Ensure new embedding dimension is smaller
        if new_embedding.embedding_dim >= pretrained_embedding.embedding_dim:
            raise ValueError("New embedding_dim must be smaller than the old embedding_dim")

        # Subselect the weights
        new_embedding.weight.data = nn.Parameter(
            subselect_weight(pretrained_embedding.weight, new_embedding.weight, mode=mode),
            requires_grad=new_embedding.weight.requires_grad,
        )

        # Handle padding_idx if it exists
        if pretrained_embedding.padding_idx is not None:
            if new_embedding.padding_idx is None:
                new_embedding.padding_idx = pretrained_embedding.padding_idx
            else:
                assert new_embedding.padding_idx == pretrained_embedding.padding_idx, "padding_idx must remain the same"