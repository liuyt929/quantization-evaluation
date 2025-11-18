# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform
import torch.nn as nn
import geoopt
from geoopt.manifolds import EuclideanStiefel,Stiefel

class RotateModule(nn.Module):
    def __init__(self, R_init=None,*R_inits):
        super(RotateModule, self).__init__()
        if R_init is not None:
            self.weight = geoopt.ManifoldParameter(R_init.to(dtype=torch.float32,device="cuda"),manifold=Stiefel())
        else:
            self.weight = nn.ParameterList([geoopt.ManifoldParameter(R_inits[i].to(dtype=torch.float32,device="cuda"),manifold=Stiefel()) for i in range(len(R_inits))])
    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1)


def rotate_mlp_output(layer, R1):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_)
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output #额外
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.inference_mode()
def rotate_model(model, args):
    R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        dtype=layer.mlp.up_proj.weight.data.dtype
        if args.optimized_rotation_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
        else:
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        rotate_mlp_output(layers[idx], R1)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)
        layer.self_attn.q_proj.weight.data=layer.self_attn.q_proj.weight.to(dtype)
        layer.self_attn.v_proj.weight.data=layer.self_attn.v_proj.weight.to(dtype)
        layer.self_attn.k_proj.weight.data=layer.self_attn.k_proj.weight.to(dtype)
        layer.self_attn.o_proj.weight.data=layer.self_attn.o_proj.weight.to(dtype)
        layer.mlp.gate_proj.weight.data=layer.mlp.gate_proj.weight.to(dtype)
        layer.mlp.up_proj.weight.data=layer.mlp.up_proj.weight.to(dtype)
        layer.mlp.down_proj.weight.data=layer.mlp.down_proj.weight.to(dtype)
        layer.post_attention_layernorm.weight.data=layer.post_attention_layernorm.weight.to(dtype)
        layer.input_layernorm.weight.data=layer.input_layernorm.weight.to(dtype)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)


def rotate_out_proj(layer):
    W=layer.self_attn.o_proj
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )

def rotate_ov_proj_heads(layer,v_proj,o_proj,R_list,num_kv_heads,head_dim,num_attn_heads): 
    
    
    R_ov = torch.stack(R_list,dim=0).to(v_proj.weight.device)
    
    o_proj.weight.data=o_proj.weight.to(v_proj.weight.device)
    
    v_proj.weight.data=((v_proj.weight.T.reshape(-1,num_kv_heads,head_dim).unsqueeze(2).to(torch.float64) @ R_ov.to(torch.float64)).reshape(-1,num_kv_heads*head_dim).T)
    if v_proj.bias is not None:
        b=v_proj.bias     
        # v_proj.bias.data = (b.reshape(num_kv_heads,head_dim).unsqueeze(2) @ R_ov).reshape(v_proj.bias.shape)
        b = b.reshape(num_kv_heads, head_dim).to(torch.float64)  # (2, 128)
        v_proj.bias.data = (b.unsqueeze(1) @ R_ov.to(torch.float64)).squeeze(1).reshape(-1)

            
            
    if layer.self_attn.num_key_value_groups != 1:
        h, d1, d2 = R_ov.shape
        repeated_R_ov = R_ov[:,None,:,:].expand(h, layer.self_attn.num_key_value_groups, d1, d2).reshape(-1, d1, d2)
        o_proj.weight.data = (o_proj.weight.reshape(-1,num_attn_heads,head_dim).unsqueeze(2).to(torch.float64) @ repeated_R_ov.to(torch.float64)).reshape(o_proj.weight.shape)
    else:
        o_proj.weight.data = (o_proj.weight.reshape(-1,num_attn_heads,head_dim).unsqueeze(2).to(torch.float64) @ R_ov.to(torch.float64)).reshape(o_proj.weight.shape)
