import functools
import math

import torch
import tqdm
import re
from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)

from data_normalization.rotation_utils import rotate_embeddings,rotate_attention_inputs,\
                                      rotate_attention_output,rotate_mlp_input,rotate_mlp_output,rotate_head,\
                                      rotate_out_proj,rotate_ov_proj_heads
import torch,transformers,sys,os,torch.nn as nn,typing,utils,transformers,tqdm,math

from utils.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2

from data_normalization.smooth_utils import smooth_ln_fcs_llama_like_2,smooth_ln_ov_llama_like,smooth_updown

@torch.no_grad()
def rotate_smooth_model_inplace(model,st_path,args):
    utils.utils.cleanup_memory(False)
    st = torch.load(st_path, map_location="cuda")
    R1=torch.load(st_path)["R_res.weight"].cuda().to(torch.float64) #R1
    model.to('cuda')
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    num_kv_heads = config.num_key_value_heads
    num_key_value_groups=num_heads //num_kv_heads
    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        dtype=layer.mlp.up_proj.weight.data.dtype
        layer=layer.to('cuda')
        pattern = r"model\.layers\." + str(idx) + r"\.R_S_modules\.R_ov\.weight\.(\d+)"

        R_ov_key = [(int(re.search(pattern, key).group(1)), key)for key in st.keys() if re.search(pattern, key)]
        
        R_ov_key.sort(key=lambda x: x[0])
        R_list = [st[key] for (_, key) in R_ov_key]
        #ov
        if args.smooth_ov:
            smooth_ln_ov_llama_like(layer.self_attn.v_proj,layer.self_attn.o_proj, st['model.layers.'+str(idx)+'.R_S_modules.S_ov.weight'],num_kv_heads,num_key_value_groups)
        # up down
        if args.smooth_up_down:
            smooth_updown(layer.mlp.up_proj, layer.mlp.down_proj,st['model.layers.'+str(idx)+'.R_S_modules.S_up_down.weight'])
        rotate_attention_inputs(layer, R1)
        rotate_attention_output(layer, R1)
        rotate_mlp_input(layer, R1)
        rotate_mlp_output(layer, R1)
        # rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)
        attn_ln = layer.input_layernorm  # attention forward norm
        qkv = [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
        ]
        
        rotate_ov_proj_heads(layer,layer.self_attn.v_proj,layer.self_attn.o_proj,R_list,num_kv_heads,head_dim,num_heads)
        # rotate_ov_proj(layer,num_heads,head_dim,R2)
        
        ffn_ln = layer.post_attention_layernorm  # feed forward norm
        fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
        if args.smooth_norm_linear:
            smooth_ln_fcs_llama_like_2(attn_ln, qkv, st['model.layers.'+str(idx)+'.R_S_modules.S_norm_qkv.weight'])
            smooth_ln_fcs_llama_like_2(ffn_ln, fcs, st['model.layers.'+str(idx)+'.R_S_modules.S_norm_upgate.weight'])
        layer.self_attn.q_proj.weight.data=layer.self_attn.q_proj.weight.to(dtype)
        layer.self_attn.v_proj.weight.data=layer.self_attn.v_proj.weight.to(dtype)
        layer.self_attn.k_proj.weight.data=layer.self_attn.k_proj.weight.to(dtype)
        layer.self_attn.o_proj.weight.data=layer.self_attn.o_proj.weight.to(dtype)
        layer.mlp.gate_proj.weight.data=layer.mlp.gate_proj.weight.to(dtype)
        layer.mlp.up_proj.weight.data=layer.mlp.up_proj.weight.to(dtype)
        layer.mlp.down_proj.weight.data=layer.mlp.down_proj.weight.to(dtype)
        layer.post_attention_layernorm.weight.data=layer.post_attention_layernorm.weight.to(dtype)
        layer.input_layernorm.weight.data=layer.input_layernorm.weight.to(dtype)
        
        
