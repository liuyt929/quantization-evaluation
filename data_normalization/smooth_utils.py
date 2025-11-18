import torch
import torch.nn as nn

# from models.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm  
class SmoothModule(nn.Module):
    def __init__(self, S_init):
        super(SmoothModule, self).__init__()
        self.weight = nn.Parameter(S_init.to(torch.float32).to(torch.device("cuda")))
    def forward(self, x, inverse=False):
        if inverse:
            x_smooth = x / self.weight
        else:
            x_smooth = x * self.weight 
        return x_smooth
       
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        # weight_scales.pow(1-alpha)/act_scales.pow(alpha)
        # act_scales/act_scales
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, LlamaRMSNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features 
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales= act_scales.to(device=device, dtype=dtype)
   
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)


    
    #一共hiddenstates个scale，每个scale计算时只和邻近的128个计算
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        # weight_scales.pow(1-alpha)/act_scales.pow(alpha)
        # act_scales/act_scales
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales,alpha=0.8):
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm

            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales,alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)

def smooth_ln_fcs_llama_like_2(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = ln.weight.device, ln.weight.dtype
    scales=scales.to(device)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        fc.to(device)

    ln.weight.data=(ln.weight.data.to(torch.float64)/scales.to(torch.float64))
    for fc in fcs:
        fc.weight.data= (fc.weight.data.to(torch.float64)*scales.to(torch.float64))
    
    
def smooth_ln_ov_llama_like(v_proj,o_proj,scales,num_key_value_heads,num_key_value_groups):
    device, dtype = v_proj.weight.device, v_proj.weight.dtype
    scales=scales.to(device)
    o_proj.weight.data = o_proj.weight.to(v_proj.weight.device)
    v_proj.weight.data = (v_proj.weight.to(torch.float64) / (scales.to(torch.float64).view(-1,1)))
    if v_proj.bias is not None:
        v_proj.bias.data = (v_proj.bias.to(torch.float64) / (scales.to(torch.float64).view(-1)))
    if o_proj.weight.shape[-1] > scales.numel():
        scales = scales.reshape(num_key_value_heads,-1)
        n_head,d = scales.shape
        scales = scales[:,None,:].expand(n_head,num_key_value_groups,d).reshape(-1)
    o_proj.weight.data = (o_proj.weight.to(torch.float64) * (scales.to(torch.float64).view(1,-1)))   

def smooth_updown(up_proj,down_proj,scales):
    device, dtype = up_proj.weight.device, up_proj.weight.dtype
    scales=scales.to(device)
    up_proj.weight.data = (up_proj.weight.to(torch.float64) / (scales.to(torch.float64).view(-1,1)))
    if up_proj.bias is not None:
        up_proj.bias.data = (up_proj.bias.to(torch.float64) / (scales.to(torch.float64).view(-1)))
    
    down_proj.weight.data =  (down_proj.weight.to(torch.float64) * (scales.to(torch.float64).view(1,-1)))
   

