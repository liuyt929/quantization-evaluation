import torch

captured_inputs = {
    "k_proj": [],
    "o_proj": [],
    "gate_proj": [],
    "down_proj": [],
}

captured_outputs = {
    "v_proj": [],
}

# 定义 forward hook
def capture_input_hook(name):
    def hook(module, input, output):
        captured_inputs[name].append(input[0].detach().cpu())  # 只存输入
    return hook

def capture_output_hook(name):
    def hook(module, input, output):
        captured_outputs[name].append(output.detach().cpu())  # 只存输出
    return hook

def register_hook(model):
    layer_0_k_proj = model.model.layers[0].self_attn.k_proj.module
    layer_0_k_proj.register_forward_hook(capture_input_hook("k_proj"))
    layer_0_q_proj = model.model.layers[0].self_attn.q_proj.module
    layer_0_q_proj.register_forward_hook(capture_input_hook("q_proj"))
    layer_0_v_proj = model.model.layers[0].self_attn.v_proj.module
    layer_0_v_proj.register_forward_hook(capture_input_hook("v_proj"))
    layer_0_o_proj = model.model.layers[0].self_attn.o_proj.module
    layer_0_o_proj.register_forward_hook(capture_input_hook("o_proj"))
    layer_0_up_proj = model.model.layers[0].mlp.up_proj.module
    layer_0_up_proj.register_forward_hook(capture_input_hook("up_proj"))
    layer_0_down_proj = model.model.layers[0].mlp.down_proj.module
    layer_0_down_proj.register_forward_hook(capture_input_hook("down_proj"))
    layer_0_gate_proj = model.model.layers[0].mlp.gate_proj.module
    layer_0_gate_proj.register_forward_hook(capture_input_hook("gate_proj"))

    print("Hooks registered!")
    return model