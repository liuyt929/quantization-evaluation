import torch
import seaborn as sns
import matplotlib.pyplot as plt
from data_normalization import calibration,smooth_utils
from error_compensate import gptq_utils, svd_utils
from scipy.stats import kurtosis
def print_quantized_values(W):
    ax = sns.histplot(W, bins=100, kde=True)
    plt.title("Weight distribution after INT4 quantization")
    plt.ylabel("Frequency (1e5)")  # 提示单位是百万
    
    # 修改 y 轴刻度标签，除以 1e6 并添加 "M" 表示百万
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(y/1e5)}" for y in yticks])
    
    # plt.tight_layout()
    plt.show()
    plt.savefig("Wquant_value.png", dpi=300, bbox_inches="tight")

def visual_W_distribute(model):
    # W = model.model.layers[0].self_attn.q_proj.weight.detach().to(torch.float32).cpu().view(-1).numpy()

    # sns.histplot(W, bins=100, kde=True)
    # plt.title("Weight distribution")
    # plt.show()
    # plt.savefig("my_plot.png", dpi=300, bbox_inches="tight")
    W = model.model.layers[8].self_attn.q_proj.module.weight.detach().to(torch.float32).cpu().view(-1).numpy()

    sns.histplot(W, bins=100, kde=True)
    plt.title("Weight distribution")
    plt.show()
    plt.savefig("my_plot4.png", dpi=300, bbox_inches="tight")
    W = model.model.layers[8].self_attn.o_proj.module.weight.detach().to(torch.float32).cpu().view(-1).numpy()

    sns.histplot(W, bins=100, kde=True)
    plt.title("Weight distribution")
    plt.show()
    plt.savefig("my_plot5.png", dpi=300, bbox_inches="tight")

# def kurtosis_compute(model, after='False'):
#     for i in range(16):
#         W = model.model.layers[i].self_attn.q_proj.weight.detach()
#         if after=='True':
#             W = model.model.layers[i].self_attn.q_proj.module.weight.detach()
#         # 将矩阵展平后计算
#         kurt = kurtosis(W.flatten().float().cpu(), fisher=True, bias=False)
#         print(f"q_proj: {kurt:.4f}")
#     for i in range(16):
#         W = model.model.layers[i].self_attn.k_proj.weight.detach()
#         if after=='True':
#             W = model.model.layers[i].self_attn.k_proj.module.weight.detach()
#         # 将矩阵展平后计算
#         kurt = kurtosis(W.flatten().float().cpu(), fisher=True, bias=False)
#         print(f"k_proj: {kurt:.4f}")
    
#     for i in range(16):
#         W = model.model.layers[i].self_attn.v_proj.weight.detach()
#         if after=='True':
#             W = model.model.layers[i].self_attn.v_proj.module.weight.detach()
#         # 将矩阵展平后计算
#         kurt = kurtosis(W.flatten().float().cpu(), fisher=True, bias=False)
#         print(f"v_proj: {kurt:.4f}")
#     for i in range(16):
#         W = model.model.layers[i].self_attn.o_proj.weight.detach()
#         if after=='True':
#             W = model.model.layers[i].self_attn.o_proj.module.weight.detach()
#         # 将矩阵展平后计算
#         kurt = kurtosis(W.flatten().float().cpu(), fisher=True, bias=False)
#         print(f"o_proj: {kurt:.4f}")
        
#     for i in range(16):
#         W = model.model.layers[i].mlp.up_proj.weight.detach()
#         if after=='True':
#             W = model.model.layers[i].self_attn.up_proj.module.weight.detach()
#         # 将矩阵展平后计算
#         kurt = kurtosis(W.flatten().float().cpu(), fisher=True, bias=False)
#         print(f"up_proj: {kurt:.4f}")
#     for i in range(16):
#         W = model.model.layers[i].mlp.down_proj.weight.detach()
#         if after=='True':
#             W = model.model.layers[i].self_attn.down_proj.module.weight.detach()
#         # 将矩阵展平后计算
#         kurt = kurtosis(W.flatten().float().cpu(), fisher=True, bias=False)
#         print(f"down: {kurt:.4f}")
#     for i in range(16):
#         W = model.model.layers[i].mlp.gate_proj.weight.detach()
#         if after=='True':
#             W = model.model.layers[i].self_attn.gate_proj.module.weight.detach()
#         # 将矩阵展平后计算
#         kurt = kurtosis(W.flatten().float().cpu(), fisher=True, bias=False)
#         print(f"gate: {kurt:.4f}")
