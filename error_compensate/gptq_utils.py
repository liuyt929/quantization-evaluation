# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import logging
import math
import pprint
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import quant_utils, utils
from error_compensate import svd_utils
import train_utils

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        inp=inp.to(self.H.device)
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        Scale = self.layer.weight.data.clone()
        Scale = Scale.float()
        W_int = self.layer.weight.data.clone()
        W_int = W_int.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W) 

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W_int1 = torch.zeros_like(W1)
            Scale1 = torch.zeros_like(W1).to(Scale.dtype)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q, int_weight, scale = self.quantizer.fake_quantize(w.unsqueeze(1))
                Q1[:, i] = q.flatten()
                q = q.flatten()
                W_int1[:, i] = int_weight.flatten()
                Scale1[:, i] = scale.flatten()

                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_int[:, i1:i2] = W_int1
            Scale[:, i1:i2] = Scale1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        if export_to_et:
            self.layer.register_buffer(
                "int_weight", W_int.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", Scale)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")
        return W_int.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype),Scale

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)
from accelerate import Accelerator

@torch.no_grad()
def gptq_fwrd(model, dataloader, args, accelerator=None):
    logging.info("-----GPTQ Quantization-----")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # 关键修改：不要手动指定设备，让模型保持原有分布
    if accelerator is not None:
        # 在分布式环境中，让每个进程处理自己设备上的层
        device = accelerator.device
        print(f"进程 {accelerator.local_process_index} 使用设备: {device}")
    else:
        device = next(model.parameters()).device

    # 检查模型实际分布
    print("=== 模型设备分布 ===")
    device_params = {}
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            dev_idx = param.device.index
            device_params[dev_idx] = device_params.get(dev_idx, 0) + param.numel()
    
    for dev_idx, param_count in device_params.items():
        print(f"GPU {dev_idx}: {param_count:,} 参数")

    layers = model.model.layers

    # 输入数据 - 只在当前进程的设备上创建
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), 
        dtype=next(iter(model.parameters())).dtype,
        device=device  # 当前进程的设备
    )
    
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # 输入数据自动适配到模块所在的设备
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    # 关键修改：不要移动第一层，保持原有设备
    original_first_layer = layers[0]
    layers[0] = Catcher(original_first_layer)
    
    # 捕获输入 - 数据会自动流向对应设备
    for batch in dataloader:
        try:
            # 让模型自动处理设备放置
            if hasattr(batch, 'to'):
                model(batch.to(device))
            else:
                model(batch[0].to(device))
        except ValueError:
            pass
    
    # 恢复第一层
    layers[0] = original_first_layer
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps, device=device)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    quantizers = {}
    sequential = [
        ["self_attn.k_proj.module", "self_attn.v_proj.module", "self_attn.q_proj.module"],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]
    
    # 关键修改：逐层处理，但不移动层设备
    for i in tqdm(range(len(layers)), desc="GPTQ Processing"):
        if accelerator is not None:
            # 在分布式环境中，让每个进程只处理特定层
            if i % accelerator.num_processes != accelerator.local_process_index:
                continue
                
        print(f"\n进程 {accelerator.local_process_index if accelerator else 0} 处理第 {i} 层", flush=True)
        
        layer = layers[i]  # 不要移动层！保持原有设备
        
        # 检查该层所在的设备
        layer_device = next(layer.parameters()).device
        print(f"第 {i} 层在设备: {layer_device}")

        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            
            for name in subset:
                print(f"{name}(设备:{subset[name].weight.device})", end="  ", flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8

                # GPTQ 在层所在的设备上执行
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    # 数据会自动流向层所在的设备
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            # 前向传播 - 数据会自动分发
            for j in range(args.nsamples):
                # 输入数据在当前进程设备，但层可能在其他设备
                # PyTorch 会自动处理设备间的数据传输
                layer_input = inps[j].unsqueeze(0)
                outs[j] = layer(
                    layer_input,
                    attention_mask=attention_mask.to(layer_device) if attention_mask is not None else None,
                    position_ids=position_ids.to(layer_device) if position_ids is not None else None,
                )[0].to(device)  # 输出移回当前进程设备
            
            for h in handles:
                h.remove()

            # 量化操作在层所在的设备上执行
            for name in subset:
                layer_device = subset[name].weight.device
                layer_w_groupsize = args.w_groupsize
                
                if args.enable_low_rank:
                    origin_W = gptq[name].layer.weight.data.clone()

                # 量化
                W_int, scale = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et
                )
                
                if args.enable_low_rank:
                    after_W = gptq[name].layer.weight.data.clone()
                    parent_name = 'model.layers.' + str(i) + '.' + name.replace(".module", "")
                    qlayers = quant_utils.find_qlayers(
                        model.model.layers[i], 
                        layers=[quant_utils.ActQuantWrapper],
                        name='model.layers.' + str(i)
                    )
                    qlayers[parent_name].add_low_rank(True, args.rank)
                    svd_utils.LayerSvdCompute.approximate_with_qweight(
                        qlayers[parent_name], parent_name, origin_W, after_W, None
                    )
                
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        # 更新输入
        for j in range(args.nsamples):
            layer_input = inps[j].unsqueeze(0)
            outs[j] = layer(
                layer_input,
                attention_mask=attention_mask.to(layer_device) if attention_mask is not None else None,
                position_ids=position_ids.to(layer_device) if position_ids is not None else None,
            )[0].to(device)
        
        # 不要移动层！保持原有分布
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # 在分布式环境中同步所有进程
    # if accelerator is not None:
    #     accelerator.wait_for_everyone()

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers
@torch.no_grad()
def rtn_fwrd(model, dev, args, custom_layers=None):
    """
    From GPTQ repo
    """
    # assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    if custom_layers:
        layers = custom_layers
    else:
        layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, torch.nn.Embedding]
        )

        for name in subset:
            layer_weight_bits = args.w_bits
            w_groupsize = args.w_groupsize
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8
            if args.export_to_et:
                layer_weight_bits = 8  # all per channel 8 bits for executorch export
                w_groupsize = -1
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
                weight_groupsize=w_groupsize,
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            q, int_weight, scale = quantizer.fake_quantize(W)
            
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype) #normal
                    
            if args.export_to_et:
                subset[name].register_buffer("int_weight", int_weight)
                subset[name].register_buffer("scale", scale)
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers
