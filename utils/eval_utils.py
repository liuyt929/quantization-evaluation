# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
import os

import torch
from tqdm import tqdm
from utils import model_utils,quant_utils
import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM


# @torch.no_grad()
# def print_3d_act(model, testenc, dev, args):
#     model.eval()
#     print("print_activations")
#     use_cache = model.config.use_cache
#     model.config.use_cache = False

#     layers = model.model.layers
#     model.model.embed_tokens = model.model.embed_tokens.to(dev)

#     layers[0] = layers[0].to(dev)

#     # Convert the whole text of evaluation dataset into batches of sequences.
#     input_ids = testenc.input_ids  # (1, text_len)
#     nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
#     input_ids = (
#         input_ids[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)
#     )  # (nsamples, seqlen)

#     batch_size = args.bsz
#     input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
#     nbatches = len(input_ids)

#     dtype = next(iter(model.parameters())).dtype
#     # The input of the first decoder layer.
#     inps = torch.zeros(
#         (nbatches, batch_size, model.seqlen, model.config.hidden_size),
#         dtype=dtype,
#         device=dev,
#     )
#     inps = [0] * nbatches
#     cache = {"i": 0, "attention_mask": None}

#     class Catcher(torch.nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module

#         def forward(self, inp, **kwargs):
#             inps[cache["i"]] = inp
#             cache["i"] += 1
#             cache["attention_mask"] = kwargs["attention_mask"]
#             cache["position_ids"] = kwargs["position_ids"]
#             raise ValueError

#     layers[0] = Catcher(layers[0])

#     for i in range(nbatches):
#         batch = input_ids[i]
#         try:
#             model(batch)
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
#     layers[0] = layers[0].cpu()

#     model.model.embed_tokens = model.model.embed_tokens.cpu()
#     position_ids = cache["position_ids"]

#     torch.cuda.empty_cache()
#     outs = [0] * nbatches
#     attention_mask = cache["attention_mask"]

#     for i in range(len(layers)):
#         layer = layers[i].to(dev)
#         # Dump the layer input and output
#         captured_inputs = {
#         "k_proj": [],
#         "o_proj":[],
#         "up_proj":[],
#         "down_proj":[]
#         }
#         def capture_input_hook(name):
#             def hook(module, input, output):
#                 captured_inputs[name].append(input[0].detach().cpu())  # 只存输入
#             return hook
#         if i==2 or i==3 or i==4 or i==5 or i==6 or i==7 or i==10 or i==11 or i==12 or i==13:
#             layer_0_k_proj =layer.self_attn.k_proj
#             layer_0_k_proj.register_forward_hook(capture_input_hook("k_proj"))

#             layer_0_o_proj = layer.self_attn.o_proj
#             layer_0_o_proj.register_forward_hook(capture_input_hook("o_proj"))

#             layer_0_up_proj = layer.mlp.up_proj
#             layer_0_up_proj.register_forward_hook(capture_input_hook("up_proj"))

#             layer_0_down_proj = layer.mlp.down_proj
#             layer_0_down_proj.register_forward_hook(capture_input_hook("down_proj"))
            
#             outs[0] = layer(
#                 inps[0],
#                 attention_mask=attention_mask,
#                 #  defined.
#                 position_ids=position_ids,
#             )[0]
#             quantizer = quant_utils.ActQuantizer()
#             quantizer.configure(4,128,False)
#             q_error=captured_inputs['o_proj'][0][1]-quantizer(captured_inputs['o_proj'][0][1])
#             print(q_error.sum())
#             torch.save(captured_inputs, f"plot/activation_smooth{str(i)}.bin")
#             print_3d.print_attn_activations_3d(captured_inputs,dir='plot/test_smooth/'+str(i))
#         else:
#             outs[0] = layer(
#                 inps[0],
#                 attention_mask=attention_mask,
#                 #  defined.
#                 position_ids=position_ids,
#             )[0]
#         layers[i] = layer.cpu()
#         del layer
#         torch.cuda.empty_cache()
#         inps, outs = outs, inps
from accelerate import Accelerator
@torch.no_grad()
def evaluator(model, testenc, args, accelerator=None):
    """
    完全支持 accelerate 的评估函数
    """
    # 使用 accelerator 的设备管理
    if accelerator is not None:
        device = accelerator.device
        print(f"评估进程 {accelerator.local_process_index} 使用设备: {device}")
    else:
        device = next(model.parameters()).device

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // model.seqlen
    
    # 在分布式环境中分配数据
    if accelerator is not None:
        total_samples = nsamples
        samples_per_process = total_samples // accelerator.num_processes
        start_idx = accelerator.local_process_index * samples_per_process
        end_idx = start_idx + samples_per_process if accelerator.local_process_index != accelerator.num_processes - 1 else total_samples
        nsamples = end_idx - start_idx
        print(f"进程 {accelerator.local_process_index} 处理样本 {start_idx} 到 {end_idx}")
    else:
        start_idx, end_idx = 0, nsamples

    input_ids = (
        input_ids[:, start_idx * model.seqlen : end_idx * model.seqlen]
        .view(nsamples, model.seqlen).to(device)
    )

    batch_size = args.bsz
    input_ids_list = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids_list)

    dtype = next(iter(model.parameters())).dtype


    # 捕获输入的设置
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # 让 accelerator 处理设备放置
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    # 关键修改：使用 accelerator 准备层
    original_first_layer = layers[0]
    # if accelerator is not None:
    #     layers[0] = accelerator.prepare(Catcher(original_first_layer))
    # else:
    #     
    layers[0] = Catcher(original_first_layer)

    # 捕获输入
    for i in range(nbatches):
        batch = input_ids_list[i]
        try:
            model(batch)
        except ValueError:
            pass
    
    # 恢复第一层
    layers[0] = original_first_layer
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches

    # 关键修改：使用 accelerator 准备所有层
    # if accelerator is not None:
    #     layers = accelerator.prepare(layers)
    # 逐层处理
    for i in tqdm(range(len(layers)), desc=f"(Eval) Layers [进程{accelerator.local_process_index if accelerator else 0}]"):
        layer = layers[i]
        # 处理每个批次 - 让 accelerator 处理设备
        for j in range(nbatches):
            # 关键：不要手动移动设备，让 accelerator 处理
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0].to(device)
        
        # 更新输入
        inps, outs = outs, inps

    # 计算 PPL
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    for i in range(nbatches):
        hidden_states = inps[i]
        
        # 应用最后的 layer norm
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        
        # LM head
        lm_logits = model.lm_head(hidden_states.to(dtype))
        
        # 计算损失
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids_list[i][:, 1:].to(shift_logits.device)
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)

    # 收集所有进程的损失
    if accelerator is not None:
        all_nlls = accelerator.gather(torch.cat(nlls))
        if accelerator.is_main_process:
            ppl = torch.exp(all_nlls.mean())
        else:
            ppl = torch.tensor(0.0).to(device)
    else:
        ppl = torch.exp(torch.cat(nlls).mean())

    model.config.use_cache = use_cache
    
    # 只在主进程上打印结果
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")
    
    return ppl.item()      
# @torch.no_grad()
# def evaluator(model, testenc, dev, args):
#     model.eval()

#     use_cache = model.config.use_cache
#     model.config.use_cache = False

#     layers = model.model.layers
#     model.model.embed_tokens = model.model.embed_tokens.to(dev)

#     layers[0] = layers[0].to(dev)

#     # Convert the whole text of evaluation dataset into batches of sequences.
#     input_ids = testenc.input_ids  # (1, text_len)
#     nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
#     input_ids = (
#         input_ids[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)
#     )  # (nsamples, seqlen)

#     batch_size = args.bsz
#     input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
#     nbatches = len(input_ids)

#     dtype = next(iter(model.parameters())).dtype
#     # The input of the first decoder layer.
#     inps = torch.zeros(
#         (nbatches, batch_size, model.seqlen, model.config.hidden_size),
#         dtype=dtype,
#         device=dev,
#     )
#     inps = [0] * nbatches
#     cache = {"i": 0, "attention_mask": None}

#     class Catcher(torch.nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module

#         def forward(self, inp, **kwargs):
#             inps[cache["i"]] = inp
#             cache["i"] += 1
#             cache["attention_mask"] = kwargs["attention_mask"]
#             cache["position_ids"] = kwargs["position_ids"]
#             raise ValueError

#     layers[0] = Catcher(layers[0])

#     for i in range(nbatches):
#         batch = input_ids[i]
#         try:
#             model(batch)
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
#     layers[0] = layers[0].cpu()

#     model.model.embed_tokens = model.model.embed_tokens.cpu()
#     position_ids = cache["position_ids"]

#     torch.cuda.empty_cache()
#     outs = [0] * nbatches
#     attention_mask = cache["attention_mask"]

#     for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
#         layer = layers[i].to(dev)

#         # Dump the layer input and output
#         if args.capture_layer_io and args.layer_idx == i:
#             captured_io = model_utils.capture_layer_io(layer, inps)
#             save_path = model_utils.get_layer_io_save_path(args)
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             torch.save(captured_io, save_path)
#             logging.info(f"Dumped layer input and output to: {save_path}")

#         for j in range(nbatches):
#             outs[j] = layer(
#                 inps[j],
#                 attention_mask=attention_mask,
#                 #  defined.
#                 position_ids=position_ids,
#             )[0]
#         layers[i] = layer.cpu()
#         del layer
#         torch.cuda.empty_cache()
#         inps, outs = outs, inps

#     if model.model.norm is not None:
#         model.model.norm = model.model.norm.to(dev)

#     model.lm_head = model.lm_head.to(dev)
#     nlls = []
#     loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
#     for i in range(nbatches):
#         hidden_states = inps[i]
#         if model.model.norm is not None:
#             hidden_states = model.model.norm(hidden_states)
#         hidden_states=hidden_states.to(dtype)
#         lm_logits = model.lm_head(hidden_states)
#         shift_logits = lm_logits[:, :-1, :]
#         shift_labels = input_ids[i][:, 1:]
#         loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
#         neg_log_likelihood = loss.float().mean(dim=1)
#         nlls.append(neg_log_likelihood)
#     nlls_tensor = torch.cat(nlls)
#     ppl = torch.exp(nlls_tensor.mean())
#     model.config.use_cache = use_cache
#     logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")
#     return ppl.item()


def tasks_evalution(model,tokenizer):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    # task_names = lm_eval_utils.pattern_match(["winogrande","arc_easy","arc_challenge"], ALL_TASKS)

    results = lm_eval.simple_evaluate(hflm, tasks=[
        # "gsm8k",
        # "mmlu",
        # "yelp_review_polarity",
        # "polemo2"
        # "mathqa",
        # "arc_challenge",
        # # # "gpqa",
        # "hellaswag",
        # # # "mgsm_direct",
        #     "boolq",
        # #     # "ifeval",
        #     "lambada_openai",
        # #     # "openbookqa",
        #     "piqa",
        # #     # "social_iqa",
        #     "winogrande"
            ],batch_size=1)['results']
    print("finish")
    print(results)
    # metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    
    # metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    # print(metric_vals)

#sentiment tasks


