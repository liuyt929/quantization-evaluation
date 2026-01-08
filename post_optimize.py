from data_normalization import calibration,smooth_utils
from error_compensate import gptq_utils, svd_utils

import utils
import torch,torch.nn as nn,torch.nn.functional as F,transformers
from loguru import logger
from train_utils.ost_train import rotate_smooth_train

from logging import Logger

import torch
from transformers import LlamaTokenizerFast,LlamaForCausalLM,AutoTokenizer,AutoModelForCausalLM

import transformers
# from models.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils,fuse_norm_utils,hadamard_utils,quant_utils,eval
from utils.process_args import process_args_ptq
from data_normalization.utils import rotate_smooth_model_inplace
from tqdm import tqdm
import train_utils.model_utils as model_utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)
from accelerate import Accelerator
log: Logger = utils.get_logger("evaluation")
import torch.distributed as dist    
import os
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard
def inspect_model(model: FSDPModule):
    assert isinstance(model, FSDPModule)

    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)    
def train() -> None:

    dist.init_process_group(backend="nccl")
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank=utils.get_local_rank()
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"我是进程 {rank}，总进程数 {world_size}，当前设备: {torch.cuda.current_device()}")
    else:
        print("分布式环境未初始化")
    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()  
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    accelerator = None 
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings=False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings=True
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=torch.bfloat16,
        token=model_args.access_token,
        cache_dir='../',
        # device_map=None, 
        # low_cpu_mem_usage=True,
        device_map="auto"
        # device_map="auto",  # 使用自动设备映射
        # max_memory={i: "6GB" for i in range(torch.cuda.device_count())}  # 限制每个GPU内存
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone().to(model.model.norm.weight.device)
        # model.lm_head.weight=model.lm_head.weight.to(model.model.norm.weight.device)
    utils.distribute_model(model)
    # model = accelerator.prepare(model)
    
    model.eval()
    fuse_norm_utils.fuse_layer_norms(model)
    model_utils.rotate_smooth_model_inplace(model,training_args.output_dir+'/model.bin',training_args)
    # accelerator.prepare(model)
    model.seqlen = training_args.model_max_length
    model.config.use_cache = False
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    # eval_utils.print_3d_act(model,testloader,'cuda',ptq_args)
    if ptq_args.rotate:
        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model,layers=[quant_utils.ActQuantWrapper])
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = ptq_args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present
    

    if ptq_args.enable_low_rank and ptq_args.w_rtn:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        for name in tqdm(qlayers, desc="adding low_rank_branch"):
            if "lm_head" in name:  # Skip lm_head quantization
                continue
            qlayers[name].add_low_rank(True,ptq_args.rank)
            svd_utils.LayerSvdCompute.approximate(qlayers[name], name, ptq_args)
    if ptq_args.w_bits < 16:
        save_dict = {}
        if ptq_args.load_qmodel_path:  # Load Quantized Rotated Model
            assert ptq_args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not ptq_args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", ptq_args.load_qmodel_path)
            save_dict = torch.load(ptq_args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not ptq_args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=ptq_args.nsamples,
                seed=ptq_args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            if ptq_args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd(
                    model,
                    "cuda",
                    ptq_args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, ptq_args,accelerator)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", ptq_args)
            save_dict["w_quantizers"] = quantizers

        if ptq_args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if ptq_args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=ptq_args.w_groupsize
                )
            torch.save(save_dict, ptq_args.save_qmodel_path)
    if ptq_args.a_bits < 16 or ptq_args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if ptq_args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, ptq_args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = ptq_args.a_bits
            layer_groupsize = ptq_args.a_groupsize
            layer_a_sym = not (ptq_args.a_asym)
            layer_a_clip = ptq_args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads
            
            if "v_proj" in name and ptq_args.v_bits < 16:  # Set the v_proj precision
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=ptq_args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (ptq_args.v_asym),
                    clip_ratio=ptq_args.v_clip_ratio,
                )

            if "o_proj" in name:
                layer_groupsize = head_dim

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision
                if ptq_args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )
   
    # Add Input Quantization
    # model = accelerator.prepare(model)
    dataset_ppl = eval_utils.evaluator(model, testloader, ptq_args,accelerator)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    eval.evaluation(model,tokenizer)
    # dist.barrier()
if __name__ == "__main__":
    train()
