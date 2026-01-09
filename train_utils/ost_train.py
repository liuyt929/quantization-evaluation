import transformers, torch, os, datasets, random,utils
import torch.nn.functional as F, torch, torch.nn as nn
import utils.data_utils as data_utils
import geoopt
from train_utils.model_utils import LM
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    default_data_collator,
)
from accelerate.hooks import remove_hook_from_module
from utils.data_utils import CustomJsonDataset, group_texts
from datasets import Dataset, IterableDataset
from train_utils.cayley_opt import SGDG
from torch.optim import lr_scheduler
from accelerate import DistributedType
from train_utils.trainer import MyTrainer
from loguru import logger
from utils.utils import distribute_model


def rotate_smooth_train(args, lm: LM,ptq_args,model_args):

    logger.info("train rotate model")
    if args.smooth_up_down:
        logger.info("train smooth up down")
    if args.smooth_up_gate:
        logger.info("train smooth up gate")
    if args.smooth_qk:
        logger.info("train smooth qk")
    if args.smooth_ov:
        logger.info("train smooth ov")
    if args.smooth_norm_linear:
        logger.info("smooth norm linear")

    lm.model.config.use_cache = False

    train_dataset, eval_dataset = get_train_eval_dataset(args, lm.tokenizer,ptq_args,model_args)
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    param_keys = get_param_keys(lm.model)
    print(param_keys)
    utils.utils.cleanup_memory()
    if args.train_distribute:
        distribute_model(lm.model)
    args.remove_unused_columns=False
    train_dataset = manual_cleanup_dataset(train_dataset, lm.model)
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    import torch.distributed as dist    
    if dist.is_initialized() and dist.get_world_size() > 1:
        print(f"--- 正在同步 Rank 0 的旋转矩阵到所有显卡 ---")
        
        # 强制所有卡停在这里，防止有的卡跑得太快
        dist.barrier()
        
        with torch.no_grad():
            for m in lm.model.modules():
                # 找到你那些被忽略的模块
                from data_normalization import smooth_utils,rotation_utils
                if isinstance(m, (rotation_utils.RotateModule, smooth_utils.SmoothModule)):
                    for param in m.parameters():
                        dist.broadcast(param.data, src=0)
        print(lm.model.R_res.weight)
        # # 同步完成，大家一起出发
        dist.barrier()
        print(f"--- 同步完成，开始训练 ---")
    # args.fsdp = "full_shard"
    trainer = MyTrainer(
        model=lm.model,
        tokenizer=lm.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )  
    def skip_prepare(*args, **kwargs):
        if len(args) == 1: return args[0]
        return args

    trainer.accelerator.prepare = skip_prepare
    trainer.model_wrapped = trainer.model
    
    print(f"Is enabled in trainer: {trainer.args.gradient_checkpointing}")
    trainer.train()
    save_ignored_params_fsdp1(lm.model, f"{args.output_dir}/model.bin",param_keys)
    if args.train_distribute:
        remove_hook_from_module(lm.model)
    return lm
from torch.distributed.tensor import DTensor, Replicate
def save_fsdp2_model(model, save_path,param_keys):
    # 只有 Rank 0 负责写入文件
    state_dict = {}
    
    for name, param in model.named_parameters():
        # 如果是 DTensor，需要将其还原（Redistribute 为 Replicate 模式）
        if name in param_keys:
            if isinstance(param, DTensor):
                full_param = param.redistribute(param.device_mesh, [Replicate()]).to_local()
                state_dict[name] = full_param.cpu() # 移到 CPU 释放显存
            else:
                state_dict[name] = param.cpu()

    if torch.distributed.get_rank() == 0:
        torch.save(state_dict, save_path)
        print(f"模型已成功保存至: {save_path}")
def manual_cleanup_dataset(dataset, model):
    import inspect
    # 获取模型 forward 的参数列表
    # 如果 model 是 FSDP 包装过的，尝试访问 model._fsdp_wrapped_module.forward
    target_module = model._fsdp_wrapped_module if hasattr(model, "_fsdp_wrapped_module") else model
    signature = inspect.signature(target_module.forward)
    _signature_columns = list(signature.parameters.keys())
    
    # 额外保留 labels，因为它是 Trainer 默认需要的
    _signature_columns += ["labels", "label", "label_ids"]
    
    columns_to_remove = [col for col in dataset.column_names if col not in _signature_columns]
    return dataset.remove_columns(columns_to_remove)
def get_train_eval_dataset(args, tokenizer,ptq_args,model_args):
    cache_dir = "./cache/" + model_args.input_model.split("/")[-1] + "_".join(["tokenized", args.train_dataset])
    
    if os.path.exists(cache_dir):
        tokenized_datasets = datasets.load_from_disk(cache_dir)
    else:
        if args.train_dataset == "wikitext2":
            train_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

        def tokenize_function(examples):
            return tokenizer(examples["text"])

        tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
        )
        grouped_datasets = group_texts(2048, tokenized_datasets)
        tokenized_datasets = Dataset.from_dict(grouped_datasets)
        tokenized_datasets.save_to_disk(cache_dir)
    
    train_size = len(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.select(range(train_size // 2))

    test_loader = data_utils.get_wikitext2(
        "wikitext2", seed=ptq_args.seed, model=model_args.input_model, seqlen=2048, eval_mode=True
    )
    nsample = test_loader["input_ids"].numel() // 2048
    input_ids = test_loader["input_ids"].reshape(-1)[: nsample * 2048]
    eval_dataset = Dataset.from_dict(dict(input_ids=input_ids.split(2048, dim=-1)))

    def f(examples):
        examples["labels"] = examples["input_ids"]
        return examples

    eval_dataset = eval_dataset.map(f)
    return tokenized_datasets, eval_dataset

def save_ignored_params_fsdp1(model, save_path, param_keys):
    state_dict = {}
    
    # 直接遍历，因为是 ignored 的，所以 param.data 就是完整的
    for name, param in model.named_parameters():
        if name in param_keys:
            # 直接去掉那层讨厌的前缀存入字典
            clean_name = name.replace("_fsdp_wrapped_module.", "")
            state_dict[clean_name] = param.cpu().clone()

    # 只有 Rank 0 负责写入文件
    if torch.distributed.get_rank() == 0:
        torch.save(state_dict, save_path)
        print(f"Ignored parameters 已成功保存至: {save_path}")
def get_param_keys(model):
    keys = list()
    for k, v in model.named_parameters():
        if v.requires_grad:
            keys.append(k)
    return keys
