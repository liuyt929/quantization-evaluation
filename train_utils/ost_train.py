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
    print(f"REAL_NAME: {lm.model.model.layers[0].__class__.__name__}")
    print(lm.model)
    trainer = MyTrainer(
        model=lm.model,
        tokenizer=lm.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )  
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
                        # 把 Rank 0 的初始值分发给所有人，保证“起跑线”绝对一致
                        dist.broadcast(param.data, src=0)
        
        # # 同步完成，大家一起出发
        dist.barrier()
        print(f"--- 同步完成，开始训练 ---")
    # st = torch.load(f"{args.output_dir}/model.bin", map_location="cpu")
    # model_state_dict = trainer.model.state_dict()
    # model_state_dict.update(st)
    # trainer.model.load_state_dict(model_state_dict, strict=False)
    trainer.train()
    acc = trainer.accelerator
    st = {k: v for k, v in (acc.get_state_dict(trainer.model)).items() if k in param_keys}
    acc.wait_for_everyone()
    if acc.is_main_process:
        torch.save(st, f"{args.output_dir}/model.bin")
    else:
        print(f"sub process{acc.process_index} exit")
        exit(0)
    lm.model = acc.unwrap_model(trainer.model)
    if args.train_distribute:
        remove_hook_from_module(lm.model)
    return lm


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


def get_param_keys(model):
    keys = list()
    for k, v in model.named_parameters():
        if v.requires_grad:
            keys.append(k)
    return keys
