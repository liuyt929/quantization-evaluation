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
from utils import data_utils, eval_utils, utils,fuse_norm_utils,hadamard_utils,quant_utils,eval
# from models.modeling_llama import LlamaForCausalLM
from utils.process_args import process_args_ptq
from accelerate import Accelerator
log: Logger = utils.get_logger("evaluation")
    
def train() -> None:
    model_args, training_args, ptq_args = process_args_ptq()
    transformers.set_seed(ptq_args.seed)

    
    # load_model        
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    ) 
    accelerator = Accelerator()
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
        device_map="auto",  # 使用自动设备映射
        )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone().to(model.model.norm.weight.device)
    model.eval()

    model.seqlen = training_args.model_max_length
    model.config.use_cache = False
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(model, testloader, ptq_args,accelerator)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    eval.evaluation(model,tokenizer)

if __name__ == "__main__":
    train()
