# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from dataclasses import dataclass, field
from typing import Optional, List, Union,Tuple

import argparse
import transformers


@dataclass
class ModelArguments:
    input_model: Optional[str] = field(
        default="test-input", metadata={"help": "Input model"}
    )
    output_rotation_path: Optional[str] = field(
        default="test-output", metadata={"help": "Output rotation checkpoint path"}
    )
    optimized_rotation_path: Optional[str] = field(
        default=None, metadata={"help": "Optimized rotation checkpoint path"}
    )
    access_token: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface access token to access gated repo like Llama"},
    )
    use_sdpa: bool = field(default=True, metadata=dict(help="Use SDPA instead of default MHA"))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )
    logging_steps: int = field(default=10)  # 每隔 10 步打印一次 loss
    fully_quant: bool = field(default=False, metadata=dict(help="Whether to use fully quantization"))
    train_distribute:bool = field(default=False, metadata=dict(help="Whether to use distributed training"))
    train_rotate: bool = field(default=True, metadata=dict(help="Whether to train with rotation"))
    train_smooth: bool = field(default=True, metadata=dict(help="Whether to train with rotation"))
    max_steps: int = field(default=100, metadata=dict(help="Maximum number of training steps"))
    loss_type: str = field(
        default="origin",
        metadata=dict(
            choices=[
                "act_recon",
                "origin",
                "mse",
                "kl",
                "kd",
                "feature_mse",
                "r_kl_top",
                "rkl",
                "kl_top",
                "kl_top_5",
                "kl_top_10",
                "kl_top_50",
                "kl_top_100",
                "kl_top_500",
            ],
            help="Loss type for training",
        ),
    )
    opt_type: str = field(default="RAdam", metadata=dict(choices=["SGDG", "RAdam", "RSGD"], help="Optimizer type for training"))
    rotate_lr: float = field(default=0.01609753172873217, metadata=dict(help="Learning rate for rotation"))
    smooth_lr: float = field(default=0.0017898822675917248, metadata=dict(help="Learning rate for smoothing"))
    rotate_momentom: float = field(default=0, metadata=dict(help="Momentum for rotation"))
    smooth_momentom: float = field(default=0.9, metadata=dict(help="Momentum for smoothing"))
    train_enable_wquant: bool = field(default=False, metadata=dict(help="Enable weight quantization during training"))
    train_dataset: str = field(default="wikitext2", metadata=dict(choices=["wikitext2", "c4", "pdb"], help="Dataset for training"))
    resume_path: str = field(default=None, metadata=dict(help="Path to resume training"))
    evaluation_strategy:str=field(default="no",metadata=dict(choices=["no", "epoch", "steps"]))
    per_device_train_batch_size: int = field(default=1, metadata=dict(help="Batch size per device for training"))
    per_device_eval_batch_size: int = field(default=2, metadata=dict(help="Batch size per device for evaluation"))
    save_strategy:str=field(default="no")
    rotate_ov: bool = field(default=False, metadata=dict(help="Rotate V's output and O's input"))
    rotate_pre_rope: bool = field(default=False, metadata=dict(help="Rotate ROPE's input"))
    rotate_post_rope: bool = field(default=False, metadata=dict(help="Rotate ROPE's output"))
    rotate_rope_perlayer: bool = field(default=True, metadata=dict(help="Whether to allow each layer to have a separate ROPE matrix"))
    smooth_up_down: bool = field(default=False, metadata=dict(help="Smooth Up's output and Down's output"))
    smooth_up_gate: bool = field(default=False, metadata=dict(help="Smooth x1 and x2"))
    smooth_qk: bool = field(default=False, metadata=dict(help="Smooth q and k"))
    smooth_ov: bool = field(default=False, metadata=dict(help="Smooth v and o"))
    smooth_norm_linear: bool = field(default=False, metadata=dict(help="Smooth norm and linear after rotation"))
    gradient_checkpointing: bool = field(default=True, metadata=dict(help="Whether to use gradient checkpointing"))
    rotate_down_dim: int = field(default=1, metadata=dict(help="Dimension for rotating down projection"))

    online_hadamard: str = field(
        default="down",
        metadata=dict(choices=["all", "v", "down", "None"], help="Online Hadamard transformation settings"),
    )
    online_qk_hadamard: bool = field(default=True, metadata=dict(help="Apply online Hadamard to Q/K"))
    report_to:str = field(default="none", metadata=dict(choices=["none", "wandb", "tensorboard"], help="Where to report metrics"))
    force_rdtype_inplace: bool = field(default=False,metadata=dict(help="when inplace weather use rtype if not we use fp64 to merge weights"))
    use_klt: bool = field(default=True,metadata=dict(help="whether to use klt"))


    sub_mean:bool = field(default=False,metadata=dict(help="whether to use sub mean"))
    post_attn: bool = field(default=False,metadata=dict(help="whether to use post attn for calculate kl loss"))
    # fsdp: Optional[str] = field(
    #     default="full_shard auto_wrap",
    #     metadata={
    #         "help": "Enable FSDP. For example: 'full_shard auto_wrap'"
    #     }
    # )
    
    # fsdp_transformer_layer_cls_to_wrap: Optional[str] = field(
    #     default="QuantDecoderLayer"
    # )
    # fsdp_config: Optional[Union[dict, str]] = field(
    #     default_factory=lambda: {
    #         "cpu_ram_efficient_loading": True,
            
    #         "transformer_layer_cls_to_wrap": ["QuantLinear","QuantDecoderLayer"]
    #     },
    #     metadata={
    #         "help": (
    #             "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
    #             "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
    #         )
    #     },
    # )

def parser_gen():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="Random Seed for HuggingFace and PyTorch"
    )

    # Rotation Arguments
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys""",
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=-1,
        help="Random Seed for generating random matrix!!",
    )
    parser.add_argument(
        "--fp32_had",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=4,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=4,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--qw_groupsize",
        type=int,
        default=128,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--kw_groupsize",
        type=int,
        default=128,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--vw_groupsize",
        type=int,
        default=128,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--ow_groupsize",
        type=int,
        default=128,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--upw_groupsize",
        type=int,
        default=128,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--gatew_groupsize",
        type=int,
        default=128,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--downw_groupsize",
        type=int,
        default=128,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_rtn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--w_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation""",
    )
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_pre_rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-RoPE quantization for K-cache (not Supported yet!)",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )

    # Save/Load Quantized Model Arguments
    parser.add_argument(
        "--load_qmodel_path",
        type=str,
        default=None,
        help="Load the quantized model from the specified path!",
    )
    parser.add_argument(
        "--save_qmodel_path",
        type=str,
        default=None,
        help="Save the quantized model to the specified path!",
    )
    parser.add_argument(
        "--export_to_et",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export the quantized model to executorch and save in save_qmodel_path",
    )
    #smoothquant Arguments
    parser.add_argument("--smooth",action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smooth_alpha", type=float, default=0.5)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)

    # low-rank Arguments
    parser.add_argument("--enable_low_rank",action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rank",type=int,default=32)

    # Experiments Arguments
    parser.add_argument(
        "--capture_layer_io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture the input and output of the specified decoder layer and dump into a file",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=10, help="Which decoder layer to capture"
    )
    parser.add_argument("--a_dynamic_method",type=str,
        default="pertoken",choices=["pertoken", "perchannel", "pertensor"], help="Dynamic quantization method"
    )
    args, unknown = parser.parse_known_args()

    # assert (
    #     args.a_groupsize == args.w_groupsize
    # ), "a_groupsize should be the same as w_groupsize!"
    assert args.k_pre_rope is False, "Pre-RoPE quantization is not supported yet!"

    return args, unknown


def process_args_ptq():
    ptq_args = None

    ptq_args, unknown_args = parser_gen()

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    
    model_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)

    if model_args.optimized_rotation_path is not None:
        ptq_args.optimized_rotation_path = model_args.optimized_rotation_path
    else:
        ptq_args.optimized_rotation_path = None
    ptq_args.bsz = training_args.per_device_eval_batch_size

    return model_args, training_args, ptq_args
