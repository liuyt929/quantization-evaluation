import subprocess


command=[
    "python optimize.py --input_model meta-llama/Llama-3.1-70B --model_max_length 2048 --bf16 True --w_bits 4 --a_bits 4  --rotate_ov=True --rotate_post_rope=False --online_qk_hadamard=False --output_dir='./70B-rotate+scale' --smooth_ov=True --a_asym --smooth_up_down=True --smooth_norm_linear=True --train_distribute=True --rank=32  --rotate --enable_low_rank --use_klt=False",
    "python test_original_model.py --input_model meta-llama/Llama-3.1-70B --model_max_length 2048 --bf16 True"
]
for i, cmd in enumerate(command):
    log_file = f"log-70b-{i}.txt"  # 生成不同的日志文件
    with open(log_file, "w") as f:
        print(f"Running: {cmd}")
        process = subprocess.run(cmd, shell=True, stdout=f, stderr=f, text=True)
