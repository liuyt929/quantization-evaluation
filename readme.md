# 环境配置
conda create -n quantization-eval python=3.9
conda activate quantization-eval

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install lm_eval
pip install fast-hadamard-transform

# 若需要下载Llama 3.1 70B，则huggingface 登录
huggingface-cli login
token: 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

# 运行
cd quantization-evaluation
python script.py
输出将重定向在log-70b-0.txt中


