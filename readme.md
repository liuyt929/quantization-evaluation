conda create -n lyt-eval python=3.9
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install lm_eval
pip install fast-hadamard-transform