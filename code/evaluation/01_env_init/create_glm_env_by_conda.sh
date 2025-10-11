# 更新 conda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --set show_channel_urls yes

# 1. Create a new, clean environment. We'll call it "glm_env".
conda create -n ms-swift-qwen2_5 python=3.10 -y

# 2. Activate the new environment. Your prompt will change to (glm_env).
conda activate ms-swift-qwen2_5

# 3. Install PyTorch with CUDA support needed for your H100 GPU.
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# 设置国内清华 apt 源
cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list


pip install 'ms-swift[all]' -U
pip install "sglang[all]<0.5" -U
pip install "vllm>=0.5.1" "transformers<4.56" "trl<0.21" -U
pip install "lmdeploy>=0.5" -U
pip install autoawq -U --no-deps
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install git+https://github.com/modelscope/ms-swift.git
pip install timm -U
pip install "deepspeed" -U
pip install qwen_vl_utils qwen_omni_utils decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy -U
pip install flash_attn -U


# 5. NOW, from within this clean environment, run your script.
# python validate_complete.py