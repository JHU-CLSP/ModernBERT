uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt numba 'torch-optimi==0.2.1' omegaconf wandb
uv pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
