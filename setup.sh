cat > /workspace/FraudGT-main/setup.sh <<'SH'
#!/bin/bash
set -e

cd /workspace/FraudGT-main

echo "=== Create venv outside repo ==="
mkdir -p /workspace/venvs
python3.9 -m venv /workspace/venvs/fraudgt
source /workspace/venvs/fraudgt/bin/activate

echo "=== Upgrade pip tools ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== Clean old installs ==="
python -m pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib torch-geometric || true
python -m pip cache purge || true

echo "=== Install PyTorch 2.3.1 CUDA 11.8 ==="
python -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu118

echo "=== Install PyG matching torch 2.3.1 + cu118 ==="
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.3.1+cu118.html

python -m pip install torch-geometric

echo "=== Install FraudGT dependencies ==="
python -m pip install \
  yacs \
  datatable \
  pandas \
  wandb \
  torchmetrics \
  ogb \
  gdown==5.2.0 \
  matplotlib \
  scikit-learn \
  numpy \
  tqdm \
  tensorboardX \
  kagglehub

echo "=== Final dependency check ==="
python - <<'PY'
import torch
import torch_geometric
import torch_sparse
import torch_scatter
import yacs
import datatable
import wandb
import torchmetrics
import ogb
import gdown
import matplotlib

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("PyG OK")
print("All dependencies OK")
PY

echo "=== Setup complete ==="
echo "To activate later:"
echo "cd /workspace/FraudGT-main"
echo "source /workspace/venvs/fraudgt/bin/activate"
SH

chmod +x /workspace/FraudGT-main/setup.sh