#!/bin/bash
# OVOW Environment Installation Script - FINAL WORKING VERSION
# Tested and confirmed working on Jan 27, 2026
# Based on actual successful deployment for IDD T1 training

set -e  # Exit on error

echo "=========================================="
echo "  Installing OVOW Environment"
echo "  Final Working Configuration"
echo "=========================================="

# Step 1: Create conda environment with Python 3.11
echo "[1/13] Creating conda environment 'ovow' with Python 3.11..."
conda create -n ovow python=3.11 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ovow

# Step 2: Install PyTorch 2.1.0 with CUDA 12.1 (CRITICAL: Must be 2.1.0 for mmcv compatibility)
echo "[2/13] Installing PyTorch 2.1.0+cu121..."
echo "  NOTE: PyTorch 2.1.0 is required for pre-built mmcv wheels"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install numpy 1.26.4 (CRITICAL: Newer versions incompatible with PyTorch 2.1.0)
echo "[3/13] Installing numpy 1.26.4..."
pip install numpy==1.26.4

# Step 4: Install mmcv-lite 2.0.1 (CRITICAL: YOLO-World requires mmcv-lite, NOT mmcv)
echo "[4/13] Installing mmcv-lite 2.0.1..."
echo "  NOTE: Using mmcv-lite to avoid numpy dependency conflicts"
echo "  NOTE: mmyolo and YOLO-World require mmcv-lite>=2.0.0rc4"
pip install --no-deps mmcv-lite==2.0.1
pip install numpy==1.26.4  # Ensure numpy stays at 1.26.4 for PyTorch 2.1.0 compatibility

# Step 5: Install mmdet and mmyolo
echo "[5/13] Installing mmdet 3.3.0 and mmyolo 0.6.0..."
pip install mmdet==3.3.0 mmyolo==0.6.0

# Step 6: Install YOLO-World at specific commit (tested working)
echo "[6/13] Installing YOLO-World at commit 4d90f458..."
if [ ! -d "YOLO-World" ]; then
    echo "  ERROR: YOLO-World directory not found. Please clone it first:"
    echo "  git clone https://github.com/AILab-CVC/YOLO-World.git"
    exit 1
fi
cd YOLO-World
git checkout 4d90f458c1d0de310643b0ac2498f188c98c819c
pip install -e . --no-deps
cd ..

# Step 7: Install GCC 11 for detectron2 compilation
echo "[7/13] Installing GCC 11 from conda-forge..."
conda install -y gxx_linux-64=11.2.0 -c conda-forge

# Step 7b: Set library path immediately for detectron2
echo "[7b/13] Setting LD_LIBRARY_PATH for detectron2..."
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Step 8: Build and install detectron2 0.6 with GCC 11
echo "[8/13] Installing detectron2 0.6 from GitHub..."
echo "  NOTE: Using --no-build-isolation (confirmed working)"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Step 9: Install Pillow 9.5.0 (CRITICAL: Newer versions break detectron2)
echo "[9/13] Installing Pillow 9.5.0..."
pip install Pillow==9.5.0

# Step 10: Install transformers and tokenizers at compatible versions
echo "[10/13] Installing transformers 4.36.0 and tokenizers 0.15.2..."
pip install transformers==4.36.0 tokenizers==0.15.2

# Step 11: Install CLIP
echo "[11/13] Installing CLIP..."
pip install 'git+https://github.com/openai/CLIP.git'

# Step 12: Install remaining dependencies
echo "[12/13] Installing remaining dependencies..."
pip install supervision==0.19.0 openmim wandb pycocotools

# Step 13: Configure runtime library path for detectron2
echo "[13/13] Configuring runtime library path for detectron2..."
echo "  NOTE: Adding LD_LIBRARY_PATH to conda activation script"
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "✅ Final Working Configuration Installed:"
echo "  - PyTorch 2.1.0 + CUDA 12.1"
echo "  - numpy 1.26.4"
echo "  - mmcv 2.0.1 (pre-built with CUDA ops)"
echo "  - mmdet 3.3.0"
echo "  - mmyolo 0.6.0"
echo "  - YOLO-World (commit 4d90f458)"
echo "  - detectron2 0.6"
echo "  - Pillow 9.5.0"
echo "  - transformers 4.36.0, tokenizers 0.15.2"
echo "  - CLIP, supervision, wandb"
echo "  - LD_LIBRARY_PATH configured automatically"
echo ""
echo "To activate the environment:"
echo "  conda activate ovow"
echo ""
echo "To verify installation:"
echo "  conda deactivate && conda activate ovow  # Reload environment to apply library path"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import mmcv; print(f\"mmcv: {mmcv.__version__}\")'"
echo "  python -c 'import mmdet; print(f\"mmdet: {mmdet.__version__}\")'"
echo "  python -c 'import mmyolo; print(f\"mmyolo: {mmyolo.__version__}\")'"
echo "  python -c 'import detectron2; print(f\"detectron2: {detectron2.__version__}\")'"
echo ""
echo "To start training:"
echo "  conda activate ovow"
echo "  cd /path/to/ovow"
echo "  CUDA_VISIBLE_DEVICES=0 bash train.sh"
echo ""


