#!/bin/bash
# Install evaluation datasets for ViTok
# Usage: ./scripts/install_eval_datasets.sh [--data_dir /path/to/data]

set -e

DATA_DIR="${DATA_DIR:-./data}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Installing evaluation datasets to: $DATA_DIR"
mkdir -p "$DATA_DIR"

# ============================================================================
# COCO 2017 Validation Set
# ============================================================================
echo ""
echo "=== COCO 2017 Validation Set ==="
COCO_DIR="$DATA_DIR/coco"

if [ -d "$COCO_DIR/val2017" ]; then
    echo "COCO val2017 already exists, skipping..."
else
    mkdir -p "$COCO_DIR"
    echo "Downloading COCO val2017 images (~1GB)..."
    wget -q --show-progress -O "$COCO_DIR/val2017.zip" \
        http://images.cocodataset.org/zips/val2017.zip

    echo "Extracting..."
    unzip -q "$COCO_DIR/val2017.zip" -d "$COCO_DIR"
    rm "$COCO_DIR/val2017.zip"

    echo "COCO val2017 installed: $(ls "$COCO_DIR/val2017" | wc -l) images"
fi

# ============================================================================
# ImageNet Validation Set
# ============================================================================
echo ""
echo "=== ImageNet Validation Set ==="
IMAGENET_DIR="$DATA_DIR/imagenet"

if [ -d "$IMAGENET_DIR/val" ]; then
    echo "ImageNet val already exists, skipping..."
else
    echo ""
    echo "ImageNet requires manual download due to licensing."
    echo ""
    echo "To set up ImageNet validation set:"
    echo "1. Register at https://image-net.org/download-images.php"
    echo "2. Download ILSVRC2012_img_val.tar (6.3GB)"
    echo "3. Extract and organize:"
    echo ""
    echo "   mkdir -p $IMAGENET_DIR"
    echo "   tar -xf ILSVRC2012_img_val.tar -C $IMAGENET_DIR"
    echo "   cd $IMAGENET_DIR && wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh && bash valprep.sh"
    echo ""
    echo "This will organize images into class folders: val/n01440764/ILSVRC2012_val_00000293.JPEG"
fi

# ============================================================================
# FID Statistics (InceptionV3)
# ============================================================================
echo ""
echo "=== FID Reference Statistics ==="
STATS_DIR="$DATA_DIR/fid_stats"
mkdir -p "$STATS_DIR"

# Check if pytorch-fid is installed
if ! python -c "import pytorch_fid" 2>/dev/null; then
    echo "Installing pytorch-fid..."
    pip install pytorch-fid
fi

# Generate ImageNet stats if ImageNet exists
if [ -d "$IMAGENET_DIR/val" ]; then
    IMAGENET_STATS="$STATS_DIR/imagenet_val_256.npz"
    if [ -f "$IMAGENET_STATS" ]; then
        echo "ImageNet FID stats already exist, skipping..."
    else
        echo "Computing ImageNet validation FID statistics..."
        echo "(This may take 10-20 minutes)"
        python -c "
import torch
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

imagenet_dir = Path('$IMAGENET_DIR/val')
output_path = Path('$IMAGENET_STATS')

# Load inception model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device).eval()

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

# Collect all images
image_paths = list(imagenet_dir.rglob('*.JPEG')) + list(imagenet_dir.rglob('*.jpg'))
print(f'Found {len(image_paths)} images')

# Compute features
features = []
batch_size = 64
batch = []

with torch.no_grad():
    for path in tqdm(image_paths, desc='Computing features'):
        try:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            batch.append(img)

            if len(batch) >= batch_size:
                batch_tensor = torch.stack(batch).to(device)
                feat = model(batch_tensor)[0].squeeze(-1).squeeze(-1)
                features.append(feat.cpu().numpy())
                batch = []
        except Exception as e:
            print(f'Error processing {path}: {e}')
            continue

    if batch:
        batch_tensor = torch.stack(batch).to(device)
        feat = model(batch_tensor)[0].squeeze(-1).squeeze(-1)
        features.append(feat.cpu().numpy())

features = np.concatenate(features, axis=0)
mu = np.mean(features, axis=0)
sigma = np.cov(features, rowvar=False)

np.savez(output_path, mu=mu, sigma=sigma)
print(f'Saved FID stats to {output_path}')
"
    fi
fi

# Generate COCO stats
COCO_STATS="$STATS_DIR/coco_val2017_256.npz"
if [ -f "$COCO_STATS" ]; then
    echo "COCO FID stats already exist, skipping..."
else
    if [ -d "$COCO_DIR/val2017" ]; then
        echo "Computing COCO val2017 FID statistics..."
        python -c "
import torch
from pytorch_fid.inception import InceptionV3
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

coco_dir = Path('$COCO_DIR/val2017')
output_path = Path('$COCO_STATS')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image_paths = list(coco_dir.glob('*.jpg'))
print(f'Found {len(image_paths)} images')

features = []
batch_size = 64
batch = []

with torch.no_grad():
    for path in tqdm(image_paths, desc='Computing features'):
        try:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            batch.append(img)

            if len(batch) >= batch_size:
                batch_tensor = torch.stack(batch).to(device)
                feat = model(batch_tensor)[0].squeeze(-1).squeeze(-1)
                features.append(feat.cpu().numpy())
                batch = []
        except Exception as e:
            continue

    if batch:
        batch_tensor = torch.stack(batch).to(device)
        feat = model(batch_tensor)[0].squeeze(-1).squeeze(-1)
        features.append(feat.cpu().numpy())

features = np.concatenate(features, axis=0)
mu = np.mean(features, axis=0)
sigma = np.cov(features, rowvar=False)

np.savez(output_path, mu=mu, sigma=sigma)
print(f'Saved FID stats to {output_path}')
"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=== Installation Summary ==="
echo "Data directory: $DATA_DIR"
echo ""

if [ -d "$COCO_DIR/val2017" ]; then
    echo "[OK] COCO val2017: $(ls "$COCO_DIR/val2017" | wc -l) images"
else
    echo "[--] COCO val2017: Not installed"
fi

if [ -d "$IMAGENET_DIR/val" ]; then
    echo "[OK] ImageNet val: $(find "$IMAGENET_DIR/val" -name "*.JPEG" -o -name "*.jpg" 2>/dev/null | wc -l) images"
else
    echo "[--] ImageNet val: Not installed (manual download required)"
fi

if [ -f "$STATS_DIR/imagenet_val_256.npz" ]; then
    echo "[OK] ImageNet FID stats: $STATS_DIR/imagenet_val_256.npz"
else
    echo "[--] ImageNet FID stats: Not computed"
fi

if [ -f "$STATS_DIR/coco_val2017_256.npz" ]; then
    echo "[OK] COCO FID stats: $STATS_DIR/coco_val2017_256.npz"
else
    echo "[--] COCO FID stats: Not computed"
fi

echo ""
echo "Done!"
