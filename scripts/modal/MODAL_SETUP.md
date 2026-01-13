# Modal Setup for ViTok

Run ViTok evaluation on cloud GPUs without needing a local GPU.

## Prerequisites

1. **Install Modal:**
   ```bash
   pip install modal
   ```

2. **Authenticate:**
   ```bash
   modal token new
   ```

3. **Create HuggingFace secret** (for model downloads):
   ```bash
   modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
   ```

## Usage

### Method 1: --modal flag (recommended)

Run from the repo root:
```bash
# With dataset preset
python scripts/eval_vae.py --modal --model L-64 --dataset coco-val

# With all options
python scripts/eval_vae.py --modal --model L-64 --dataset imagenet-val \
    --max-size 512 --batch-size 32 --num-samples 5000
```

### Method 2: Direct Modal CLI

```bash
modal run scripts/modal/run_eval.py --model L-64 --dataset coco-val
```

## Dataset Presets

| Preset | Path | Description |
|--------|------|-------------|
| `coco-val` | `/data/coco/val2017` | COCO 2017 validation (5K images) |
| `imagenet-val` | `/data/imagenet/val` | ImageNet validation (50K images) |
| `div8k` | `/data/div8k/val` | DIV8K validation set |

COCO is auto-downloaded on first run. For other datasets, cache them first:
```bash
modal run scripts/modal/setup_data.py
```

## Architecture

All Modal configuration lives in `scripts/modal/`:

```
scripts/modal/
├── modal_config.py    # Shared config (images, volumes, GPU presets)
├── run_eval.py        # Evaluation runner (called by --modal flag)
├── setup_data.py      # Dataset caching utility
└── MODAL_SETUP.md     # This file
```

### modal_config.py exports:

- **GPU configs:** `EVAL_CONFIG`, `TRAINING_CONFIG`, `INFERENCE_CONFIG`
- **Images:** `base_image`, `eval_image`
- **Helpers:** `with_vitok_code()` - adds vitok package to images
- **Paths:** `DATASET_PATHS` - preset dataset locations

### Adding a new Modal script

```python
from scripts.modal.modal_config import EVAL_CONFIG, eval_image, with_vitok_code

app = modal.App("my-app")
image = with_vitok_code(eval_image, ["scripts/my_script.py"])

@app.function(image=image, **EVAL_CONFIG)
def my_function():
    ...
```

## Do Not Modify

**eval_vae.py should not contain Modal-specific code** beyond the simple import:
```python
if args.modal:
    from scripts.modal.run_eval import run_eval_on_modal
    stats = run_eval_on_modal(args)
```

All Modal logic belongs in `scripts/modal/`.
