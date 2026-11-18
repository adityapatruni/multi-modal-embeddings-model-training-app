# mm-embed

CLIP-style multimodal (image + text) embedding training scaffold with MLflow tracking.

## Quick start
1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare data: a CSV or JSONL with columns/keys:
   - `image_path`: path to image file
   - `text`: caption or text

3. Train (single GPU):
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

4. Train (distributed):
   ```bash
   torchrun --nproc_per_node=NUM_GPUS scripts/train.py --config configs/default.yaml
   ```

5. Evaluate:
   ```bash
   python scripts/eval.py --config configs/default.yaml --checkpoint runs/exp1/best.pt
   ```

## MLflow
Default config enables MLflow. Set in `configs/default.yaml`:
- `tracking.experiment`: experiment name
- `tracking.run_name`: run name (optional)
- `tracking.tracking_uri`: remote server (optional)

## Folder structure
- `configs/`: YAML configs
- `data/`: data docs and optional local data
- `scripts/`: entrypoints
- `src/`: library code
