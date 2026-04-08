# perf_model

GEMM-first performance modeling scaffold aligned with a SYN-PERF style flow:

1. `kernel_desc`: decompose a GEMM problem into CTA-like tasks.
2. `scheduler`: map tasks onto SMs.
3. `features`: build analytical task / SM / GPU features.
4. `dataset`: serialize features and measured latency into training samples.
5. `model`: train an MLP latency estimator.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Initial scope

- CUTLASS-style GEMM decomposition
- Round-robin SM scheduling
- Tensor Core / SIMT analytical features
- CSV dataset building
- Simple PyTorch MLP baseline

## Example commands

Build a toy processed dataset from a CSV with columns `M,N,K,latency_us`:

```bash
python scripts/build_dataset.py \
  --input data/raw/profiles/gemm_latency.csv \
  --output data/processed/train.csv \
  --gpu configs/gpu/a100.yaml \
  --kernel configs/kernels/cutlass_gemm_tensorop.yaml
```

Train the baseline MLP:

```bash
python scripts/train_mlp.py \
  --train data/processed/train.csv \
  --epochs 10 \
  --checkpoint data/models/checkpoints/mlp.pt
```
