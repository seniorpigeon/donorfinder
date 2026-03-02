# FRN-based FMT Donor Matching (Inverse Problem)

This repository implements an end-to-end pipeline for FMT donor-recipient matching with:
- `Enc(x, G_frn) -> z`
- forward simulator `fr(z_pre, z_donor) -> (z_post_pred, y_pred)`
- inverse generator `fp(z_pre, z_target) -> z_d_hat`
- donor-bank retrieval by cosine similarity in embedding space.

Default demo uses synthetic data and runs fully on CPU.

## One-command run

```bash
bash scripts/run_all_synth.sh
```

If you run modules manually, use `python -m ...` form (for example `python -m src.train.train_stage_a --config configs/default.yaml`).

The pipeline performs:
1. Synthetic data generation (`R_pre, D, R_post, y` + fake `dist_mat/frc_map/tree`)
2. Stage A training (`Enc + fr`)
3. Stage B training (`fp`, freezing `Enc + fr`)
4. Stage C joint finetuning (`Enc + fr + fp`)
5. Donor embedding cache + Top-K inference

## Key outputs

- Checkpoints:
  - `outputs/checkpoints/stage_a_best.pt`
  - `outputs/checkpoints/stage_b_best.pt`
  - `outputs/checkpoints/stage_c_best.pt`
- Logs:
  - `outputs/logs/stage_a_metrics.json`
  - `outputs/logs/stage_b_metrics.json`
  - `outputs/logs/stage_c_metrics.json`
- Donor cache:
  - `outputs/donor_bank_cache/donor_bank_embeddings.npz`
- Inference results:
  - `outputs/logs/topk_results.csv`
  - `outputs/logs/topk_z_post_pred.npy`
  - `outputs/logs/z_d_hat.npy`

## Graph modes

Configured in `configs/default.yaml`:
- `graph.graph_mode: frc` (default)
  - aggregate species abundance to FRC
  - graph edges from FRC tree (`edges` or `newick`)
- `graph.graph_mode: species`
  - species kNN graph from `dist_mat`
  - edge weight `exp(-dist/sigma)`

## Real data interface (placeholder)

Implemented in `src/data/preprocess.py` and `src/data/dataset_fmt.py`:
- load abundance table + metadata
- align taxa and apply transform (`log1p` or `clr`)
- build `(R_pre, D, R_post, y)` dataset

You can replace synthetic files in `data/synthetic/` with real files and adjust paths in `configs/default.yaml`.

## Install

```bash
pip install -r requirements.txt
```
