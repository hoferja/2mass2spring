# Project 2: MLP vs PINN vs FNO on a 2-DoF Mass-Spring-Damper

Trajectory-level state estimation of a robot mass in series with a passive
compliant base, excited by a chirp on the actuated mass. See
`project2_concept.txt` for the locked design.

## Layout

```
2mass2spring/
    constants.py             # physical & sampling constants, N_VALUES, STRATA
    utils.py                 # dataset I/O, normalizers, metrics, timing
    data/
        generate.py          # ODE sim, chirp gen, --noise_std flag
    models/
        mlp.py
        pinn.py              # forward net + chirp + ode_residual
        fno.py               # 1D FNO + build_fno_input
    train.py                 # --model {mlp,fno,pinn}, single file
    eval.py                  # auto-detects model from checkpoint
    run_sweep_mlp.py
    run_sweep_fno.py
    run_sweep_pinn.py        # lambda search + final sweep
    analyze.py               # comparison table + plots (incl. loss curves)
    configs/
    results/plots/
    checkpoints/
```

## Run order

```bash
# 1. Generate data (two .npz files: train pool + held-out test).
python data/generate.py --n_samples 5000 --seed 0 --out data/dataset_train.npz
python data/generate.py --n_samples 1000 --seed 1 --out data/dataset_test.npz

# 2. Three sweeps. Each writes results/metrics_<model>.json incrementally.
python run_sweep_mlp.py
python run_sweep_fno.py
python run_sweep_pinn.py            # lambda search at N=1000, then sweep

# 3. Comparison artifacts.
python analyze.py --results_dir results
```

** PLEASE NOTE: due to a bug, CUDA support does not yet work, please add the flag `--device "cpu"` after the python file name in stage 2.

## Unified metrics schema

Every sweep writes `results/metrics_<model>.json` with:

```
{
  "model": "mlp" | "pinn" | "fno",
  "lambda": float,                    # pinn only
  "by_n": {
    "<N>": {
      "N": int,
      "param_count": int,
      "train_wall_clock_s": float,
      "peak_gpu_memory_mb": float,
      "epochs_run": int,
      "best_val_loss": float,
      "inference_time_per_traj_ms": float,
      "flops_per_forward": int | null,
      "overall":     {"rmse_traj", "rmse_peak", "bias_peak", "count"},
      "per_stratum": {"r_0.0_1.0": {...}, "r_1.0_1.2": {...},
                      "r_1.2_1.5": {...}, "r_gt_1.5":  {...}},
      "overlay_sample": {"index", "amplification_ratio",
                          "t", "pred_sum", "true_sum"},
      "history": [{"epoch", ...per-epoch losses..., "lr"}, ...]
    }
  }
}
```

- MLP / FNO `history` keys: `train_loss`, `val_loss`, `lr`.
- PINN `history` keys: `train_data`, `train_res`, `train_total`, `val_rmse`,
  `val_peak_bias`, `lr`.

`analyze.py` reads `history` to produce per-model loss curves
(`results/plots/loss_curves_<model>.png`) and, for PINN, the data-vs-residual
breakdown (`results/plots/pinn_residual_curves.png`).
