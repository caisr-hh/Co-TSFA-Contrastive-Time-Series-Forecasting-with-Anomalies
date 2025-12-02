import subprocess
import os
import csv
import re
import numpy as np
import pandas as pd
from itertools import product

# ---------------- Config ----------------
results_file = "result_file.csv"
os.makedirs("logs", exist_ok=True)

# Load already completed combinations
if os.path.exists(results_file):
    done_df = pd.read_csv(results_file)
    done_tags = set(
        f"train{str(r['train_cat'])}_test{str(r['test_cat'])}_"
        f"cont{str(r['continuous_ano_type']) if pd.notna(r['continuous_ano_type']) else '-'}_"
        f"ptype{str(r['pointwise_ano_type']) if pd.notna(r['pointwise_ano_type']) else '-'}_"
        f"pr{str(r['pointwise_ratio']) if pd.notna(r['pointwise_ratio']) else '-'}_"
        f"seed{seed}"
        for _, r in done_df.iterrows()
        for seed in map(int, r['seeds'].split(";"))
    )
else:
    done_tags = set()

seeds = [0, 1, 2]

# anomaly categories considered in train/test
train_categories = ["none", "continuous", "pointwise"]
test_categories  = ["none", "continuous", "pointwise"]

# hyper-parameters (unchanged)
continuous_ano_types = ["input_only", "input_to_output"]
pointwise_ano_types = ["const", "missing", "gaussian"]
pointwise_ratios = [0.1, 0.2, 0.3]
pointwise_scale_const = 0.5
pointwise_scale_gaussian = 2

# Base TimesNet config (unchanged)
BASE_CMD = [
    "python",
    "-u", "./run.py",
    "--task_name", "long_term_forecast",
    "--method", "Co_TSFA",
    "--is_training", "1",
    "--root_path", "./data",
    "--data_path", "electricity.csv",
    "--data", "electricity",
    "--target", "OT",
    "--model", "TimesNet",
    "--model_id", "run_Co_TSFA",
    "--features", "S",
    "--learning_rate", "0.001",
    "--train_epochs", "10",
    "--batch_size", "128",
    "--loss", "SMAPE",
    "--enc_in", "1",
    "--dec_in", "1",
    "--c_out", "1",
    "--d_model", "64",
    "--d_ff", "128",
    "--n_head", "4",
    "--seq_len", "16",
    "--label_len", "1",
    "--pred_len", "1",
]

# ---------------- Helpers ----------------
def extract_float(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def build_cmd(seed,
              train_cat,
              test_cat,
              continuous_ano_type=None,
              pointwise_ano_type=None,
              pointwise_ratio=None):
    """Build a command for one run. All parameters preserved exactly."""
    cmd = BASE_CMD + [
        "--seed", str(seed),
        "--ano_category_in_train", train_cat,
        "--ano_category_in_test", test_cat,
    ]

    if (train_cat == "continuous") or (test_cat == "continuous"):
        c = continuous_ano_type if continuous_ano_type else continuous_ano_types[0]
        cmd += ["--continuous_ano_type", c]

    if (train_cat == "pointwise") or (test_cat == "pointwise"):
        ptype = pointwise_ano_type or pointwise_ano_types[0]
        pr = pointwise_ratio if pointwise_ratio is not None else pointwise_ratios[0]
        cmd += [
            "--pointwise_ano_type", ptype,
            "--pointwise_ano_ratio", str(pr),
            "--pointwise_ano_scale_const", str(pointwise_scale_const),
            "--pointwise_ano_scale_gaussian", str(pointwise_scale_gaussian),
        ]

    return cmd

# Create results CSV header if missing
if not os.path.exists(results_file):
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "train_cat", "test_cat",
            "continuous_ano_type",
            "pointwise_ano_type",
            "pointwise_ratio",
            "avg_test_mae", "avg_test_mse",
            "seeds"
        ])

# ------------- Sweep Logic --------------
for train_cat, test_cat in product(train_categories, test_categories):

    sweep_continuous = (train_cat == "continuous") or (test_cat == "continuous")
    sweep_pointwise = (train_cat == "pointwise") or (test_cat == "pointwise")

    if sweep_continuous and sweep_pointwise:
        param_grid = list(product(continuous_ano_types, pointwise_ano_types, pointwise_ratios))
    elif sweep_continuous:
        param_grid = [(c, None, None) for c in continuous_ano_types]
    elif sweep_pointwise:
        param_grid = [(None, p, r) for p in pointwise_ano_types for r in pointwise_ratios]
    else:
        param_grid = [(None, None, None)]

    for cont_type, ptype, pratio in param_grid:

        mae_list, mse_list = [], []

        tag_parts = [
            f"train{train_cat}",
            f"test{test_cat}",
            f"cont{cont_type if cont_type else '-'}",
            f"ptype{ptype if ptype else '-'}",
            f"pr{pratio if pratio is not None else '-'}"
        ]
        base_tag = "_".join(tag_parts)

        for seed in seeds:

            log_tag = f"{base_tag}_seed{seed}"
            log_path = f"logs/{log_tag}.log"

            if log_tag in done_tags:
                print(f"Skipping already completed: {log_tag}")
                continue

            print(f"Running: {log_tag}")

            cmd = build_cmd(seed, train_cat, test_cat,
                            continuous_ano_type=cont_type,
                            pointwise_ano_type=ptype,
                            pointwise_ratio=pratio)

            with open(log_path, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

            with open(log_path, "r") as f:
                log_text = f.read()

            mse = extract_float(r"mse:(\d+\.\d+)", log_text)
            mae = extract_float(r"mae:(\d+\.\d+)", log_text)

            if mae is None or mse is None:
                print(f"Failed to parse metrics for {log_tag}")
                continue

            mae_list.append(mae)
            mse_list.append(mse)
            print(f"{log_tag}  |  MSE: {mse:.6g}  MAE: {mae:.6g}")

        if len(mae_list) == len(seeds):
            avg_mae = float(np.mean(mae_list))
            avg_mse = float(np.mean(mse_list))
            print(f"{base_tag} | averaged over {len(seeds)} seeds | "
                  f"MSE: {avg_mse:.6g} MAE: {avg_mae:.6g}")

            with open(results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    train_cat, test_cat,
                    cont_type or "",
                    ptype or "",
                    pratio if pratio is not None else "",
                    avg_mae, avg_mse,
                    ";".join(map(str, seeds)),
                ])
        else:
            print(f"Incomplete seeds for {base_tag}")
