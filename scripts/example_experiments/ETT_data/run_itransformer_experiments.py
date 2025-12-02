import os
import subprocess

# Parameter sweeps
seeds = list(range(15))  # 0–14
ano_types = ["input_to_output", "input_only"]
methods = ["Co_TSFA", "no_cl"]
cl_weights = [0.1]  # only used for supervised_cl

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

results_file = "result_file.csv"

# Create results CSV header (only if missing)
if not os.path.exists(results_file):
    with open(results_file, "w") as f:
        f.write(
            "method,model,ano_type,cl_weight,seed,tag,mae,mse,rmse,mape,mspe,smape\n"
        )

# Loop over parameter combinations
for seed in seeds:
    for ano_type in ano_types:
        for method in methods:

            # Only apply CL weight for supervised_cl
            weights = cl_weights if method == "Co-TSFA" else [0]

            for cl_weight in weights:

                log_file = (
                    f"logs/itransformer_{method}_{ano_type}_clw{cl_weight}_seed{seed}.log"
                )

                # Build command
                cmd = [
                    "python",
                    "-u",
                    "./run.py",
                    "--task_name", "long_term_forecast",
                    "--method", method,
                    "--continuous_ano_type", ano_type,
                    "--test_on_anomalies", "1",
                    "--is_training", "1",
                    "--root_path", "./data/",
                    "--data_path", "ETTh1.csv",
                    "--data", "ETTh1",
                    "--target", "OT",
                    "--model", "iTransformer",
                    "--model_id", f"{method}_{ano_type}_clw{cl_weight}_seed{seed}",
                    "--features", "S",
                    "--learning_rate", "0.0001",
                    "--train_epochs", "2",
                    "--batch_size", "128",
                    "--loss", "SMAPE",
                    "--enc_in", "1",
                    "--dec_in", "1",
                    "--c_out", "1",
                    "--d_model", "512",
                    "--d_ff", "2048",
                    "--n_head", "4",
                    "--seq_len", "128",
                    "--label_len", "64",
                    "--pred_len", "64",
                    "--seed", str(seed),
                    "--contrastive_weight", str(cl_weight),
                ]

                print(
                    f"Running iTransformer ETTh1: "
                    f"seed={seed}, cl_weight={cl_weight}, "
                    f"ano_type={ano_type}, method={method}"
                )

                # Run experiment and save log
                with open(log_file, "w") as lf:
                    subprocess.run(cmd, stdout=lf, stderr=lf, text=True)

                # Extract and save RESULT line
                subprocess.run(
                    f"grep 'RESULT' {log_file} >> {results_file}",
                    shell=True,
                )
