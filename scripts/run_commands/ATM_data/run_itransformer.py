#!/bin/bash

python -u ./run.py \
  --task_name long_term_forecast \
  --method Co_TSFA \
  --is_training 1 \
  --root_path ./data/ \
  --data_path atm_data.csv \
  --data atm \
  --target dispensed_amount \
  --model iTransformer \
  --test_on_anomalies 1 \
  --model_id run_Co_TSFA \
  --continuous_ano_type input_output \
  --contrastive_weight 1 \
  --features S \
  --learning_rate 0.00001 \
  --train_epochs 2 \
  --batch_size 128 \
  --loss SMAPE \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --d_ff 2048 \
  --n_head 4 \
  --seq_len 128 \
  --label_len 64 \
  --pred_len 64 \
  --seed 0
