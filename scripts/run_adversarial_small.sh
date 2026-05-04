#!/usr/bin/env bash
set -e

python scripts/prepare_dataset.py \
  --config configs/default_config.yaml \
  --max_train_samples 5000 \
  --max_eval_samples 500 \
  --max_test_samples 500

python scripts/train_qlora.py \
  --config configs/default_config.yaml \
  --max_train_samples 5000 \
  --max_eval_samples 500

python scripts/adversarial_self_optimize.py \
  --config configs/default_config.yaml \
  --start_adapter_path models/writing_prompts_lora \
  --iterations 2 \
  --prompts_per_iteration 32 \
  --candidates_per_prompt 3
