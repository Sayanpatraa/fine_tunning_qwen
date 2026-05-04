# Writing Prompts LLM Fine-Tuning

This repository fine-tunes an open Hugging Face causal language model on the Kaggle Writing Prompts dataset.

It supports:

- Kaggle dataset download
- Prompt-story dataset preparation
- QLoRA fine-tuning
- Dynamic hyperparameters through YAML and CLI overrides
- Testing / story generation
- Streamlit UI for training, testing, and parameter control

---

## 1. Create environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2. Download Kaggle dataset

Create your Kaggle token:

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Then run:

```bash
kaggle datasets download -d ratthachat/writing-prompts -p data/raw --unzip
```

Expected files include:

```text
train.wp_source
train.wp_target
valid.wp_source
valid.wp_target
test.wp_source
test.wp_target
```

---

## 3. Prepare dataset

```bash
python scripts/prepare_dataset.py --config configs/default_config.yaml
```

---

## 4. Train model

Basic run:

```bash
python scripts/train_qlora.py --config configs/default_config.yaml
```

Override parameters dynamically:

```bash
python scripts/train_qlora.py \
  --config configs/default_config.yaml \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --output_dir models/qwen_3b_writing_lora \
  --num_train_epochs 2 \
  --learning_rate 0.00015 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_seq_length 2048 \
  --max_train_samples 5000
```

For an A10G 24GB GPU, you can try:

```bash
python scripts/train_qlora.py \
  --config configs/default_config.yaml \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_dir models/qwen_7b_writing_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048
```

---

## 5. Test generation

```bash
python scripts/test_generate.py \
  --config configs/default_config.yaml \
  --adapter_path models/writing_prompts_lora \
  --prompt "A lonely astronaut discovers that the moon has been writing letters to Earth."
```

---

## 6. Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI lets you change fine-tuning parameters and launch dataset prep, training, and generation from one place.

---

## Recommended training strategy

Start with a small run first:

```bash
python scripts/train_qlora.py \
  --config configs/default_config.yaml \
  --max_train_samples 5000 \
  --max_eval_samples 500
```

Then scale once the pipeline works.

Good starting models:

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
mistralai/Mistral-7B-Instruct-v0.3
meta-llama/Llama-3.1-8B-Instruct
```

Some gated models, such as Meta Llama models, require accepting the model license on Hugging Face and logging in:

```bash
huggingface-cli login
```

---

## Notes

If `nvidia-smi` is not available, you are probably not on a usable GPU machine or the NVIDIA driver is missing.

For story generation, do not overtrain. One epoch is often enough for this dataset.


---

# Adversarial Self-Optimization Mode

This repo also includes a self-improving adversarial loop.

It is not a classic GAN. For language models, a practical adversarial setup is:

```text
Generator LLM
    ↓
Generate multiple candidate stories
    ↓
Critic scores each candidate
    ↓
Best story = chosen, worst story = rejected
    ↓
DPO trains the model to prefer chosen over rejected
    ↓
Repeat for several iterations
```

This uses DPO because TRL expects preference data with:

```text
prompt
chosen
rejected
```

The official TRL DPOTrainer documentation states that each training example should contain a prompt plus a preferred `chosen` completion and a dispreferred `rejected` completion.

## Run adversarial self-optimization

First do normal SFT:

```bash
python scripts/prepare_dataset.py --config configs/default_config.yaml

python scripts/train_qlora.py \
  --config configs/default_config.yaml \
  --max_train_samples 5000
```

Then run adversarial self-optimization:

```bash
python scripts/adversarial_self_optimize.py \
  --config configs/default_config.yaml \
  --start_adapter_path models/writing_prompts_lora \
  --iterations 3 \
  --prompts_per_iteration 128 \
  --candidates_per_prompt 4
```

Each iteration saves:

```text
models/adversarial_runs/iteration_01/scored_candidates.csv
models/adversarial_runs/iteration_01/preference_pairs.csv
models/adversarial_runs/iteration_01/iteration_metrics.json
models/adversarial_runs/iteration_01/dpo_adapter/
```

The final adapter path is saved in:

```text
models/adversarial_runs/FINAL_ADAPTER_PATH.txt
```

## What the critic scores

The default critic is heuristic and cheap. It scores:

- prompt relevance
- lexical diversity
- repetition control
- story length
- narrative shape

You can replace `score_story()` in:

```text
src/adversarial_utils.py
```

with a stronger reward model or an LLM judge later.

## Important warning

This loop can optimize toward the critic's weaknesses. That is the central failure mode.

Do not trust critic score alone. Always inspect generated stories manually and keep a held-out prompt set for qualitative testing.


---

# Fully Integrated Streamlit UI

The Streamlit UI now calls Python pipeline functions directly instead of launching CLI scripts.

Run:

```bash
streamlit run streamlit_app.py
```

The UI includes:

```text
Prepare Dataset
Train SFT
Adversarial Self-Optimization
Generate / Test
Metrics
Config Preview
```

## Streamlit cache behavior

The app uses:

```python
@st.cache_data
```

for:

```text
default config loading
processed dataset summaries
adversarial metrics tables
```

and:

```python
@st.cache_resource
```

for:

```text
loaded generation model + tokenizer
```

This means generation is much faster after the first load.

If you train a new adapter or change model paths, click:

```text
Clear Streamlit cache
```

from the sidebar before generating with the new model.

The training functions clear model cache automatically after SFT or adversarial DPO completes.
