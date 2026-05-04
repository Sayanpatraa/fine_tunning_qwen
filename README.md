
# Qwen & WritingPrompts — Fine-Tuning Pipeline

> **QLoRA supervised fine-tuning of Qwen2.5 on the Reddit WritingPrompts dataset,
> with train / validation / test splits and comprehensive automatic evaluation.**

---

## Dataset

| Split      | Examples   | Description                                     |
|------------|------------|-------------------------------------------------|
| Train      | ~245 k     | 90 % of total; used for gradient updates        |
| Validation | ~13.7 k    | 5 % ; used for `eval_loss` and early stopping   |
| Test       | ~13.7 k    | 5 % ; held out; only touched during evaluation  |

The dataset is the **WritingPrompts** corpus (Fan et al., 2018), scraped from Reddit's
`r/WritingPrompts` subreddit and released by Facebook AI Research.
Each row is a *(prompt, story)* pair — a short creative-writing seed plus a human-written
story of several hundred words in response.

**Source options**
- **HuggingFace Hub** (default, no credentials): `euclaise/writingprompts`
- **Kaggle** (original): `ratthachat/writing-prompts` — requires `~/.kaggle/kaggle.json`

---

## Hardware requirements

| Mode              | Min VRAM  | Recommended                              |
|-------------------|-----------|------------------------------------------|
| 7B QLoRA (4-bit)  | 16 GB     | Single A100-40GB or 2× RTX 3090          |
| 7B full fine-tune | ~80 GB    | 4× A100-80GB + DeepSpeed ZeRO-3          |
| 1.5B QLoRA (4-bit)| 8 GB      | RTX 3080 / T4 (quick testing)            |

For a fast smoke-test, use `--model Qwen/Qwen2.5-1.5B-Instruct --max_train 2000 --epochs 1`.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Optional: Flash Attention 2 (significant speedup on Ampere+)
pip install flash-attn --no-build-isolation

# 2. Prepare the data  (downloads ~4 GB from HuggingFace)
python data_prep.py

# 3. Fine-tune  (reads config.yaml)
python train.py

# 4. Evaluate the fine-tuned adapter vs the baseline
python evaluate.py                   # fine-tuned
python evaluate.py --baseline        # un-fine-tuned Qwen for comparison
```

---

## File structure

```
qwen-writing-prompts/
├── config.yaml          ← all hyperparameters (model, LoRA, training, eval)
├── requirements.txt     ← Python dependencies
├── data_prep.py         ← download + clean + split the dataset
├── train.py             ← QLoRA training with TRL SFTTrainer
├── evaluate.py          ← perplexity, ROUGE, BERTScore, diversity metrics
│
├── data/
│   └── processed/       ← HuggingFace DatasetDict (train / validation / test)
│
└── outputs/
    ├── qwen-writing-prompts/
    │   ├── checkpoint-*/        ← intermediate checkpoints
    │   ├── best_adapter/        ← best LoRA adapter weights
    │   └── merged_model/        ← merged full model (optional)
    └── results/
        ├── metrics_*.json       ← evaluation metrics
        └── samples_*.json       ← generated stories for human review
```

---

## Configuration reference (`config.yaml`)

### Model
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"   # or Qwen2.5-1.5B-Instruct / 72B-Instruct
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"  # or "eager"
```

### QLoRA
```yaml
lora:
  r: 64               # rank — higher → more parameters, more expressive
  lora_alpha: 128     # scaling factor (α / r = 2 is a common heuristic)
  lora_dropout: 0.05
  target_modules:     # all linear projections in Qwen2's attention + MLP
    - q_proj / k_proj / v_proj / o_proj
    - gate_proj / up_proj / down_proj
```

### Training
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8   # effective batch size = 32
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
```

---

## Data pipeline details (`data_prep.py`)

1. **Load** raw data from HuggingFace or Kaggle
2. **Clean** — strip Reddit markdown artefacts, collapse whitespace, remove `[WP]` tags
3. **Filter** — remove rows with empty prompts or very short stories
4. **Format** as [ChatML](https://huggingface.co/docs/transformers/chat_templating):
   ```
   <|im_start|>system
   You are a creative writer. Given a writing prompt, craft a compelling, vivid...
   <|im_end|>
   <|im_start|>user
   Writing prompt: [ WP ] You wake up to discover you have the ability to...
   <|im_end|>
   <|im_start|>assistant
   The alarm clock showed 6:47 AM when I first noticed something was wrong...
   <|im_end|>
   ```
5. **Split** deterministically (90 / 5 / 5) with `seed=42`

---

## Training details (`train.py`)

- **SFTTrainer** (TRL) with **causal language modelling** loss on full sequences
- **QLoRA**: 4-bit NF4 quantisation of base weights, LoRA adapters in bfloat16
- **Cosine LR schedule** with 5 % warmup; gradient clipping at 1.0
- **Evaluation every 500 steps** on `eval_loss`; best checkpoint automatically kept
- **After training**: LoRA weights are optionally merged into the full model for clean export

---

## Evaluation details (`evaluate.py`)

| Metric          | Description                                               |
|-----------------|-----------------------------------------------------------|
| **Perplexity**  | exp(mean NLL) on held-out test `text` sequences           |
| **ROUGE-1/2/L** | Lexical n-gram overlap with reference stories             |
| **BERTScore F1**| Semantic similarity via DeBERTa-XL token embeddings       |
| **Distinct-1/2**| Diversity of generated unigrams / bigrams                 |

Compare fine-tuned vs baseline with:
```bash
python evaluate.py --baseline                          # base Qwen
python evaluate.py --model_path ./outputs/.../merged   # merged model
```
Results are written to `outputs/results/metrics_*.json` and `samples_*.json`.

---

## Common CLI flags

```bash
# data_prep.py
python data_prep.py --source kaggle       # use Kaggle API
python data_prep.py --max_train 5000      # smaller dataset for testing
python data_prep.py --no_tokenize         # skip tokenizer (raw ChatML)

# train.py
python train.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1 --max_train 2000
python train.py --resume ./outputs/.../checkpoint-1000
python train.py --full_finetune           # disable QLoRA
python train.py --skip_merge              # don't merge weights after training

# evaluate.py
python evaluate.py --num_samples 200
python evaluate.py --model_path ./outputs/qwen-writing-prompts/merged_model
python evaluate.py --baseline
```

---

## Citation

```bibtex
@inproceedings{fan-etal-2018-hierarchical,
  title     = {Hierarchical Neural Story Generation},
  author    = {Fan, Angela and Lewis, Mike and Dauphin, Yann},
  booktitle = {Proceedings of ACL 2018},
  year      = {2018},
}
```
