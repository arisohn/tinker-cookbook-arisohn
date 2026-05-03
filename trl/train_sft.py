"""SFT fine-tune Qwen3-8B-Base on OpenThoughts3-1.2M-shuffle-1k via HuggingFace TRL.

- Full fine-tuning (no PEFT/LoRA), bf16, FlashAttention-2.
- assistant_only_loss=True with a custom Qwen3-compatible chat template that
  marks assistant tokens via {% generation %} ... {% endgeneration %} blocks.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_PATH = "/workspace/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
DATA_PATH = "/workspace/datasets--open-thoughts--OpenThoughts3-1.2M-shuffle-1k/data/train-00000-of-00001.parquet"
OUTPUT_DIR = "/workspace/trl/outputs/qwen3-8b-openthoughts-sft"

# Qwen3 chat-ml format with {% generation %} markers so TRL's
# assistant_only_loss=True can produce the assistant token mask.
ASSISTANT_LOSS_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>\\n' }}"
    "{%- endgeneration %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def load_messages_dataset() -> Dataset:
    df = pd.read_parquet(DATA_PATH)
    role_map = {"human": "user", "gpt": "assistant"}
    msgs = [
        [{"role": role_map[t["from"]], "content": t["value"]} for t in conv]
        for conv in df["conversations"]
    ]
    return Dataset.from_dict({"messages": msgs})


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.chat_template = ASSISTANT_LOSS_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    train_ds = load_messages_dataset()
    print(f"[data] num_examples={len(train_ds)}", flush=True)

    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        run_name="qwen3-8b-openthoughts-sft",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        max_length=8192,
        packing=False,
        assistant_only_loss=True,
        completion_only_loss=False,
        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=True,
        logging_steps=5,
        logging_first_step=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join(OUTPUT_DIR, "tb"),
        seed=42,
        data_seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        use_liger_kernel=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[done] training finished, final model saved to:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    sys.exit(main())
