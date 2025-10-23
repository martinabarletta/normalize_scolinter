# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 21:05:45 2025

@author: matil
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback, TrainerState, TrainerControl
)
from peft import LoraConfig, get_peft_model

# --- GPU check ---
if not torch.cuda.is_available():
    print("Aucune GPU détectée. L'entraînement sera très lent.")
else:
    print("GPU détecté :", torch.cuda.get_device_name(0))

# --- Carica dataset JSONL ---
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/scolinter_ita/train.jsonl",
        "validation": "data/scolinter_ita/validation.jsonl"
    }
)

# --- Modello e tokenizer ---
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

# --- Configurazione LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Preprocess dataset ---
def preprocess(example):
    inputs = [f"Normalizza il testo seguente: {x}" for x in example["input"]]
    targets = example["output"]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids

    # Maschera pad token
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

# --- DEBUG labels ---
example = tokenized["train"][0]
print("Input decodificato:")
print(tokenizer.decode(example["input_ids"], skip_special_tokens=True))

valid_labels = [l for l in example["labels"] if l != -100]
print(f"Label validi: {len(valid_labels)} / {len(example['labels'])}")
print("Label decodificato:")
print(tokenizer.decode([x for x in example["labels"] if x != -100]))

# --- Data collator ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- Callback grad_norm ---
class GradNormCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            model = kwargs["model"]
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Grad_norm: {total_norm:.4f}")

# --- Parametri di training ---
training_args = TrainingArguments(
    output_dir="./flan-t5-lora-normalisation",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    bf16=True,  # BF16 per stabilità su A6000
    save_strategy="epoch",
    logging_steps=50,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_total_limit=2,
)

# --- Mini-batch di test (prima del training) ---
mini_batch_raw = tokenized["train"][:2]
mini_batch = {
    "input_ids": torch.tensor(mini_batch_raw["input_ids"]).to(model.device),
    "attention_mask": torch.tensor(mini_batch_raw["attention_mask"]).to(model.device),
    "labels": torch.tensor(mini_batch_raw["labels"]).to(model.device),
}
outputs = model(**mini_batch)
print(f"Mini-batch loss di test: {outputs.loss.item()}")

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[GradNormCallback],
)

# --- Avvio fine-tuning ---
trainer.train()
trainer.evaluate()

# --- Salvataggio modello LoRA finale ---
trainer.save_model("./flan-lora-normalisation/final")
print("Fine-tuning terminato e modello salvato in ./flan-lora-normalisation/final")

# --- Test inferenza ---
text = "Ho visto un cane bellissmo stamane"
inputs = tokenizer(f"Normalizza il testo seguente: {text}", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
