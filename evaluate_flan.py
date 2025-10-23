# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:24:36 2025

@author: matil
"""
import torch
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

# ### Config ###
model_path = "./flan-lora-normalisation/final"
dataset_path = {
    "train": "data/scolinter_ita/train.jsonl",
    "validation": "data/scolinter_ita/validation.jsonl"
}
batch_size = 8

# ### Carica modello e tokenizer ###
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# ### Carica dataset ###
dataset = load_dataset("json", data_files=dataset_path)
val_set = dataset["validation"]

# ### Metriche ###
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

preds = []
refs = []

# ### Generazione batch con barra di avanzamento ###
print("Generazione testi di validation set...")
for start_idx in tqdm(range(0, len(val_set), batch_size), desc="Validation generation", ncols=100):
    end_idx = min(start_idx + batch_size, len(val_set))
    batch = val_set.select(range(start_idx, end_idx))

    inputs_texts = ["Normalizza il testo seguente: {}".format(batch[i]['input']) for i in range(len(batch))]
    inputs = tokenizer(
        inputs_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=128
    )

    pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds.extend(pred_texts)
    refs.extend([batch[i]['output'] for i in range(len(batch))])

# ### Calcolo metriche ###
rouge_result = rouge.compute(predictions=preds, references=refs)
bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs])
exact_match = np.mean([p.strip() == r.strip() for p, r in zip(preds, refs)])

# ### BLEU classico (nltk) ###
refs_tokenized = [[r.split()] for r in refs]
preds_tokenized = [p.split() for p in preds]
bleu_nltk = corpus_bleu(refs_tokenized, preds_tokenized)

# ### Stampa risultati ###
print("\nRISULTATI:")
if isinstance(rouge_result['rougeL'], dict):
    print("ROUGE-L F1: {:.4f}".format(rouge_result['rougeL']['fmeasure']))
else:
    print("ROUGE-L F1: {:.4f}".format(rouge_result['rougeL']))
print("SacreBLEU: {:.4f}".format(bleu_result['score']))
print("BLEU (nltk corpus-level): {:.4f}".format(bleu_nltk))
print("Exact Match: {:.2f}%".format(exact_match*100))

# ### Test qualitativo ###
testi = [
    "ho visto un cane bellissmo stamane",
    "mi chiamo martina e abito a roma",
    "la mia scuola e molto bellaa!!",
]

print("\nEsempi di inferenza qualitativa:")
for t in testi:
    inp = "Normalizza il testo seguente: {}".format(t)
    inputs = tokenizer(inp, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=128
    )
    print("\nInput: {}".format(t))
    print("Output: {}".format(tokenizer.decode(output[0], skip_special_tokens=True)))

# ### Prime 5 predizioni validation set ###
print("\nPrime 5 predizioni del validation set:")
for i in range(min(5, len(preds))):
    print("\nTarget   : {}".format(refs[i]))
    print("Predicted: {}".format(preds[i]))
