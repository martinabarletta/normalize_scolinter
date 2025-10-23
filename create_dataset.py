# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 21:26:34 2025

@author: matil

code pour la création du dataset à utiliser pour finetuning/LoRA
input - csv d'export de la plateforme Scolinter, contient input (transcriptions) et output(normalisations)
output - json train, validation et test

le dataset en entrée contient environ 1500 lignes, les textes sont en italien (dont l'encodage utilisé)
"""

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv("./scolinter_italien_fev_2025.csv", encoding="ISO-8859-1", sep=";")  # replace with your actual filename

#delete empty lines
df = df.dropna(subset=["transcription", "normalisation"])
df = df[df["transcription"].str.strip() != ""]
df = df[df["normalisation"].str.strip() != ""]

#remove copies
df = df.drop_duplicates(subset=["transcription", "normalisation"])

#remove input == output
df = df[df["transcription"].str.lower().str.strip() != df["normalisation"].str.lower().str.strip()]

print(f"{len(df)} texts in dataset")

# 5️⃣ Divisione train / val / test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 6️⃣ Funzione di scrittura in JSONL con formattazione tipo "prompt + completion"
def write_jsonl(dataframe, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            example = {
                "input": f"{row['transcription'].strip()}",
                "output": row["normalisation"].strip(),
            }
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

os.makedirs("data", exist_ok=True)
write_jsonl(train_df, "data/train.jsonl")
write_jsonl(val_df, "data/validation.jsonl")
write_jsonl(test_df, "data/test.jsonl")

print("dataset ready!")
