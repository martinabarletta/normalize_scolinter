# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 21:26:34 2025

@author: matil
"""

import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv("./scolinter_italien_fev_2025.csv", encoding="ISO-8859-1", sep=";")  # replace with your actual filename

#check col names, must contain "transcription" and "normalisation"
print(df.columns)

#Supprimer les lignes où 'transcription' est NaN ou vide
df = df.dropna(subset=["transcription"])  # supprime les NaN
df = df[df["transcription"].str.strip() != ""]  # supprime les lignes vides
#<empty/>
df = df[df["transcription"].str.strip() != "<empty/>"]  # supprime les lignes vides
df = df[df["normalisation"].str.strip() != ""]  # supprime les lignes vides


# Make sure they match exactly: 'transcription' and 'normalisation'
assert "transcription" in df.columns and "normalisation" in df.columns, \
    "The CSV must have 'transcription' and 'normalisation' columns."

#Diviser le dataset en train / validation (90 / 10)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

#Fonction d’écriture en JSONL UTF-8
def write_jsonl(dataframe, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for _, row in dataframe.iterrows():
            ex = {
                "input": str(row["transcription"]).strip(),
                "output": str(row["normalisation"]).strip(),
            }
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

#Écrire les deux fichiers
os.makedirs("data", exist_ok=True)
write_jsonl(train_df, "data/train.jsonl")
write_jsonl(val_df, "data/validation.jsonl")

print("Fichiers créés avec succès :")
print(" - data/train.jsonl")
print(" - data/validation.jsonl")