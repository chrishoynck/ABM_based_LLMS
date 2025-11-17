import json, os
import pandas as pd
from datasets import load_dataset
ds = load_dataset("nvidia/Nemotron-Personas")
ds_small = ds["train"].shuffle(seed=42).select(range(10000))
# print(ds['train'][0]['professional_persona'])

# if not "personas_10k.jsonl" in os.listdir("data/"):
#     ds_small.to_json("data/personas_10k.jsonl", lines=True)

# personas = []
# with open("data/personas_10k.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         personas.append(json.loads(line))
# print(personas[0].keys())

df = ds_small.to_pandas()

# Clip every string field to max 120 chars
MAX_LEN = 120
def _clip_cell(x, n=MAX_LEN):
    return x[:n] if isinstance(x, str) else x

def age_of_person(row):
    try:
        age = int(row['age'])
    except:
        print(f"Could not convert age: {row['age']}")
        return False
    if age <= 16 or age >= 80:
        return False
    else:
        return True
    
df = df.applymap(_clip_cell)
df = df[df.apply(age_of_person, axis=1)]

if "personas_10k.csv" not in os.listdir("data/"):
    df.to_csv("data/personas_10k.csv", index=False)
df = pd.read_csv("data/personas_10k.csv")
print(df.head())
print(df.iloc[0]['professional_persona'])
print (df.iloc[0]['age'])
print(df.columns)