import ast
import math
import os
import pandas as pd
from datasets import load_dataset
ds = load_dataset("nvidia/Nemotron-Personas")
ds_small = ds["train"].shuffle(seed=42).select(range(10000))
df = ds_small.to_pandas()

# Clip every string field to max 120 chars
MAX_LEN = 200
NEW = True


def _clip_cell(x, n=MAX_LEN):
    if isinstance(x, str) and len(x) > n:
        if x[0] == '[' and x[-1] == ']':
            return x
        return x[:n]
    else:
        return x

def age_of_person(row):
    try:
        age = int(row['age'])
    except Exception:
        print(f"Could not convert age: {row['age']}")
        return False
    if age <= 16 or age >= 80:
        return False
    else:
        return True
    
df = df.map(_clip_cell)
df = df[df.apply(age_of_person, axis=1)]
columns_keep = ['persona', 'age', 'marital_status', 'hobbies_and_interests_list', 'skills_and_expertise_list','sex','bachelors_field', 'occupation', 'city' ]

if "personas_10k.csv" not in os.listdir("data/") or NEW:
    df.to_csv("data/personas_10k.csv", columns=columns_keep, index=False)
df = pd.read_csv("data/personas_10k.csv")


def parse_list_field(v):
    if pd.isna(v):
        return []
    v = str(v).strip()
    # JSON / Python-list style: ["a", "b", ...]
    if v.startswith("["):
        try:
            return [x.strip() for x in ast.literal_eval(v)]
        except Exception:
            pass
    # fallback: comma-separated
    return [x.strip() for x in v.split(",") if x.strip()]

def extract_name(persona_text):
    parts = persona_text.strip().split()
    return " ".join(parts[:2])  # first two words

def row_to_persona(row):
    return {
        "name": extract_name(row["persona"]),
        "persona_text": row["persona"],  # the long description text
        "age": int(row["age"]) if not math.isnan(row["age"]) else None,
        "sex": row["sex"],
        "marital_status": row["marital_status"],
        "bachelors_field": row["bachelors_field"],
        "occupation": row["occupation"],
        "city": row["city"],
        "hobbies": parse_list_field(row["hobbies_and_interests_list"]),
        "skills": parse_list_field(row["skills_and_expertise_list"]),
    }

def load_distorted_tweets(filepath="data/distorted_tweets.csv", numtweets=1000, seed=42):
    df = pd.read_csv(filepath)
    df_sampled = df.sample(n=numtweets, replace=True, random_state=seed)
    return df_sampled['tweet'].tolist()

def load_depressed_personas(filepath="data/depressed.csv", personass_to_load=1, seed=42):
    df = pd.read_csv(filepath)
    return [row_to_persona(row) for _, row in df.sample(n=personass_to_load, replace=True, random_state=seed).iterrows()]

def load_personas_from_file(filepath="data/personas_10k.csv", personass_to_load=10, seed=42):
    df = pd.read_csv(filepath)
    return [row_to_persona(row) for _, row in df.sample(n=personass_to_load, replace=False, random_state=seed).iterrows()]
