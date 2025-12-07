import pandas as pd
import numpy as np
import re

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("data/anime_data.csv")

print("👉 Original shape:", df.shape)


# ==========================
# 1. Remove duplicates
# ==========================
df = df.drop_duplicates(subset=["title"])
print("✔ Removed duplicates. New shape:", df.shape)


# ==========================
# 2. Handle missing values
# ==========================

# fill missing synopsis with empty string
df["synopsis"] = df["synopsis"].fillna("")

# fill missing genres with empty string
df["genres"] = df["genres"].fillna("")

# episodes missing → set to 0
df["episodes"] = df["episodes"].fillna(0).astype(int)

# score missing → replace with mean
df["score"] = df["score"].fillna(df["score"].mean())

# favorites missing → 0
df["favorites"] = df["favorites"].fillna(0)

print("✔ Missing values handled.")


# ==========================
# 3. Clean genres
# ==========================

def clean_genres(g):
    if isinstance(g, str):
        return [x.strip() for x in g.split(",") if x.strip() != ""]
    return []

df["genres_list"] = df["genres"].apply(clean_genres)
print("✔ Genres cleaned → genres_list")


# ==========================
# 4. Clean synopsis text
# ==========================

def clean_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = txt.replace("(Source: ANN)", "")
    txt = txt.replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt)  # normalize spaces
    return txt.strip()

df["synopsis_clean"] = df["synopsis"].apply(clean_text)
print("✔ Cleaned synopsis text")


# ==========================
# 5. Normalize numeric columns
# ==========================

numeric_cols = ["episodes", "score", "rank", "popularity", "favorites"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

print("✔ Normalized numeric data")


# ==========================
# 6. Save cleaned file
# ==========================

df.to_csv("data/anime_clean.csv", index=False, encoding="utf-8-sig")
print("🎉 Saved anime_clean.csv successfully!")
