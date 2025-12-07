import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return encoded


# ==========================
# 1) Histogram Score
# ==========================
def score_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["score"].dropna(), bins=20, color="orange", edgecolor="black")
    plt.title("Phân bố điểm số")
    return plot_to_base64()


# ==========================
# 2) Top genres
# ==========================
def top_genres(df, top_n=10):
    genre_count = df["genres"].value_counts().head(top_n)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=genre_count.values, y=genre_count.index)
    plt.title("Top Genres")
    return plot_to_base64()


# ==========================
# 3) Top Favorites
# ==========================
def top_favorites(df_fav):
    top_fav = df_fav["anime_id"].value_counts().head(10)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=top_fav.values, y=top_fav.index)
    plt.title("Top Favorites")
    return plot_to_base64()


# ==========================
# 4) Heatmap
# ==========================
def correlation_heatmap(df):
    numeric_cols = ["score", "popularity", "favorites"]
    df_num = df[numeric_cols].dropna()

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap tương quan")
    return plot_to_base64()
