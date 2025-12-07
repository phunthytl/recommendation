import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Font đẹp hơn
sns.set(style="whitegrid")


def load_data(path="../data/anime_clean.csv"):
    return pd.read_csv(path)


# ========================
# 1. Histogram Score
# ========================
def plot_score_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df["score"], bins=30, kde=True, color="skyblue")
    plt.title("Distribution of Anime Scores")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.show()


# ========================
# 2. Top Genres
# ========================
def plot_top_genres(df, top_n=20):
    # explode genres_list
    df_genres = df.explode("genres_list")
    genre_count = df_genres["genres_list"].value_counts().head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_count.values, y=genre_count.index)
    plt.title(f"Top {top_n} Genres")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.show()


# ========================
# 3. Popularity vs Score
# ========================
def plot_popularity_vs_score(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="score", y="popularity", alpha=0.5)
    plt.title("Score vs Popularity")
    plt.xlabel("Score")
    plt.ylabel("Popularity (lower = more popular)")
    plt.show()


# ========================
# 4. Heatmap của numeric correlation
# ========================
def plot_correlation_heatmap(df):
    numeric = df[["score", "rank", "popularity", "favorites", "episodes"]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="Blues")
    plt.title("Correlation Heatmap")
    plt.show()


# ========================
# RUN ALL
# ========================
def run_all_visualizations(path="../data/anime_clean.csv"):
    df = load_data(path)

    print("📊 Running EDA Visualizations...")
    plot_score_distribution(df)
    plot_top_genres(df)
    plot_popularity_vs_score(df)
    plot_correlation_heatmap(df)
    print("✔ DONE!")
