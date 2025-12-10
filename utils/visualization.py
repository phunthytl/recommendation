import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
from itertools import combinations


# ==========================
# 1) Histogram Score
# ==========================
def score_distribution(df):
    """Distribution of anime scores"""
    fig = px.histogram(
        df,
        x="score",
        nbins=20,
        title="Phân bố điểm số Anime",
        labels={"score": "Điểm", "count": "Số lượng"},
        color_discrete_sequence=["#FF7F0E"]
    )
    fig.update_layout(height=500, showlegend=False)
    return fig.to_html(include_plotlyjs='cdn', div_id="score-dist-chart")


# ==========================
# 2) Top genres
# ==========================
def top_genres(df, top_n=10):
    """Top genres bar chart - count each genre separately"""
    # Split genres (they are comma-separated in the dataframe)
    all_genres = []
    for genres_str in df["genres"].dropna():
        if isinstance(genres_str, str):
            # Split by comma and strip whitespace
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.extend(genres)
    
    # Count genres
    from collections import Counter
    genre_count = Counter(all_genres)
    genre_df = pd.DataFrame(
        list(genre_count.most_common(top_n)),
        columns=['Genre', 'Count']
    )
    
    fig = px.bar(
        genre_df,
        x='Count',
        y='Genre',
        orientation='h',
        title="Top Genres",
        labels={"Count": "Số lượng", "Genre": "Thể loại"},
        color='Count',
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=500, showlegend=False)
    return fig.to_html(include_plotlyjs=False, div_id="genres-chart")


# ==========================
# 3) Heatmap (Plotly)
# ==========================
def correlation_heatmap(df):
    """Correlation heatmap"""
    numeric_cols = ["score", "popularity", "favorites"]
    df_num = df[numeric_cols].dropna()
    
    corr_matrix = df_num.corr()
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 12}
        )
    )
    fig.update_layout(
        title="Heatmap tương quan (Anime)",
        height=500,
        xaxis_title="",
        yaxis_title=""
    )
    return fig.to_html(include_plotlyjs=False, div_id="heatmap-chart")


# ==========================
# 4) Rating Distribution (Bar Chart)
# ==========================
def rating_distribution(df_rating):
    """Distribution of ratings (1-10 scale)"""
    if df_rating.empty or "rating" not in df_rating.columns:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Phân bố đánh giá người dùng", height=500)
        return fig.to_html(include_plotlyjs=False, div_id="rating-dist-chart")
    
    rating_counts = df_rating["rating"].value_counts().sort_index()
    
    if len(rating_counts) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Phân bố đánh giá người dùng", height=500)
        return fig.to_html(include_plotlyjs=False, div_id="rating-dist-chart")
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Phân bố đánh giá người dùng",
        labels={"x": "Điểm đánh giá", "y": "Số lượng"},
        color=rating_counts.values,
        color_continuous_scale="Viridis"
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.update_layout(height=500, showlegend=False)
    return fig.to_html(include_plotlyjs=False, div_id="rating-dist-chart")


# ==========================
# 6) Top Rated Anime Count (Bar Chart)
# ==========================
def top_rated_count_chart(df_rating, df_anime):
    """Top anime by rating count"""
    if df_rating.empty:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Top 15 Anime được đánh giá nhiều nhất", height=600)
        return fig.to_html(include_plotlyjs=False, div_id="top-rated-chart")
    
    top_rated = (
        df_rating.groupby("anime_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    
    if len(top_rated) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Top 15 Anime được đánh giá nhiều nhất", height=600)
        return fig.to_html(include_plotlyjs=False, div_id="top-rated-chart")
    
    # Merge with anime info
    top_rated = top_rated.merge(df_anime[["id", "title"]], left_on="anime_id", right_on="id")
    
    fig = px.bar(
        top_rated,
        x="count",
        y="title",
        orientation='h',
        title="Top 15 Anime được đánh giá nhiều nhất",
        labels={"count": "Số lượng đánh giá", "title": "Anime"},
        color="count",
        color_continuous_scale="Reds"
    )
    fig.update_layout(height=600, showlegend=False)
    return fig.to_html(include_plotlyjs=False, div_id="top-rated-chart")

def similarity_heatmap(cb_model, df_anime, top_n=20):
    """
    Heatmap similarity của 20 anime (hoặc số tùy chọn)
    dùng TF-IDF vector từ Content-Based model.
    """
    # Lấy 20 anime ngẫu nhiên hoặc top popularity
    sample = df_anime.head(top_n)
    titles = sample["title"].tolist()
    ids = sample["id"].tolist()

    # Lấy vector TF-IDF từ model
    vectors = []
    for aid in ids:
        vectors.append(cb_model.vectorize(aid))  # bạn đã có hàm này trong model TF-IDF

    vectors = np.vstack(vectors)

    # cosine similarity
    sim_matrix = cosine_similarity(vectors)

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=titles,
        y=titles,
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        text=np.round(sim_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))

    fig.update_layout(
        title="Heatmap Similarity Anime–Anime (Cosine)",
        height=800,
        width=800
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="sim-heatmap")

def genre_overlap_heatmap(df_anime):
    # Chuẩn hóa genres_list thành list python
    df = df_anime.copy()
    df["genres_list"] = df["genres_list"].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # Lấy tất cả genre
    genres = sorted({g for lst in df["genres_list"] for g in lst})

    # Tạo ma trận đếm
    matrix = pd.DataFrame(0, index=genres, columns=genres)

    for lst in df["genres_list"]:
        for g1, g2 in combinations(lst, 2):
            matrix.loc[g1, g2] += 1
            matrix.loc[g2, g1] += 1
        for g in lst:
            matrix.loc[g, g] += 1

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=genres,
        y=genres,
        colorscale="Blues",
        text=matrix.values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))

    fig.update_layout(
        title="Genre Overlap Heatmap",
        height=900,
        width=900
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="genre-heatmap") 

def sparse_matrix_heatmap(df_rating, df_anime, top_users=30, top_items=30):
    # Lấy user và item phổ biến nhất
    top_u = df_rating["user_id"].value_counts().head(top_users).index
    top_i = df_rating["anime_id"].value_counts().head(top_items).index

    df = df_rating[df_rating["user_id"].isin(top_u) & df_rating["anime_id"].isin(top_i)]

    pivot = df.pivot_table(
        index="user_id",
        columns="anime_id",
        values="rating",
        fill_value=0
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd"
    ))

    fig.update_layout(
        title="User–Item Rating Matrix (Top Users/Items)",
        height=700
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="matrix-heatmap")