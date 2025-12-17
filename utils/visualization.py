import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import combinations


# Vẽ biểu đồ cột thể hiện phân bố điểm rating của người dùng

def rating_distribution(df_rating):
    # Kiểm tra dữ liệu rating rỗng hoặc thiếu cột rating
    if df_rating.empty or "rating" not in df_rating.columns:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Phân bố đánh giá người dùng", height=500)
        return fig.to_html(include_plotlyjs=False, div_id="rating-dist-chart")
    
    # Đếm số lượng rating theo từng mức điểm
    rating_counts = df_rating["rating"].value_counts().sort_index()
    
    # Trường hợp không có rating hợp lệ
    if len(rating_counts) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Phân bố đánh giá người dùng", height=500)
        return fig.to_html(include_plotlyjs=False, div_id="rating-dist-chart")
    
    # Tạo biểu đồ cột phân bố rating
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
    return fig.to_html(include_plotlyjs='cdn', div_id="rating-dist-chart")


# Vẽ biểu đồ top N thể loại anime phổ biến nhất

def top_genres(df, top_n=10):
    # Tách genres dạng chuỗi thành từng thể loại riêng lẻ
    all_genres = []
    for genres_str in df["genres"].dropna():
        if isinstance(genres_str, str):
            genres = [g.strip() for g in genres_str.split(',')]
            all_genres.extend(genres)
    
    # Đếm số lần xuất hiện của mỗi thể loại
    from collections import Counter
    genre_count = Counter(all_genres)
    genre_df = pd.DataFrame(
        list(genre_count.most_common(top_n)),
        columns=['Genre', 'Count']
    )
    
    # Vẽ biểu đồ cột ngang cho top genres
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


# Vẽ heatmap tương quan giữa các thuộc tính số của anime

def correlation_heatmap(df):
    # Chọn các cột số để tính tương quan
    numeric_cols = ["score", "popularity", "favorites"]
    df_num = df[numeric_cols].dropna()
    
    # Tính ma trận tương quan
    corr_matrix = df_num.corr()
    
    # Tạo biểu đồ heatmap
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


# Vẽ histogram thể hiện mức độ hoạt động của người dùng

def user_activity_distribution(df_rating):
    # Trường hợp chưa có dữ liệu rating
    if df_rating.empty:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu rating", showarrow=False)
        fig.update_layout(title="Phân bố mức độ hoạt động người dùng")
        return fig.to_html(include_plotlyjs=False, div_id="user-activity-chart")

    # Đếm số rating của mỗi user
    counts = df_rating.groupby("user_id").size()

    # Vẽ histogram số rating trên mỗi user
    fig = px.histogram(
        counts,
        nbins=30,
        title="Phân bố số lượng đánh giá trên mỗi người dùng",
        labels={"value": "Số lượng rating", "count": "Số người dùng"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig.update_layout(height=500, showlegend=False)
    return fig.to_html(include_plotlyjs=False, div_id="user-activity-chart")


# Vẽ biểu đồ top anime được đánh giá nhiều nhất

def top_rated_count_chart(df_rating, df_anime):
    # Trường hợp chưa có dữ liệu rating
    if df_rating.empty:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Top 15 Anime được đánh giá nhiều nhất", height=600)
        return fig.to_html(include_plotlyjs=False, div_id="top-rated-chart")
    
    # Đếm số lượt rating theo anime
    top_rated = (
        df_rating.groupby("anime_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    
    # Trường hợp không có anime nào được đánh giá
    if len(top_rated) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu đánh giá", showarrow=False)
        fig.update_layout(title="Top 15 Anime được đánh giá nhiều nhất", height=600)
        return fig.to_html(include_plotlyjs=False, div_id="top-rated-chart")
    
    # Gộp dữ liệu rating với thông tin anime
    top_rated = top_rated.merge(df_anime[["id", "title"]], left_on="anime_id", right_on="id")
    
    # Vẽ biểu đồ cột ngang top anime
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


# Vẽ heatmap thể hiện mức độ chồng chéo giữa các thể loại anime

def genre_overlap_heatmap(df_anime):
    # Chuẩn hóa cột genres_list về dạng list
    df = df_anime.copy()
    df["genres_list"] = df["genres_list"].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # Lấy danh sách tất cả các thể loại
    genres = sorted({g for lst in df["genres_list"] for g in lst})

    # Khởi tạo ma trận đếm chồng chéo thể loại
    matrix = pd.DataFrame(0, index=genres, columns=genres)

    # Đếm số lần các thể loại xuất hiện cùng nhau
    for lst in df["genres_list"]:
        for g1, g2 in combinations(lst, 2):
            matrix.loc[g1, g2] += 1
            matrix.loc[g2, g1] += 1
        for g in lst:
            matrix.loc[g, g] += 1

    # Vẽ heatmap chồng chéo thể loại
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


# Vẽ histogram thể hiện độ phổ biến của anime theo số lượng rating

def item_popularity_distribution(df_rating):
    # Trường hợp chưa có dữ liệu rating
    if df_rating.empty:
        fig = go.Figure()
        fig.add_annotation(text="Chưa có dữ liệu rating", showarrow=False)
        fig.update_layout(title="Phân bố độ phổ biến Anime")
        return fig.to_html(include_plotlyjs=False, div_id="item-popularity-chart")

    # Đếm số rating trên mỗi anime
    counts = df_rating.groupby("anime_id").size()

    # Vẽ histogram độ phổ biến anime
    fig = px.histogram(
        counts,
        nbins=30,
        title="Phân bố số lượng đánh giá trên mỗi Anime",
        labels={"value": "Số lượng rating", "count": "Số Anime"},
        color_discrete_sequence=["#d62728"]
    )
    fig.update_layout(height=500, showlegend=False)
    return fig.to_html(include_plotlyjs=False, div_id="item-popularity-chart")
