import os, json
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session

from models.recommend import RecommenderSystem
from utils.visualization import *
from utils.evaluation import (
    evaluate_cf_rmse_mae, precision_recall_at_k, coverage, evaluate_content_based
)

# CẤU HÌNH ỨNG DỤNG
app = Flask(__name__)
app.secret_key = "anime-secret-key"

# Các đường dẫn và tham số cấu hình chính
DATA_PATH = "data/anime_clean.csv"
USER_FILE = "data/users.csv"
RATING_FILE = "data/ratings.csv"
MODEL_DIR = "models"
ITEMS_PER_PAGE = 24
TOP_K = 10
CHARTS_CACHE_FILE = "utils/.charts_cache.json"
EVAL_CACHE_FILE = "utils/.eval_cache.json"

# Load dữ liệu anime đã làm sạch
df = pd.read_csv(DATA_PATH)

# Khởi tạo hệ thống gợi ý với đường dẫn dữ liệu và thư mục model
rec_system = RecommenderSystem(
    anime_path=DATA_PATH,
    rating_path=RATING_FILE,
    model_dir=MODEL_DIR
)

# Thử load model đã huấn luyện để dùng ngay khi chạy app
try:
    rec_system.load()
    rec_system.load_data()
    print("Đã tải thành công hệ thống gợi ý từ thư mục models")
except Exception as e:
    print(f"Cảnh báo: Không thể tải các mô hình gợi ý: {e}")
    
# Tạo file csv nếu chưa có để tránh lỗi đọc/ghi
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["user_id", "username", "password"]).to_csv(
        USER_FILE, index=False
    )
    print(f"Created {USER_FILE}")

if not os.path.exists(RATING_FILE):
    pd.DataFrame(columns=["user_id", "anime_id", "rating"]).to_csv(
        RATING_FILE, index=False
    )
    print(f"Created {RATING_FILE}")


# Tạo danh sách phân trang
def get_pagination(page, total_pages):
    pagination = []
    if page > 3:
        pagination.append(1)
    if page > 4:
        pagination.append("...")
    for p in range(page - 2, page + 3):
        if 1 <= p <= total_pages:
            pagination.append(p)
    if page < total_pages - 3:
        pagination.append("...")
    if page < total_pages - 2:
        pagination.append(total_pages)
    return pagination

# Lấy thời gian sửa file làm hash đơn giản
def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    return os.path.getmtime(filepath)

# Lấy biểu đồ từ cache nếu có
def get_charts_from_cache():
    if os.path.exists(CHARTS_CACHE_FILE):
        try:
            with open(CHARTS_CACHE_FILE, 'r') as f:
                cache = json.load(f)
            return cache.get('charts')
        except:
            return None
    return None

# Lưu biểu đồ vào cache
def save_charts_to_cache(charts):
    try:
        cache = {
            'charts': charts,
            'timestamp': datetime.now().isoformat()
        }
        with open(CHARTS_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass

# Lấy kết quả đánh giá model từ cache
def get_eval_from_cache():
    if os.path.exists(EVAL_CACHE_FILE):
        try:
            with open(EVAL_CACHE_FILE, 'r') as f:
                cache = json.load(f)
            return cache.get('metrics')
        except:
            return None
    return None

# Lưu kết quả đánh giá model vào cache
def save_eval_to_cache(metrics):
    try:
        cache = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(EVAL_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass


# Trang chủ hiển thị danh sách anime kèm phân trang
@app.route("/")
def home():
    # Lấy số trang hiện tại
    page = int(request.args.get("page", 1))
    per_page = ITEMS_PER_PAGE

    # Tính tổng số trang
    total_items = len(df)
    total_pages = (total_items + per_page - 1) // per_page

    # Chuẩn hóa số trang
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    # Cắt dữ liệu theo trang
    start = (page - 1) * per_page
    end = start + per_page
    items = df.iloc[start:end]

    # Tạo thanh phân trang
    pagination = get_pagination(page, total_pages)

    return render_template(
        "home.html",
        animes=items.to_dict(orient="records"),
        page=page,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages,
        pagination=pagination,
    )

# Chi tiết anime kèm gợi ý Content-Based
@app.route("/anime/<int:anime_id>")
def detail(anime_id):
    # Lấy thông tin anime theo id
    anime = df[df["id"] == anime_id]
    if anime.empty:
        return "Anime not found", 404

    anime = anime.iloc[0]

    # Gợi ý anime tương tự bằng Content-Based
    recommendations = []
    try:
        recommendations = rec_system.recommend_content(anime_id, top_k=TOP_K)
    except Exception as e:
        print(f"Content-based recommendation failed: {e}")

    # Lấy rating của user nếu đã đăng nhập
    user_rating = None
    if "user_id" in session:
        user_id = session["user_id"]
        if os.path.exists(RATING_FILE):
            rating_df = pd.read_csv(RATING_FILE)
            row = rating_df[
                (rating_df["user_id"] == user_id)
                & (rating_df["anime_id"] == anime_id)
            ]
            if not row.empty:
                user_rating = int(row.iloc[0]["rating"])

    return render_template(
        "detail.html",
        anime=anime,
        recommendations=recommendations,
        user_rating=user_rating,
    )

# Đăng nhập
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        df_users = pd.read_csv(USER_FILE, encoding="utf-8-sig")
        df_users["username"] = df_users["username"].astype(str).str.strip()
        df_users["password"] = df_users["password"].astype(str).str.strip()

        user = df_users[
            (df_users["username"] == username) & (df_users["password"] == password)
        ]

        if len(user) == 1:
            session["user_id"] = int(user.iloc[0]["user_id"])
            session["username"] = username
            return redirect(url_for("home"))

        return render_template("login.html", error="Sai tài khoản hoặc mật khẩu")

    return render_template("login.html")

# Đăng ký
@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration page."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        df_users = pd.read_csv(USER_FILE)
        if username in df_users["username"].values:
            return render_template("register.html", error="Username đã tồn tại")

        new_id = 1 if df_users.empty else df_users["user_id"].max() + 1
        df_users.loc[len(df_users)] = [new_id, username, password]
        df_users.to_csv(USER_FILE, index=False)

        return redirect(url_for("login"))

    return render_template("register.html")

# Đăng xuất
@app.route("/logout")
def logout():
    """User logout."""
    session.clear()
    return redirect(url_for("home"))


# Đánh giá anime
@app.route("/rate/<int:anime_id>/<int:score>")
def rate(anime_id, score):
    # Ghi hoặc cập nhật rating của user cho một anime
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if os.path.exists(RATING_FILE):
        df_rating = pd.read_csv(RATING_FILE)
    else:
        df_rating = pd.DataFrame(columns=["user_id", "anime_id", "rating"])

    mask = (df_rating["user_id"] == user_id) & (df_rating["anime_id"] == anime_id)

    if mask.any():
        df_rating.loc[mask, "rating"] = score
    else:
        df_rating.loc[len(df_rating)] = [user_id, anime_id, score]

    df_rating.to_csv(RATING_FILE, index=False)
    return redirect(url_for("detail", anime_id=anime_id))

@app.route("/rate/delete/<int:anime_id>")
def delete_rating(anime_id):
    # Xóa rating của user cho một anime
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not os.path.exists(RATING_FILE):
        return redirect(url_for("detail", anime_id=anime_id))

    df_rating = pd.read_csv(RATING_FILE)
    df_rating = df_rating[
        ~((df_rating["user_id"] == user_id) & (df_rating["anime_id"] == anime_id))
    ]

    df_rating.to_csv(RATING_FILE, index=False)
    return redirect(url_for("rating_history"))


# Lịch sử rating và gợi ý theo Collaborative Filtering
@app.route("/ratings")
def rating_history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not os.path.exists(RATING_FILE):
        rating_df = pd.DataFrame(columns=["user_id", "anime_id", "rating"])
    else:
        rating_df = pd.read_csv(RATING_FILE)

    anime_df = df

    user_ratings = (
        rating_df[rating_df["user_id"] == user_id]
        .sort_index(ascending=False)
    )

    # Gộp rating với thông tin anime để render UI
    merged = user_ratings.merge(
        anime_df, left_on="anime_id", right_on="id", how="left"
    )

    ratings = []
    for _, row in merged.iterrows():
        ratings.append(
            {
                "id": int(row["anime_id"]),
                "title": row["title"],
                "image": row["image"],
                "genres": row["genres"],
                "rating": int(row["rating"]),
            }
        )

    # Sinh gợi ý CF dựa trên lịch sử rating hiện tại
    recommendations = rec_system.recommend_cf(user_id, top_k=TOP_K)

    return render_template("ratings.html", ratings=ratings, recommendations=recommendations)


# Trang admin hiển thị thống kê, biểu đồ và đánh giá mô hình
@app.route("/admin")
def admin_dashboard():
    df_anime = df.copy()
    df_users = pd.read_csv(USER_FILE)
    df_rating = pd.read_csv(RATING_FILE)

    # Thống kê cơ bản
    total_items = len(df_anime)
    total_users = len(df_users)
    total_rating = len(df_rating)
    users_with_rating = df_rating["user_id"].nunique()

    # Phân tích rating
    top_rated = (
        df_rating.groupby("anime_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
        .merge(df_anime[["id", "title", "image"]],
               left_on="anime_id", right_on="id")
    )

    avg_rating = (
        df_rating.groupby("anime_id")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    avg_rating = avg_rating[avg_rating["count"] >= 5]
    avg_rating = avg_rating.sort_values("mean", ascending=False).head(10)
    avg_rating = avg_rating.merge(
        df_anime[["id", "title", "image"]],
        left_on="anime_id", right_on="id"
    )

    # Trạng thái mô hình
    model_status = "Loaded" if rec_system.cf.P is not None else "Not loaded"
    num_cf_users = len(rec_system.cf.user_id_to_idx)
    num_cf_items = len(rec_system.cf.item_id_to_idx)
    num_factors = rec_system.cf.n_factors

    # Biểu đồ
    charts = get_charts_from_cache()

    if charts is None:
        print("Đang tạo lại các biểu đồ thống kê...")

        rating_dist_img = rating_distribution(df_rating)
        user_activity_img = user_activity_distribution(df_rating)
        item_popularity_img = item_popularity_distribution(df_rating)

        genres_img = top_genres(df_anime)
        heatmap_img = correlation_heatmap(df_anime)
        genre_overlap_img = genre_overlap_heatmap(df_anime)

        top_rated_count_img = top_rated_count_chart(df_rating, df_anime)

        charts = {
            "rating_dist_img": rating_dist_img,
            "user_activity_img": user_activity_img,
            "item_popularity_img": item_popularity_img,
            "genres_img": genres_img,
            "heatmap_img": heatmap_img,
            "genre_overlap_img": genre_overlap_img,
            "top_rated_count_img": top_rated_count_img,
        }
        save_charts_to_cache(charts)
    else:
        print("Sử dụng biểu đồ từ bộ nhớ cache")

    # Đánh giá mô hình
    eval_metrics = get_eval_from_cache()

    if eval_metrics is None:
        print("Đang tính toán các chỉ số đánh giá mô hình...")
        if len(df_rating) > 20:
            rmse, mae = evaluate_cf_rmse_mae(rec_system.cf, df_rating)
            cf_precision, cf_recall = precision_recall_at_k(rec_system.cf, df_rating, k=10)
            cf_cov = coverage(rec_system.cf, total_items)
        else:
            rmse = mae = cf_precision = cf_recall = cf_cov = 0

        cb_precision, cb_recall = evaluate_content_based(rec_system, df_anime, k=10)

        eval_metrics = {
            "rmse": rmse,
            "mae": mae,
            "cf_precision": cf_precision,
            "cf_recall": cf_recall,
            "cf_cov": cf_cov,
            "cb_precision": cb_precision,
            "cb_recall": cb_recall,
        }
        save_eval_to_cache(eval_metrics)

    return render_template(
        "admin.html",
        total_items=total_items,
        total_users=total_users,
        total_rating=total_rating,
        users_with_rating=users_with_rating,

        top_rated=top_rated.to_dict("records"),
        avg_rating=avg_rating.to_dict("records"),

        model_status=model_status,
        num_cf_users=num_cf_users,
        num_cf_items=num_cf_items,
        num_factors=num_factors,

        **charts,
        **eval_metrics,
    )


# Giới thiệu
@app.route("/intro")
def intro():
    return render_template("intro.html")

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)