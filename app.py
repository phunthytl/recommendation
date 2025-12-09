from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd, os, numpy as np, matplotlib
matplotlib.use('Agg')

from models.collaborative import CollaborativeFiltering  # CF dùng favorites.csv (cho trang /favorites)
from models.recommend import RecommenderSystem
from utils.visualization import (
    score_distribution, top_genres, top_favorites, correlation_heatmap
)
from utils.evaluation import (evaluate_cf_rmse_mae, precision_recall_at_k, coverage, evaluate_content_based)

app = Flask(__name__)
app.secret_key = "anime-secret-key"

# ===========================
# LOAD DATA
# ===========================
DATA_PATH = "data/anime_clean.csv"
USER_FILE = "data/users.csv"
FAV_FILE = "data/favorites.csv"
RATING_FILE = "data/ratings.csv"

df = pd.read_csv(DATA_PATH)

# CF cũ: dùng favorites.csv cho trang /favorites và /admin
cf_model = CollaborativeFiltering(DATA_PATH, FAV_FILE)

# Model-based recommender (SVD + TF-IDF) dùng ratings.csv cho trang /ratings
rec_system = RecommenderSystem(
    anime_path=DATA_PATH,
    rating_path=RATING_FILE,
    model_dir="models"
)
# cố gắng load model đã train sẵn; nếu chưa có thì cứ để vậy, /ratings sẽ xử lý
try:
    rec_system.load()
    rec_system.load_data()
    print("✅ Loaded model-based recommender from /models")
except Exception as e:
    print("⚠ Không load được model-based recommender:", e)

# ensure files exist
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["user_id", "username", "password"]).to_csv(USER_FILE, index=False)

if not os.path.exists(FAV_FILE):
    pd.DataFrame(columns=["user_id", "anime_id"]).to_csv(FAV_FILE, index=False)

if not os.path.exists(RATING_FILE):
    pd.DataFrame(columns=["user_id", "anime_id", "rating"]).to_csv(RATING_FILE, index=False)


# Phân trang
def get_pagination(page, total_pages):
    pagination = []
    if page > 3:
        pagination.append(1)
    if page > 4:
        pagination.append("...")
    for p in range(page-2, page+3):
        if 1 <= p <= total_pages:
            pagination.append(p)
    if page < total_pages - 3:
        pagination.append("...")
    if page < total_pages - 2:
        pagination.append(total_pages)
    return pagination


@app.route("/")
def home():
    page = int(request.args.get("page", 1))
    per_page = 18

    total_items = len(df)
    total_pages = (total_items + per_page - 1) // per_page

    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page
    items = df.iloc[start:end]

    pagination = get_pagination(page, total_pages)

    return render_template(
        "home.html",
        animes=items.to_dict(orient="records"),
        page=page,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages,
        pagination=pagination
    )


# Chi tiết anime – GỢI Ý CONTENT-BASED
@app.route("/anime/<int:anime_id>")
def detail(anime_id):
    anime = df[df["id"] == anime_id]
    if anime.empty:
        return "Anime not found", 404

    anime = anime.iloc[0]

    # GỢI Ý BẰNG MODEL-BASED TF-IDF (KHÔNG DÙNG content_based.py CŨ)
    recommendations = []
    try:
        recommendations = rec_system.recommend_content(anime_id, top_k=10)
    except Exception as e:
        print("Content-based model failed:", e)

    # Favorite & rating check
    user_fav = False
    user_rating = None

    if "user_id" in session:
        user_id = session["user_id"]

        # FAVORITE
        fav_df = pd.read_csv(FAV_FILE)
        user_fav = ((fav_df["user_id"] == user_id) &
                    (fav_df["anime_id"] == anime_id)).any()

        # RATING
        if os.path.exists(RATING_FILE):
            rating_df = pd.read_csv(RATING_FILE)
            row = rating_df[(rating_df["user_id"] == user_id) &
                            (rating_df["anime_id"] == anime_id)]
            if not row.empty:
                user_rating = int(row.iloc[0]["rating"])

    return render_template(
        "detail.html",
        anime=anime,
        recommendations=recommendations,
        is_favorite=user_fav,
        user_rating=user_rating
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


# Thêm/Xóa yêu thích
@app.route("/favorite/<int:anime_id>")
def favorite(anime_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    fav_df = pd.read_csv(FAV_FILE)
    user_id = session["user_id"]

    exists = fav_df[(fav_df["user_id"] == user_id) &
                    (fav_df["anime_id"] == anime_id)]

    if exists.empty:
        fav_df.loc[len(fav_df)] = [user_id, anime_id]
    else:
        fav_df = fav_df[~((fav_df["user_id"] == user_id) &
                          (fav_df["anime_id"] == anime_id))]

    fav_df.to_csv(FAV_FILE, index=False)

    return redirect(url_for("detail", anime_id=anime_id))


# Yêu thích – vẫn dùng CF theo favorites.csv như cũ
@app.route("/favorites")
def favorites():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    cf_model.reload()  # CF theo favorites

    user_id = session["user_id"]

    df_fav = pd.read_csv(FAV_FILE)
    df_anime = pd.read_csv(DATA_PATH)

    fav_ids = df_fav[df_fav["user_id"] == user_id]["anime_id"].tolist()
    favorites = df_anime[df_anime["id"].isin(fav_ids)].to_dict(orient="records")

    recs = cf_model.recommend(user_id, top_k=10)

    return render_template("favorites.html",
                           favorites=favorites,
                           recommendations=recs)


# Đánh giá 1–10
@app.route("/rate/<int:anime_id>/<int:score>")
def rate(anime_id, score):
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if os.path.exists(RATING_FILE):
        df_rating = pd.read_csv(RATING_FILE)
    else:
        df_rating = pd.DataFrame(columns=["user_id", "anime_id", "rating"])

    mask = (df_rating["user_id"] == user_id) & (df_rating["anime_id"] == anime_id)

    if mask.any():
        df_rating.loc[mask, "rating"] = score       # update
    else:
        df_rating.loc[len(df_rating)] = [user_id, anime_id, score]   # insert

    df_rating.to_csv(RATING_FILE, index=False)
    return redirect(url_for("detail", anime_id=anime_id))


@app.route("/rate/delete/<int:anime_id>")
def delete_rating(anime_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not os.path.exists(RATING_FILE):
        return redirect(url_for("detail", anime_id=anime_id))

    df_rating = pd.read_csv(RATING_FILE)
    df_rating = df_rating[~((df_rating["user_id"] == user_id) &
                            (df_rating["anime_id"] == anime_id))]

    df_rating.to_csv(RATING_FILE, index=False)
    return redirect(url_for("detail", anime_id=anime_id))


# Lịch sử đánh giá + CF SVD dựa trên ratings.csv
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

    user_ratings = rating_df[rating_df["user_id"] == user_id]

    # merge ratings + anime info
    merged = user_ratings.merge(anime_df, left_on="anime_id", right_on="id", how="left")

    ratings = []
    for _, row in merged.iterrows():
        ratings.append({
            "id": int(row["anime_id"]),
            "title": row["title"],
            "image": row["image"],
            "genres": row["genres"],
            "rating": int(row["rating"])
        })

    # ----- CF model-based (SVD) -----
    recommendations = []

    recommendations = rec_system.recommend_cf(user_id, top_k=10)


    return render_template("ratings.html",
                           ratings=ratings,
                           recommendations=recommendations)


@app.route("/admin")
def admin_dashboard():

    df_anime = df
    df_users = pd.read_csv(USER_FILE)
    df_fav = pd.read_csv(FAV_FILE)
    df_rating = pd.read_csv(RATING_FILE)

    # ======= 1. BASIC STATS =========
    total_items = len(df_anime)
    total_users = len(df_users)
    total_fav = len(df_fav)
    total_rating = len(df_rating)

    users_with_rating = df_rating["user_id"].nunique()

    # ======= 2. RATING ANALYSIS ========
    # Top anime nhiều đánh giá nhất
    top_rated = (
        df_rating.groupby("anime_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )

    # merge info anime
    top_rated = top_rated.merge(df_anime[["id", "title", "image"]], left_on="anime_id", right_on="id")

    # Top điểm trung bình cao nhất (ít nhất 5 rating)
    avg_rating = (
        df_rating.groupby("anime_id")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    avg_rating = avg_rating[avg_rating["count"] >= 5]
    avg_rating = avg_rating.sort_values("mean", ascending=False).head(10)
    avg_rating = avg_rating.merge(df_anime[["id", "title", "image"]], left_on="anime_id", right_on="id")

    # ======= 3. MODEL STATUS ==========
    model_status = "Loaded" if rec_system.cf.user_factors is not None else "Not loaded"

    num_cf_users = len(rec_system.cf.user_id_to_idx)
    num_cf_items = len(rec_system.cf.item_id_to_idx)
    num_factors = rec_system.cf.n_factors

    # ======= 4. GRAPHS (REUSE FUNCTION) ==========
    score_img = score_distribution(df_anime)
    genres_img = top_genres(df_anime)
    fav_img = top_favorites(df_fav)
    heatmap_img = correlation_heatmap(df_anime)

    # ========== MODEL EVALUATION ==========

    df_rating = pd.read_csv(RATING_FILE)

    # CF (SVD) eval
    if len(df_rating) > 20:
        rmse, mae = evaluate_cf_rmse_mae(rec_system.cf, df_rating)
        cf_precision, cf_recall = precision_recall_at_k(rec_system.cf, df_rating, k=10)
        cf_cov = coverage(rec_system.cf, total_items)
    else:
        rmse, mae = 0, 0
        cf_precision, cf_recall = 0, 0
        cf_cov = 0

    # Content-based eval
    cb_precision, cb_recall = evaluate_content_based(rec_system, df, k=10)

    return render_template(
        "admin.html",

        # Basic stats
        total_items=total_items,
        total_users=total_users,
        total_fav=total_fav,
        total_rating=total_rating,
        users_with_rating=users_with_rating,

        # rating analytics
        top_rated=top_rated.to_dict(orient="records"),
        avg_rating=avg_rating.to_dict(orient="records"),

        # model stats
        model_status=model_status,
        num_cf_users=num_cf_users,
        num_cf_items=num_cf_items,
        num_factors=num_factors,

        # charts
        score_img=score_img,
        genres_img=genres_img,
        fav_img=fav_img,
        heatmap_img=heatmap_img,

        # full data (nếu admin cần)
        users=df_users.to_dict(orient="records"),
        anime=df_anime.to_dict(orient="records"),
        favorites=df_fav.to_dict(orient="records"),
        ratings=df_rating.to_dict(orient="records"),
        
        # evaluation
        rmse=rmse,
        mae=mae,
        cf_precision=cf_precision,
        cf_recall=cf_recall,
        cf_cov=cf_cov,
        cb_precision=cb_precision,
        cb_recall=cb_recall,
    )


# Đăng xuất
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
