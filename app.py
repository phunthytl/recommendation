from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64

# models
from models.content_based import ContentBased
from models.collaborative import CollaborativeFiltering

from utils.visualization import (score_distribution, top_genres, top_favorites,  correlation_heatmap)
from utils.evaluation import (precision_recall_at_k_cf, precision_recall_at_k_cb, coverage)
app = Flask(__name__)
app.secret_key = "anime-secret-key"

# ===========================
# LOAD DATA
# ===========================
DATA_PATH = "data/anime_clean.csv"
USER_FILE = "data/users.csv"
FAV_FILE = "data/favorites.csv"

df = pd.read_csv(DATA_PATH)
cb_model = ContentBased(df)
cf_model = CollaborativeFiltering(DATA_PATH, FAV_FILE)

# ensure files exist
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["user_id", "username", "password"]).to_csv(USER_FILE, index=False)

if not os.path.exists(FAV_FILE):
    pd.DataFrame(columns=["user_id", "anime_id"]).to_csv(FAV_FILE, index=False)


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


# Chi tiết anime
@app.route("/anime/<int:anime_id>")
def detail(anime_id):
    anime = df[df["id"] == anime_id]
    if anime.empty:
        return "Anime not found", 404

    anime = anime.iloc[0]

    rec_ids = cb_model.recommend(anime_id, top_k=10)
    rec_df = df[df["id"].isin(rec_ids)]

    # Kiểm tra xem user đã favorite chưa
    user_fav = False
    if "user_id" in session:
        fav_df = pd.read_csv(FAV_FILE)
        user_fav = ((fav_df["user_id"] == session["user_id"]) &
                    (fav_df["anime_id"] == anime_id)).any()

    return render_template(
        "detail.html",
        anime=anime,
        recommendations=rec_df.to_dict(orient="records"),
        is_favorite=user_fav
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


# Yêu thích
@app.route("/favorites")
def favorites():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    cf_model.reload()

    user_id = session["user_id"]

    df_fav = pd.read_csv(FAV_FILE)
    df_anime = pd.read_csv(DATA_PATH)

    fav_ids = df_fav[df_fav["user_id"] == user_id]["anime_id"].tolist()
    favorites = df_anime[df_anime["id"].isin(fav_ids)].to_dict(orient="records")

    recs = cf_model.recommend(user_id, top_k=10)

    return render_template("favorites.html",
                           favorites=favorites,
                           recommendations=recs)


# Admin
@app.route("/admin")
def admin_dashboard():
    cf_model.reload()

    df_anime = df
    df_users = pd.read_csv(USER_FILE)
    df_fav = pd.read_csv(FAV_FILE)

    # Stats
    total_items = len(df_anime)
    total_users = len(df_users)
    total_fav = len(df_fav)

    all_genres = set()
    for g in df_anime["genres"].dropna():
        all_genres.update([x.strip() for x in g.split(",")])
    total_genres = len(all_genres)

    # Biểu đồ
    score_img = score_distribution(df_anime)
    genres_img = top_genres(df_anime)
    fav_img = top_favorites(df_fav)
    heatmap_img = correlation_heatmap(df_anime)

    # Đánh giá model
    cf_precision, cf_recall = precision_recall_at_k_cf(cf_model, df_fav, k=10)
    cb_precision, cb_recall = precision_recall_at_k_cb(cb_model, df_anime, k=10)

    # COVERAGE
    cf_cov = coverage(cf_model, len(df_anime))

    return render_template(
        "admin.html",
        total_items=total_items,
        total_users=total_users,
        total_fav=total_fav,
        total_genres=total_genres,

        # charts
        score_img=score_img,
        genres_img=genres_img,
        fav_img=fav_img,
        heatmap_img=heatmap_img,

        # CF basic
        cf_precision=cf_precision,
        cf_recall=cf_recall,

        # CBF basic
        cb_precision=cb_precision,
        cb_recall=cb_recall,

        # Coverage
        cf_cov=cf_cov,

        users=df_users.to_dict(orient="records"),
        anime=df_anime.to_dict(orient="records"),
        favorites=df_fav.to_dict(orient="records"),
    )


# Đăng xuất
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
