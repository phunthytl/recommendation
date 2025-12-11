# ===========================
# IMPORTS
# ===========================
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session

from models.recommend import RecommenderSystem
from utils.visualization import (
    score_distribution, top_genres, correlation_heatmap,
    rating_distribution, top_rated_count_chart
)
from utils.evaluation import (
    evaluate_cf_rmse_mae, precision_recall_at_k, coverage, evaluate_content_based
)

# ===========================
# CONFIGURATION
# ===========================
app = Flask(__name__)
app.secret_key = "anime-secret-key"

# Constants
DATA_PATH = "data/anime_clean.csv"
USER_FILE = "data/users.csv"
RATING_FILE = "data/ratings.csv"
MODEL_DIR = "models"
ITEMS_PER_PAGE = 24
TOP_K = 10
CHARTS_CACHE_FILE = ".charts_cache.json"
EVAL_CACHE_FILE = ".eval_cache.json"

# ===========================
# LOAD DATA
# ===========================
df = pd.read_csv(DATA_PATH)


# ===========================
# INITIALIZE RECOMMENDER SYSTEM
# ===========================
rec_system = RecommenderSystem(
    anime_path=DATA_PATH,
    rating_path=RATING_FILE,
    model_dir=MODEL_DIR
)

try:
    rec_system.load()
    rec_system.load_data()
    print("✅ Loaded model-based recommender from models/")
except Exception as e:
    print(f"⚠️  Warning: Could not load recommender models: {e}")


# ===========================
# ENSURE DATA FILES EXIST
# ===========================
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["user_id", "username", "password"]).to_csv(
        USER_FILE, index=False
    )
    print(f"✅ Created {USER_FILE}")

if not os.path.exists(RATING_FILE):
    pd.DataFrame(columns=["user_id", "anime_id", "rating"]).to_csv(
        RATING_FILE, index=False
    )
    print(f"✅ Created {RATING_FILE}")


# ===========================
# UTILITY FUNCTIONS
# ===========================
def get_pagination(page, total_pages):
    """Generate pagination list for navigation."""
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


def get_file_hash(filepath):
    """Get file modification time as a simple hash."""
    if not os.path.exists(filepath):
        return None
    return os.path.getmtime(filepath)


def get_charts_from_cache():
    """Load cached charts from file if data hasn't changed."""
    if not os.path.exists(CHARTS_CACHE_FILE):
        return None
    
    try:
        with open(CHARTS_CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        # Check if source data files have changed
        anime_hash = get_file_hash(DATA_PATH)
        rating_hash = get_file_hash(RATING_FILE)
        
        if (cache.get('anime_hash') == anime_hash and 
            cache.get('rating_hash') == rating_hash):
            return cache.get('charts')
    except:
        pass
    
    return None


def save_charts_to_cache(charts):
    """Save generated charts to cache file."""
    try:
        anime_hash = get_file_hash(DATA_PATH)
        rating_hash = get_file_hash(RATING_FILE)
        
        cache = {
            'charts': charts,
            'anime_hash': anime_hash,
            'rating_hash': rating_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(CHARTS_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass


def get_eval_from_cache():
    """Load cached evaluation metrics if data hasn't changed."""
    if not os.path.exists(EVAL_CACHE_FILE):
        return None
    
    try:
        with open(EVAL_CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        rating_hash = get_file_hash(RATING_FILE)
        
        if cache.get('rating_hash') == rating_hash:
            return cache.get('metrics')
    except:
        pass
    
    return None


def save_eval_to_cache(metrics):
    """Save evaluation metrics to cache file."""
    try:
        rating_hash = get_file_hash(RATING_FILE)
        
        cache = {
            'metrics': metrics,
            'rating_hash': rating_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(EVAL_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass


@app.route("/")
def home():
    """Display homepage with paginated anime list."""
    page = int(request.args.get("page", 1))
    per_page = ITEMS_PER_PAGE

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
        pagination=pagination,
    )


# Chi tiết anime – GỢI Ý CONTENT-BASED
@app.route("/anime/<int:anime_id>")
def detail(anime_id):
    """Display anime details with content-based recommendations."""
    anime = df[df["id"] == anime_id]
    if anime.empty:
        return "Anime not found", 404

    anime = anime.iloc[0]

    # Content-based recommendations using TF-IDF
    recommendations = []
    try:
        recommendations = rec_system.recommend_content(anime_id, top_k=TOP_K)
    except Exception as e:
        print(f"Content-based recommendation failed: {e}")

    # Check user rating
    user_fav = False
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
        is_favorite=user_fav,
        user_rating=user_rating,
    )


# ===========================
# AUTHENTICATION ROUTES
# ===========================
@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page and authentication."""
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


@app.route("/logout")
def logout():
    """User logout."""
    session.clear()
    return redirect(url_for("home"))


# ===========================
# RATING ROUTES
# ===========================
@app.route("/rate/<int:anime_id>/<int:score>")
def rate(anime_id, score):
    """Submit or update user rating for an anime."""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if os.path.exists(RATING_FILE):
        df_rating = pd.read_csv(RATING_FILE)
    else:
        df_rating = pd.DataFrame(columns=["user_id", "anime_id", "rating"])

    mask = (df_rating["user_id"] == user_id) & (df_rating["anime_id"] == anime_id)

    if mask.any():
        df_rating.loc[mask, "rating"] = score  # update existing
    else:
        df_rating.loc[len(df_rating)] = [user_id, anime_id, score]  # insert new

    df_rating.to_csv(RATING_FILE, index=False)
    return redirect(url_for("detail", anime_id=anime_id))


@app.route("/rate/delete/<int:anime_id>")
def delete_rating(anime_id):
    """Delete user rating for an anime."""
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



# ===========================
# RECOMMENDATION ROUTES
# ===========================
@app.route("/ratings")
def rating_history():
    """Display user's rating history and collaborative filtering recommendations."""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not os.path.exists(RATING_FILE):
        rating_df = pd.DataFrame(columns=["user_id", "anime_id", "rating"])
    else:
        rating_df = pd.read_csv(RATING_FILE)

    anime_df = df

    user_ratings = rating_df[rating_df["user_id"] == user_id]

    # Merge ratings with anime info
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

    recommendations = rec_system.recommend_cf(user_id, top_k=TOP_K)

    return render_template("ratings.html", ratings=ratings, recommendations=recommendations)


@app.route("/admin")
def admin_dashboard():
    df_anime = df.copy()
    df_users = pd.read_csv(USER_FILE)
    df_rating = pd.read_csv(RATING_FILE)

    # ========== BASIC STATISTICS ==========
    total_items = len(df_anime)
    total_users = len(df_users)
    total_rating = len(df_rating)
    users_with_rating = df_rating["user_id"].nunique()

    # ========== RATING ANALYSIS ==========
    # Top anime with most ratings
    top_rated = (
        df_rating.groupby("anime_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )

    top_rated = top_rated.merge(
        df_anime[["id", "title", "image"]], left_on="anime_id", right_on="id"
    )

    # Top average scores (min 5 ratings)
    avg_rating = (
        df_rating.groupby("anime_id")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    avg_rating = avg_rating[avg_rating["count"] >= 5]
    avg_rating = avg_rating.sort_values("mean", ascending=False).head(10)
    avg_rating = avg_rating.merge(
        df_anime[["id", "title", "image"]], left_on="anime_id", right_on="id"
    )

    # ========== MODEL STATUS ==========
    model_status = "Loaded" if rec_system.cf.user_factors is not None else "Not loaded"
    num_cf_users = len(rec_system.cf.user_id_to_idx)
    num_cf_items = len(rec_system.cf.item_id_to_idx)
    num_factors = rec_system.cf.n_factors

    # ========== VISUALIZATION CHARTS ==========
    # Try to load from cache first
    charts = get_charts_from_cache()
    
    if charts is None:
        # Cache miss - regenerate all charts
        print("📊 Regenerating charts (cache miss)...")
        score_img = score_distribution(df_anime)
        genres_img = top_genres(df_anime)
        heatmap_img = correlation_heatmap(df_anime)
        rating_dist_img = rating_distribution(df_rating)
        top_rated_count_img = top_rated_count_chart(df_rating, df_anime)
        
        # Save to cache
        charts = {
            'score_img': score_img,
            'genres_img': genres_img,
            'heatmap_img': heatmap_img,
            'rating_dist_img': rating_dist_img,
            'top_rated_count_img': top_rated_count_img
        }
        save_charts_to_cache(charts)
    else:
        # Cache hit - use cached charts
        print("⚡ Using cached charts")
        score_img = charts['score_img']
        genres_img = charts['genres_img']
        heatmap_img = charts['heatmap_img']
        rating_dist_img = charts['rating_dist_img']
        top_rated_count_img = charts['top_rated_count_img']

    # ========== MODEL EVALUATION ==========
    # Try to load from cache first
    eval_metrics = get_eval_from_cache()
    
    if eval_metrics is None:
        # Cache miss - recalculate
        print("📈 Calculating evaluation metrics...")
        if len(df_rating) > 20:
            rmse, mae = evaluate_cf_rmse_mae(rec_system.cf, df_rating)
            cf_precision, cf_recall = precision_recall_at_k(rec_system.cf, df_rating, k=10)
            cf_cov = coverage(rec_system.cf, total_items)
        else:
            rmse, mae = 0, 0
            cf_precision, cf_recall = 0, 0
            cf_cov = 0

        cb_precision, cb_recall = evaluate_content_based(rec_system, df, k=10)
        
        eval_metrics = {
            'rmse': float(rmse) if rmse is not None else 0,
            'mae': float(mae) if mae is not None else 0,
            'cf_precision': float(cf_precision),
            'cf_recall': float(cf_recall),
            'cf_cov': float(cf_cov),
            'cb_precision': float(cb_precision),
            'cb_recall': float(cb_recall)
        }
        save_eval_to_cache(eval_metrics)
    else:
        print("⚡ Using cached evaluation metrics")
    
    rmse = eval_metrics['rmse']
    mae = eval_metrics['mae']
    cf_precision = eval_metrics['cf_precision']
    cf_recall = eval_metrics['cf_recall']
    cf_cov = eval_metrics['cf_cov']
    cb_precision = eval_metrics['cb_precision']
    cb_recall = eval_metrics['cb_recall']

    return render_template(
        "admin.html",
        # Basic statistics
        total_items=total_items,
        total_users=total_users,
        total_rating=total_rating,
        users_with_rating=users_with_rating,
        # Rating analytics
        top_rated=top_rated.to_dict(orient="records"),
        avg_rating=avg_rating.to_dict(orient="records"),
        # Model statistics
        model_status=model_status,
        num_cf_users=num_cf_users,
        num_cf_items=num_cf_items,
        num_factors=num_factors,
        # Charts
        score_img=score_img,
        genres_img=genres_img,
        heatmap_img=heatmap_img,
        rating_dist_img=rating_dist_img,
        top_rated_count_img=top_rated_count_img,
        # Full data
        users=df_users.to_dict(orient="records"),
        anime=df_anime.to_dict(orient="records"),
        ratings=df_rating.to_dict(orient="records"),
        # Evaluation metrics
        rmse=rmse,
        mae=mae,
        cf_precision=cf_precision,
        cf_recall=cf_recall,
        cf_cov=cf_cov,
        cb_precision=cb_precision,
        cb_recall=cb_recall,
    )


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)