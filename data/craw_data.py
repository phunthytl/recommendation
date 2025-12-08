import requests
import pandas as pd
import time

TARGET = 3000
all_items = []
page = 1

print("🚀 Bắt đầu crawl 3000 anime (đã lọc 18+)...")


# ============================
# DANH SÁCH THỂ LOẠI CẤM
# ============================
BLOCK_GENRES = {
    "Hentai", "Ecchi", "Erotica", "Adult",
    "Yaoi", "Yuri", "Boys Love", "Girls Love",
    "Shounen Ai", "Shoujo Ai"
}

# TỪ KHÓA CẤM TRONG NỘI DUNG
BLOCK_KEYWORDS = [
    "hentai", "ecchi", "adult", "nsfw", "mature",
    "erotic", "yaoi", "yuri", "bl", "gl"
]


def is_blocked(title, synopsis, genres_text):
    """Trả về True nếu anime chứa nội dung nhạy cảm."""
    text = (title + " " + synopsis + " " + genres_text).lower()

    # chặn theo từ khóa
    for kw in BLOCK_KEYWORDS:
        if kw in text:
            return True

    # chặn theo thể loại
    for g in genres_text.split(","):
        if g.strip() in BLOCK_GENRES:
            return True

    return False


# ============================
# BẮT ĐẦU CRAWL
# ============================
while len(all_items) < TARGET:
    url = f"https://api.jikan.moe/v4/anime?page={page}"

    print(f"🔎 Crawling page {page}... (current valid: {len(all_items)})")

    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Error:", response.status_code)
        time.sleep(1)
        continue

    res = response.json()
    anime_list = res.get("data", [])

    if not anime_list:
        print("⛔ No more data từ API. Dừng lại.")
        break

    for a in anime_list:
        title = a.get("title", "")
        synopsis = a.get("synopsis", "") or ""
        genres_text = ", ".join([g["name"] for g in a.get("genres", [])])

        # 🛑 LỌC NỘI DUNG NHẠY CẢM
        if is_blocked(title, synopsis, genres_text):
            continue

        # ✔ Anime hợp lệ → lưu
        item = {
            "mal_id": a["mal_id"],
            "title": title,
            "type": a.get("type", ""),
            "episodes": a.get("episodes", 0),
            "status": a.get("status", ""),
            "score": a.get("score", 0.0),
            "rank": a.get("rank", 0),
            "popularity": a.get("popularity", 0),
            "favorites": a.get("favorites", 0),
            "synopsis": synopsis,
            "genres": genres_text,
            "image": a["images"]["jpg"]["large_image_url"],
        }

        all_items.append(item)

        if len(all_items) >= TARGET:
            break

    if not res["pagination"]["has_next_page"]:
        print("⛔ API báo hết trang.")
        break

    page += 1
    time.sleep(0.4)


# ============================
# CHUẨN HÓA ID
# ============================

df = pd.DataFrame(all_items)

df = df.drop(columns=["mal_id"])

df = df.reset_index(drop=True)
df["id"] = df.index + 1

cols = ["id"] + [c for c in df.columns if c != "id"]
df = df[cols]

df.to_csv("data/anime_data.csv", index=False, encoding="utf-8-sig")

print("\n🎉 DONE! Collected", len(df), "anime sạch → saved to anime_data.csv")
