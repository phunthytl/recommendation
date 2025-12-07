import requests
import pandas as pd
import time

TARGET = 2000
all_items = []
page = 1

print("🚀 Bắt đầu crawl 2000 anime...")

while len(all_items) < TARGET:
    url = f"https://api.jikan.moe/v4/anime?page={page}"

    print(f"🔎 Crawling page {page}... (current: {len(all_items)})")

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
        item = {
            # ID tạm, lát nữa bỏ
            "mal_id": a["mal_id"],
            "title": a.get("title", ""),
            "type": a.get("type", ""),
            "episodes": a.get("episodes", 0),
            "status": a.get("status", ""),
            "score": a.get("score", 0.0),
            "rank": a.get("rank", 0),
            "popularity": a.get("popularity", 0),
            "favorites": a.get("favorites", 0),
            "synopsis": a.get("synopsis", "") or "",
            "genres": ", ".join([g["name"] for g in a.get("genres", [])]),
            "image": a["images"]["jpg"]["large_image_url"],
        }

        all_items.append(item)

        if len(all_items) >= TARGET:
            break

    has_next = res["pagination"]["has_next_page"]
    if not has_next:
        print("⛔ API báo hết trang.")
        break

    page += 1
    time.sleep(0.4)

# ============================
# Chuẩn hóa ID 0 → N-1
# ============================

df = pd.DataFrame(all_items)

# Bỏ mal_id hoàn toàn nếu bạn không cần
df = df.drop(columns=["mal_id"])

# Tạo id mới liên tục
df = df.reset_index(drop=True)
df["id"] = df.index

# Trả id lên đầu
cols = ["id"] + [c for c in df.columns if c != "id"]
df = df[cols]

df.to_csv("anime_data.csv", index=False, encoding="utf-8-sig")

print("\n🎉 DONE! Collected", len(df), "anime → saved to anime_final.csv")
print("📁 File xuất: anime_final.csv")
