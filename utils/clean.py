import pandas as pd
import numpy as np
import re
import json

# Đọc dữ liệu anime
df = pd.read_csv("data/anime_data.csv")
print("Kích thước dữ liệu ban đầu:", df.shape)

# Loại bỏ các anime bị trùng tên để tránh nhiễu dữ liệu
df = df.drop_duplicates(subset=["title"])
print("Đã loại bỏ trùng lặp. Kích thước mới:", df.shape)

# Điền giá trị rỗng cho synopsis để tránh lỗi xử lý text
df["synopsis"] = df["synopsis"].fillna("")

# Điền giá trị rỗng cho genres để dễ tách thể loại
df["genres"] = df["genres"].fillna("")

# Điền số tập bị thiếu bằng 0 và ép kiểu int
df["episodes"] = df["episodes"].fillna(0).astype(int)

# Điền score bị thiếu bằng giá trị trung bình toàn bộ dataset
df["score"] = df["score"].fillna(df["score"].mean())

# Điền favorites bị thiếu bằng 0
df["favorites"] = df["favorites"].fillna(0)
print("Đã xử lý các giá trị bị thiếu.")

# Hàm tách chuỗi genres thành danh sách thể loại
def clean_genres(g):
    if isinstance(g, str):
        return [x.strip() for x in g.split(",") if x.strip() != ""]
    return []

# Tách genres để tạo cột genres_list
df["genres_list"] = df["genres"].apply(clean_genres)
print("Đã làm sạch genres và tạo cột genres_list.")

# Hàm làm sạch văn bản synopsis
def clean_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = txt.replace("(Source: ANN)", "")
    txt = txt.replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

# Tạo cột synopsis_clean đã được làm sạch
df["synopsis_clean"] = df["synopsis"].apply(clean_text)
print("Đã làm sạch nội dung synopsis.")

# Xóa cột synopsis gốc để giảm dung lượng và trùng lặp
df = df.drop(columns=["synopsis"])

# Danh sách các cột số cần chuẩn hóa
numeric_cols = ["episodes", "score", "rank", "popularity", "favorites"]

# Ép kiểu numeric và xử lý dữ liệu không hợp lệ
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

print("Đã chuẩn hóa các cột dữ liệu số.")

# Lưu dữ liệu đã làm sạch ra
df.to_csv("data/anime_clean.csv", index=False, encoding="utf-8-sig")
print("Đã lưu file anime_clean.csv.")

json_path = "data/anime_clean.json"
df.to_json(json_path, orient="records", force_ascii=False, indent=2)
print("Đã lưu file anime_clean.json.")

print("Hoàn tất quá trình làm sạch và xuất dữ liệu.")
