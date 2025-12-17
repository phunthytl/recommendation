import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.collaborative import CollaborativeFiltering
from models.content_based import ContentBased

class RecommenderSystem:
    def __init__(self, anime_path="data/anime_clean.csv", rating_path="data/ratings.csv", model_dir="models"):
        self.anime_path = anime_path
        self.rating_path = rating_path
        self.model_dir = model_dir
        self.df_anime = None  # DataFrame chứa dữ liệu anime
        self.df_ratings = None # DataFrame chứa dữ liệu rating

        # Khởi tạo mô hình CF
        self.cf = CollaborativeFiltering(n_factors=64, lr=0.01, reg=0.02, epochs=10, seed=42)
        # Khởi tạo mô hình Content-Based
        self.content = ContentBased(max_features=7000)

        # Thời điểm sửa file ratings.csv gần nhất để theo dõi cập nhật realtime
        self._ratings_mtime = None
        # Số dòng ratings hiện có trong bộ nhớ
        self._ratings_len = 0

    # Load dữ liệu anime và rating từ file CSV
    def load_data(self):
        self.df_anime = pd.read_csv(self.anime_path) # Đọc dữ liệu anime
        self.df_ratings = pd.read_csv(self.rating_path) # Đọc dữ liệu rating

        # Khởi tạo thông tin theo dõi thay đổi file rating
        try:
            self._ratings_mtime = os.path.getmtime(self.rating_path)
        except Exception:
            self._ratings_mtime = None
        self._ratings_len = len(self.df_ratings)

    # Kiểm tra ratings.csv có bị cập nhật và áp dụng online update cho CF
    def _refresh_ratings_and_update_cf(self):
        try:
            mtime = os.path.getmtime(self.rating_path)
        except Exception:
            return

        # Lần đầu chưa có mtime thì chỉ lưu lại
        if self._ratings_mtime is None:
            self._ratings_mtime = mtime
            return

        # Không có thay đổi file
        if mtime == self._ratings_mtime:
            return

        # File thay đổi thì reload dữ liệu rating
        new_df = pd.read_csv(self.rating_path)
        new_len = len(new_df)

        # Trường hợp file bị ghi đè hoặc giảm dòng thì reload toàn bộ
        if new_len < self._ratings_len:
            self.df_ratings = new_df
            self._ratings_len = new_len
            self._ratings_mtime = mtime
            return

        # Trường hợp chỉ append thêm rating mới
        if new_len > self._ratings_len:
            appended = new_df.iloc[self._ratings_len:new_len][["user_id", "anime_id", "rating"]]
            # Áp dụng cập nhật online từng rating mới cho CF
            for _, r in appended.iterrows():
                self.cf.partial_update(int(r["user_id"]), int(r["anime_id"]), float(r["rating"]))
            # Cập nhật dữ liệu rating trong bộ nhớ
            self.df_ratings = new_df
            self._ratings_len = new_len

        # Cập nhật mtime mới nhất
        self._ratings_mtime = mtime

    # Huấn luyện toàn bộ hệ thống recommender
    def train(self):
        # Load dữ liệu trước khi train
        self.load_data()

        # Huấn luyện mô hình Collaborative Filtering
        print("Bắt đầu huấn luyện Collaborative Filtering (MF-SGD)...")
        self.cf.fit(self.df_ratings)

        # Huấn luyện mô hình Content-Based
        print("Bắt đầu huấn luyện Content-Based (TF-IDF)...")
        self.content.fit(self.df_anime)

    # Lưu toàn bộ model CF và Content-Based
    def save(self):
        self.cf.save(self.model_dir)
        self.content.save(self.model_dir)

    # Load model đã huấn luyện từ thư mục model
    def load(self):
        # Load model CF và Content-Based
        self.cf.load(self.model_dir)
        self.content.load(self.model_dir)

        # Load lại dữ liệu để phục vụ recommend
        self.load_data()

    # Gợi ý anime theo Collaborative Filtering với hỗ trợ realtime
    def recommend_cf(self, user_id, top_k=10):
        # Kiểm tra và cập nhật CF nếu có rating mới
        self._refresh_ratings_and_update_cf()

        # Lấy danh sách toàn bộ anime_id
        all_ids = self.df_anime["id"].astype(int).tolist()
        # Sinh danh sách gợi ý từ mô hình CF
        pairs = self.cf.recommend(int(user_id), all_ids, self.df_ratings, top_k)
        return self._info(pairs)

    # Gợi ý anime tương tự dựa trên Content-Based
    def recommend_content(self, anime_id, top_k=10):
        pairs = self.content.similar_items(int(anime_id), top_k)
        return self._info(pairs)

    # Bổ sung thông tin anime cho kết quả gợi ý
    def _info(self, pairs):
        results = []
        for aid, score in pairs:
            row = self.df_anime[self.df_anime["id"] == int(aid)]
            if row.empty:
                continue
            r = row.iloc[0]
            results.append({
                "id": int(r["id"]),
                "title": r["title"],
                "image": r.get("image", ""),
                "genres": r.get("genres", ""),
                "score": float(score)
            })
        return results


# Chạy train và save model khi file được chạy trực tiếp
if __name__ == "__main__":
    rec = RecommenderSystem()
    rec.train()
    rec.save()
    print("Đã lưu toàn bộ model.")
