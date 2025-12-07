import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBased:
    def __init__(self, df):
        """
        df: DataFrame đã load từ anime_clean.csv
        """
        self.df = df.reset_index(drop=True)

        # tạo map id → index
        self.id_to_index = {int(row["id"]): idx for idx, row in self.df.iterrows()}

        # chuẩn hóa genres
        self.df["genres_join"] = self.df["genres_list"].apply(
            lambda x: " ".join(eval(x)) if isinstance(x, str) else ""
        )

        # xử lý synopsis
        self.df["synopsis_clean"] = self.df["synopsis_clean"].fillna("")

        # TF-IDF cho mô tả
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.synopsis_matrix = self.tfidf.fit_transform(self.df["synopsis_clean"])

        # TF-IDF cho thể loại
        self.genre_vec = TfidfVectorizer()
        self.genre_matrix = self.genre_vec.fit_transform(self.df["genres_join"])

        # ghép vector nội dung + thể loại
        self.combined = np.hstack([
            self.synopsis_matrix.toarray(),
            self.genre_matrix.toarray()
        ])

        # ma trận similarity
        self.sim = cosine_similarity(self.combined)

    def recommend(self, anime_id, top_k=6):
        """
        Trả về danh sách anime_id được gợi ý
        """
        anime_id = int(anime_id)
        if anime_id not in self.id_to_index:
            return []

        idx = self.id_to_index[anime_id]

        scores = list(enumerate(self.sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # bỏ chính bản thân anime
        scores = scores[1 : top_k + 1]

        result_ids = [int(self.df.iloc[i]["id"]) for i, _ in scores]
        return result_ids
