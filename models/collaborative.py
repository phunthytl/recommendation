import numpy as np
import joblib, os


class CollaborativeFiltering:
    """
    Drop-in replacement for old SVD-based CF.

    Keeps the same public API:
    - fit(df_ratings)
    - predict(user_id, anime_id)
    - recommend(user_id, all_anime_ids, df_ratings, top_k)
    - save(path) / load(path)

    Adds:
    - partial_update(user_id, anime_id, rating)  -> for real-time updates
    """

    def __init__(self, n_factors=64, lr=0.01, reg=0.02, epochs=10, seed=42):
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.user_id_to_idx = {}
        self.idx_to_user_id = {}

        self.item_id_to_idx = {}
        self.idx_to_item_id = {}

        self.P = None  # (n_users, k)
        self.Q = None  # (n_items, k)
        self.bu = None  # (n_users,)
        self.bi = None  # (n_items,)

        self.global_mean = 0.0
        self._sum_ratings = 0.0
        self._count_ratings = 0

    # ------------------------
    # internal init helpers
    # ------------------------
    def _init_user_vec(self):
        return self.rng.normal(0, 0.1, self.n_factors), 0.0

    def _init_item_vec(self):
        return self.rng.normal(0, 0.1, self.n_factors), 0.0

    def _ensure_user(self, user_id):
        if user_id in self.user_id_to_idx:
            return self.user_id_to_idx[user_id]
        idx = len(self.user_id_to_idx)
        self.user_id_to_idx[user_id] = idx
        self.idx_to_user_id[idx] = user_id

        p, b = self._init_user_vec()
        self.P = p.reshape(1, -1) if self.P is None else np.vstack([self.P, p])
        self.bu = np.array([b], dtype=float) if self.bu is None else np.append(self.bu, b)
        return idx

    def _ensure_item(self, anime_id):
        if anime_id in self.item_id_to_idx:
            return self.item_id_to_idx[anime_id]
        idx = len(self.item_id_to_idx)
        self.item_id_to_idx[anime_id] = idx
        self.idx_to_item_id[idx] = anime_id

        q, b = self._init_item_vec()
        self.Q = q.reshape(1, -1) if self.Q is None else np.vstack([self.Q, q])
        self.bi = np.array([b], dtype=float) if self.bi is None else np.append(self.bi, b)
        return idx

    # ------------------------
    # public API
    # ------------------------
    def fit(self, df_ratings):
        df = df_ratings.copy()
        df = df[["user_id", "anime_id", "rating"]].dropna()

        self._sum_ratings = float(df["rating"].astype(float).sum())
        self._count_ratings = int(len(df))
        self.global_mean = self._sum_ratings / max(1, self._count_ratings)

        print(f"[CF/MFSGD] Global Mean = {self.global_mean:.3f}")
        data = df[["user_id", "anime_id", "rating"]].to_records(index=False)

        # reset model
        self.user_id_to_idx, self.idx_to_user_id = {}, {}
        self.item_id_to_idx, self.idx_to_item_id = {}, {}
        self.P = self.Q = self.bu = self.bi = None

        for ep in range(1, self.epochs + 1):
            np.random.shuffle(data)
            se = 0.0
            for u, i, r in data:
                err = self.partial_update(int(u), int(i), float(r), update_global_mean=False)
                se += err * err
            rmse = float(np.sqrt(se / len(data)))
            print(f"[CF/MFSGD] Epoch {ep}/{self.epochs} - Train RMSE: {rmse:.4f}")

        print("[CF/MFSGD] Done!")

    def predict(self, user_id, anime_id, clip=True):
        if user_id not in self.user_id_to_idx or anime_id not in self.item_id_to_idx:
            return np.nan

        uid = self.user_id_to_idx[user_id]
        iid = self.item_id_to_idx[anime_id]
        pred = (
            self.global_mean
            + self.bu[uid]
            + self.bi[iid]
            + float(np.dot(self.P[uid], self.Q[iid]))
        )

        if clip:
            pred = float(np.clip(pred, 1, 10))
        return float(pred)

    def partial_update(self, user_id, anime_id, rating, update_global_mean=True):
        """
        Online update 1 interaction (real-time).
        """
        user_id = int(user_id)
        anime_id = int(anime_id)
        rating = float(rating)

        if update_global_mean:
            self._sum_ratings += rating
            self._count_ratings += 1
            self.global_mean = self._sum_ratings / max(1, self._count_ratings)

        uid = self._ensure_user(user_id)
        iid = self._ensure_item(anime_id)

        pred = (
            self.global_mean
            + self.bu[uid]
            + self.bi[iid]
            + float(np.dot(self.P[uid], self.Q[iid]))
        )
        err = rating - pred

        # bias updates
        self.bu[uid] += self.lr * (err - self.reg * self.bu[uid])
        self.bi[iid] += self.lr * (err - self.reg * self.bi[iid])

        Pu = self.P[uid].copy()
        Qi = self.Q[iid].copy()

        # factor updates
        self.P[uid] += self.lr * (err * Qi - self.reg * Pu)
        self.Q[iid] += self.lr * (err * Pu - self.reg * Qi)

        return float(err)

    def recommend(self, user_id, all_anime_ids, df_ratings, top_k=10):
        """
        Keep signature compatible with old code.
        df_ratings is used only to filter items that user has already rated.
        """
        user_id = int(user_id)
        if user_id not in self.user_id_to_idx:
            return []

        seen = set(df_ratings[df_ratings["user_id"] == user_id]["anime_id"].astype(int).tolist())
        candidates = [int(aid) for aid in all_anime_ids if int(aid) not in seen and int(aid) in self.item_id_to_idx]
        if not candidates:
            return []

        uid = self.user_id_to_idx[user_id]
        cand_idx = np.array([self.item_id_to_idx[aid] for aid in candidates], dtype=int)

        # vectorized scoring
        pu = self.P[uid]  # (k,)
        scores = (
            self.global_mean
            + self.bu[uid]
            + self.bi[cand_idx]
            + (self.Q[cand_idx] @ pu)
        )

        # take top_k
        if len(scores) <= top_k:
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, top_k)[:top_k]
            order = order[np.argsort(-scores[order])]

        results = [(candidates[j], float(scores[j])) for j in order[:top_k]]
        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(
            {
                "n_factors": self.n_factors,
                "lr": self.lr,
                "reg": self.reg,
                "epochs": self.epochs,
                "seed": self.seed,
                "user_id_to_idx": self.user_id_to_idx,
                "idx_to_user_id": self.idx_to_user_id,
                "item_id_to_idx": self.item_id_to_idx,
                "idx_to_item_id": self.idx_to_item_id,
                "P": self.P,
                "Q": self.Q,
                "bu": self.bu,
                "bi": self.bi,
                "global_mean": self.global_mean,
                "_sum_ratings": self._sum_ratings,
                "_count_ratings": self._count_ratings,
            },
            os.path.join(path, "cf_model.joblib"),
        )
        print("[CF/MFSGD] Saved model!")

    def load(self, path):
        obj = joblib.load(os.path.join(path, "cf_model.joblib"))
        self.n_factors = obj["n_factors"]
        self.lr = obj["lr"]
        self.reg = obj["reg"]
        self.epochs = obj["epochs"]
        self.seed = obj["seed"]
        self.rng = np.random.default_rng(self.seed)

        self.user_id_to_idx = obj["user_id_to_idx"]
        self.idx_to_user_id = obj["idx_to_user_id"]
        self.item_id_to_idx = obj["item_id_to_idx"]
        self.idx_to_item_id = obj["idx_to_item_id"]

        self.P = obj["P"]
        self.Q = obj["Q"]
        self.bu = obj["bu"]
        self.bi = obj["bi"]

        self.global_mean = obj["global_mean"]
        self._sum_ratings = obj.get("_sum_ratings", float(self.global_mean))
        self._count_ratings = obj.get("_count_ratings", 1)

        print("[CF/MFSGD] Loaded model!")
