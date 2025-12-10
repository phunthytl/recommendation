import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.collaborative import CollaborativeFiltering
from utils.evaluation import (
    evaluate_cf_rmse_mae,
    precision_recall_at_k,

)

print("\n🚀 Loading ratings.csv...\n")
df = pd.read_csv("data/ratings.csv")

cf = CollaborativeFiltering(n_factors=15)

print("🔧 Evaluating RMSE / MAE ...")
rmse, mae = evaluate_cf_rmse_mae(cf, df)
print(f"📌 RMSE : {rmse:.4f}")
print(f"📌 MAE  : {mae:.4f}")

print("\n🎯 Evaluating Precision@10 / Recall@10 ...")
p10, r10 = precision_recall_at_k(cf, df, k=10)
print(f"📌 Precision@10 : {p10:.4f}")
print(f"📌 Recall@10    : {r10:.4f}")

print("\n🎉 DONE!")