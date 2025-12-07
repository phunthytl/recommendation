import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set UTF-8 encoding for stdout (fixes emoji issues on Windows)
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from models.collaborative import CollaborativeFiltering
from utils.evaluation import precision_recall_at_k_cf, coverage

df_fav = pd.read_csv('data/favorites.csv')
df_anime = pd.read_csv('data/anime_clean.csv')

cf = CollaborativeFiltering()   # will load data from default paths
prec, rec = precision_recall_at_k_cf(cf, df_fav, k=10)
cov = coverage(cf, total_items=len(df_anime))

print()
print('Results:')
print('Precision@10:', prec)
print('Recall@10:', rec)
print('Coverage:', cov)