import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle
import os

# Define paths
ratings_path = os.path.join('data', 'raw', 'ratings.csv')
model_path = os.path.join('saved_models', 'svd_model_optimized.pkl')

print("Loading ratings...")
# Load ratings
df = pd.read_csv(ratings_path)

# Prepare for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

print("Training SVD Model (this may take a moment)...")
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

print(f"Saving model to {model_path}...")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(algo, f)

print("Done! You can now run the app.")
