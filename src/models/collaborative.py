import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class CollaborativeRecommender:
    """
    Collaborative filtering recommender using matrix factorization (SVD).
    This implementation uses scipy's SVD and doesn't require scikit-surprise.
    """
    
    def __init__(self, model_path=None, movies_path=None, ratings_path=None, n_factors=50):
        """
        Initialize the collaborative recommender.
        
        Args:
            model_path: Path to pre-trained model (optional, for backward compatibility)
            movies_path: Path to movies CSV file
            ratings_path: Path to ratings CSV file (required for training)
            n_factors: Number of latent factors for SVD
        """
        self.n_factors = n_factors
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_id_map = None
        self.movie_id_map = None
        self.reverse_movie_map = None
        
        # Load movies dataframe
        if movies_path and os.path.exists(movies_path):
            self.movies_df = pd.read_csv(movies_path)
        else:
            raise FileNotFoundError(f"Movies file not found at: {movies_path}")
        
        # Try to load pre-trained scipy model first
        scipy_model_path = model_path.replace('.pkl', '_scipy.pkl') if model_path else None
        
        if scipy_model_path and os.path.exists(scipy_model_path):
            self._load_model(scipy_model_path)
        elif ratings_path and os.path.exists(ratings_path):
            # Train new model from ratings
            ratings_df = pd.read_csv(ratings_path)
            self._train(ratings_df)
            # Save the trained model
            if scipy_model_path:
                self._save_model(scipy_model_path)
        else:
            # Try default ratings path
            default_ratings = 'data/raw/ratings.csv'
            if os.path.exists(default_ratings):
                ratings_df = pd.read_csv(default_ratings)
                self._train(ratings_df)
                if scipy_model_path:
                    self._save_model(scipy_model_path)
            else:
                raise FileNotFoundError("No model or ratings data found to train collaborative filtering")
    
    def _train(self, ratings_df):
        """Train the SVD model on ratings data."""
        # Create user and movie mappings
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.reverse_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        n_users = len(unique_users)
        n_movies = len(unique_movies)
        
        # Create sparse user-item matrix
        row_indices = ratings_df['userId'].map(self.user_id_map).values
        col_indices = ratings_df['movieId'].map(self.movie_id_map).values
        ratings = ratings_df['rating'].values
        
        self.global_mean = ratings.mean()
        
        # Create sparse matrix
        user_item_matrix = csr_matrix((ratings, (row_indices, col_indices)), 
                                       shape=(n_users, n_movies))
        
        # Convert to dense and center the ratings
        matrix_dense = user_item_matrix.toarray()
        
        # Replace zeros with global mean for SVD
        matrix_dense[matrix_dense == 0] = self.global_mean
        matrix_centered = matrix_dense - self.global_mean
        
        # Perform SVD
        k = min(self.n_factors, min(n_users, n_movies) - 1)
        U, sigma, Vt = svds(csr_matrix(matrix_centered), k=k)
        
        # Store factors
        sigma_diag = np.diag(sigma)
        self.user_factors = U @ sigma_diag
        self.item_factors = Vt.T
        
        self.model = True  # Flag that model is trained
    
    def _save_model(self, path):
        """Save the trained model to disk."""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'global_mean': self.global_mean,
            'user_id_map': self.user_id_map,
            'movie_id_map': self.movie_id_map,
            'reverse_movie_map': self.reverse_movie_map
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_model(self, path):
        """Load a pre-trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.global_mean = model_data['global_mean']
        self.user_id_map = model_data['user_id_map']
        self.movie_id_map = model_data['movie_id_map']
        self.reverse_movie_map = model_data['reverse_movie_map']
        self.model = True
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair."""
        if user_id not in self.user_id_map:
            return self.global_mean
        if movie_id not in self.movie_id_map:
            return self.global_mean
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        pred = self.global_mean + np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
        # Clip to valid rating range
        return np.clip(pred, 0.5, 5.0)
        
    def recommend(self, user_id, top_n=10):
        """
        Recommend movies for a given user based on collaborative filtering.
        """
        if user_id not in self.user_id_map:
            # For unknown users, return popular movies
            return self._recommend_popular(top_n)
        
        user_idx = self.user_id_map[user_id]
        
        # Compute predictions for all movies
        predictions = self.global_mean + np.dot(self.user_factors[user_idx], self.item_factors.T)
        
        # Get top N movie indices
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        # Prepare results
        results = []
        for idx in top_indices:
            movie_id = self.reverse_movie_map[idx]
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                movie_title = movie_row['title'].values[0]
                rating = np.clip(predictions[idx], 0.5, 5.0)
                results.append({'Title': movie_title, 'Estimated Rating': round(rating, 2)})
            
        return pd.DataFrame(results)
    
    def _recommend_popular(self, top_n=10):
        """Fallback: recommend popular movies for unknown users."""
        # Just return first N movies as fallback
        results = []
        for _, row in self.movies_df.head(top_n).iterrows():
            results.append({'Title': row['title'], 'Estimated Rating': 4.0})
        return pd.DataFrame(results)