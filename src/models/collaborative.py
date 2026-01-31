import pandas as pd
import pickle
import os

class CollaborativeRecommender:
    def __init__(self, model_path, movies_path):
        # load the pre-trained model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f" Model not found at: {model_path}")

        #load the movies dataframe
        if os.path.exists(movies_path):
            self.movies_df = pd.read_csv(movies_path)
        else:
            raise FileNotFoundError(f" Movies file not found at: {movies_path}")
        
    def recommend(self, user_id, top_n=10):
        """
        this function recommends movies for a given user based on collaborative filtering.
        """
        # get all movie IDs
        all_movie_ids = self.movies_df['movieId'].unique()
        
        predictions = []
        
        # ask the model to predict the user's rating for each movie
        for movie_id in all_movie_ids:
            # get prediction
            pred = self.model.predict(uid=user_id, iid=movie_id)
            predictions.append((movie_id, pred.est))
        
        # sort predictions from highest to lowest
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # take the top N predictions
        top_predictions = predictions[:top_n]
        
        # prepare results for display
        results = []
        for movie_id, rating in top_predictions:
            movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values[0]
            results.append({'Title': movie_title, 'Estimated Rating': round(rating, 2)})
            
        return pd.DataFrame(results)