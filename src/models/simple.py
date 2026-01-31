import pandas as pd 
import os 
import sys 

#importing the file in dataloader : 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_loader import DataLoader

class SimpleRecommender:
    def __init__(self, df):
        self.df = df
        
    def recommend(self , top_n = 10) : 
        # calculate the average rating across all movies : 
        global_avg = self.df['average_rating'].mean() 
        # taking the movie that have the +90 % of vote count :
        threshold = self.df["vote_count"].quantile(0.9)
        
        # the new dataframe containing only the movies that pass the threshold :
        movies = self.df[self.df["vote_count"] >= threshold] 
        
        
        def weight_calculation(movie) : 
            V = movie['vote_count']
            R = movie['average_rating']
            C = global_avg
            # weighted rating formula
            WR = (V / (V + 1000)) * R + (1000 / (V + 1000)) * C
            return WR
        
        movies["score"] = movies.apply(weight_calculation , axis = 1)
        movies = movies.sort_values("score" , ascending = False)
        
        return movies[['title' , 'vote_count', 'average_rating' , 'score']].head(top_n)
    
    
    
    
    
    
if __name__ == "__main__":
    print("DATA LOADING .... ")
    loader = DataLoader()
    _, _, data = loader.load_data()

    print("CALCULATING RECOMMENDATIONS...")
    model = SimpleRecommender(data)
    top_movies = model.recommend(10)

    print("\nTOP 10 MOVIES (BY POPULARITY AND QUALITY):")
    print(top_movies)