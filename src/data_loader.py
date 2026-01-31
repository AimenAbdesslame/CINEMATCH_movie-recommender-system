import pandas as pd
import os



class DataLoader:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.dirname(current_dir) 
        
        self.movies_path = os.path.join(project_root, 'data', 'raw', 'movies.csv')
        self.ratings_path = os.path.join(project_root, 'data', 'raw', 'ratings.csv')

    def load_data(self):

        print("Loading Data...")
        
    
        movies = pd.read_csv(self.movies_path)
        ratings = pd.read_csv(self.ratings_path)

        # 2 calcuate the mean and count of rating for each movie : 
        print(" Aggregating Ratings...")
        movie_stats = ratings.groupby('movieId').agg({'rating': ['mean','count']})
        movie_stats.columns = ["average_rating","vote_count"]

        # 3.merging the movies df with the new movie_stats df : 
        # ندمج معلومات الفيلم (الاسم) مع إحصائياته (التقييم)
        df_merged =  movies.merge(movie_stats , on = 'movieId' , how = "left")
        # 4. تنظيف القيم الفارغة
        df_merged['average_rating'] = df_merged['average_rating'].fillna(0)
        # الأفلام الجديدة التي ليس لها تقييم نضع لها 0 بدلاً من NaN
        df_merged['vote_count'] = df_merged['vote_count'].fillna(0)
        
   

        print(f" Data Loaded! Total Movies: {len(df_merged)}")
        return movies, ratings, df_merged

# هذا الجزء للتشغيل التجريبي فقط (Test Run)
if __name__ == "__main__":
    loader = DataLoader()
    m, r, df = loader.load_data()
    print("\n---first 5 rows ---")
    print(df[['title', 'average_rating', 'vote_count']].head())