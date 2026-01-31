import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentRecommender:
    def __init__(self, df):
        self.df = df

        #Data Cleaning : Replacing nan and replacing | with space
        self.df['genres'] = self.df['genres'].fillna('').str.replace("|", " ")       

        # construct the model upon initialization
        tfidf = TfidfVectorizer(stop_words='english')
        
        #appling the model to the genres column and converting it to matrix
        tfidf_matrix = tfidf.fit_transform(self.df['genres'])
        
        # matrix shape : 
        print("TF-IDF matrix shape:", tfidf_matrix.shape) 

        # similarity matrix calculation
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # series to map movie titles to indices
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        

    def recommend(self, title, n=10):
        #ensure that the movie exist :
        if title not in self.indices:
            return [f"sorry'{title}' not found in database"]

        # get the index of the movie that matches the title : 
        idx = self.indices[title]

        # getting the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # make a sort based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        #taking similar movies scores (excluding the first one which is the movie itself)
        sim_scores = sim_scores[1:n+1]

        # geting movies indices
        movie_indices = [i[0] for i in sim_scores]
        
        return self.df['title'].iloc[movie_indices].tolist()

#TEST : 
if __name__ == "__main__":
    import sys
    import os
    
    # importing the file in dataloader :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data_loader import DataLoader

    # data loading
    loader = DataLoader()
    movies_df, _, _ = loader.load_data() 

    # model initialization
    model = ContentRecommender(movies_df)
    
    # example recommendation
    test_movie = "Toy Story (1995)" # try changing the name to "Batman Forever (1995)"
    print(f"\n Since you liked '{test_movie}', you might also like:")
    
    recommendations = model.recommend(test_movie)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")