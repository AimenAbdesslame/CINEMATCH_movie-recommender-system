# Smart Movie Recommender System

A hybrid movie recommendation engine built with **Python** and **Streamlit**. This application mimics a modern streaming service experience (like Netflix) to provide personalized movie suggestions using three different algorithmic approaches.



##  Features

1.  ** Top Charts (Simple Recommender):**
    - Displays the top-rated movies based on a weighted rating formula (IMDb formula).
    - Perfect for new users with no history.

2.  ** Because You Watched (Content-Based):**
    - Recommend movies similar to a specific movie you like.
    - Uses metadata (Genres, Keywords) to find similarity.

3.  ** For You (Collaborative Filtering):**
    - Personalized recommendations for specific users based on their past ratings.
    - Powered by **SVD (Singular Value Decomposition)** machine learning model.
    - Achieved an RMSE of **0.87** on the test set.

4.  ** Dynamic UI: **
    - Fetches real-time high-quality movie posters using **TMDB API**.
    - Dark mode "Netflix-style" interface.

---

##  Project Structure

```text
MOVIE-RECOMMENDER-SYSTEM/
├── .streamlit/
│   └── secrets.toml       # API Keys (Not included in repo)
├── data/
│   └── raw/
│       ├── movies.csv     # Movie titles and genres
│       ├── ratings.csv    # User ratings
│       └── links.csv      # IDs to fetch posters
├── saved_models/
│   └── svd_model_optimized.pkl  # Pre-trained SVD model
├── src/
│   └── models/
│       ├── simple.py          # Weighted Rating Logic
│       ├── content_based.py   # Similarity Logic
│       └── collaborative.py   # SVD Model Loader
├── app.py                     # Main Streamlit Application
└── README.md



#  The math behind.excalidraw File 
  See : https://excalidraw.com/#json=0XDqbThk7CxMm8ntFDcVE,uqBIBxfrGERgyYqhlj-VdQ
