import streamlit as st
import pandas as pd
import requests
import sys
import os

# Add the project root to the path so we can import src modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.models.simple import SimpleRecommender
from src.models.content_based import ContentRecommender
from src.models.collaborative import CollaborativeRecommender

# -----------------------------------------------------------------------------
# 1. Page Config & Netflix/IMDB Style CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Netflix/IMDB Inspired CSS
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@400;500;700&display=swap');
    
    /* Hide Streamlit Defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark Background */
    .stApp {
        background: linear-gradient(180deg, #0d0d0d 0%, #1a1a1a 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #141414;
        border-right: 1px solid #333;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #e5e5e5 !important;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Logo/Brand in Sidebar */
    .brand-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.5rem;
        color: #E50914;
        letter-spacing: 2px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Roboto', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
        border-left: 4px solid #E50914;
        padding-left: 15px;
    }
    
    .section-subheader {
        font-family: 'Roboto', sans-serif;
        font-size: 1rem;
        color: #808080;
        margin-bottom: 25px;
        padding-left: 19px;
    }
    
    /* Movie Card */
    .movie-card {
        background: #181818;
        border-radius: 4px;
        overflow: hidden;
        transition: all 0.3s ease;
        margin-bottom: 20px;
        position: relative;
    }
    
    .movie-card:hover {
        transform: scale(1.05);
        z-index: 10;
        box-shadow: 0 14px 30px rgba(0,0,0,0.7);
    }
    
    .movie-card img {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        display: block;
    }
    
    .movie-info {
        padding: 12px;
        background: linear-gradient(to top, #181818 80%, transparent);
    }
    
    .movie-title {
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        color: #fff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 5px;
    }
    
    .movie-rating {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .rating-star {
        color: #F5C518; /* IMDB Yellow */
        font-size: 0.85rem;
    }
    
    .rating-value {
        font-family: 'Roboto', sans-serif;
        font-size: 0.85rem;
        color: #F5C518;
        font-weight: 700;
    }
    
    .match-badge {
        background: #46D369;
        color: #000;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: #E50914;
        color: white;
        border: none;
        border-radius: 4px;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        padding: 10px 24px;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #f40612;
        color: white;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: #333;
        border: 1px solid #444;
        color: #fff;
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background: #333;
        border: 1px solid #444;
        color: #fff;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #E50914 !important;
    }
    
    /* Divider */
    hr {
        border-color: #333;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data & Model Loading (Cached)
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    """Loads dataset using the DataLoader class and reads links.csv."""
    loader = DataLoader()
    movies, ratings, df_merged = loader.load_data()
    
    # Load links for API calls
    links = pd.read_csv("data/raw/links.csv")
    
    # Ensure tmdbId is numeric and drop NaNs
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)
    
    return movies, ratings, df_merged, links

try:
    movies, ratings, df_merged, links = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


@st.cache_resource
def load_models():
    """Initializes and caches the recommender models."""
    simple = SimpleRecommender(df_merged)
    content = ContentRecommender(df_merged)
    collab = CollaborativeRecommender(
        model_path='saved_models/svd_model_optimized.pkl',
        movies_path='data/raw/movies.csv'
    )
    return simple, content, collab

with st.spinner(""):
    simple_model, content_model, collab_model = load_models()

# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------

def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    try:
        row = links[links['movieId'] == movie_id]
        if row.empty:
            return "https://via.placeholder.com/500x750/1a1a1a/808080?text=No+Poster"
        
        tmdb_id = row['tmdbId'].values[0]
        api_key = st.secrets["tmdb_api_key"]
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
        
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        return "https://via.placeholder.com/500x750/1a1a1a/808080?text=No+Poster"
    except Exception:
        return "https://via.placeholder.com/500x750/1a1a1a/808080?text=Error"


def display_movies_grid(movies_data, show_match=False):
    """Displays movies in a Netflix-style grid."""
    
    # Normalize input
    if isinstance(movies_data, list):
        temp_df = pd.DataFrame(movies_data, columns=['title'])
        title_map = df_merged[['title', 'movieId', 'average_rating']].drop_duplicates(subset='title')
        display_df = temp_df.merge(title_map, on='title', how='left')
        
    elif isinstance(movies_data, pd.DataFrame):
        if 'Title' in movies_data.columns:
            movies_data = movies_data.rename(columns={'Title': 'title'})
        
        if 'movieId' not in movies_data.columns:
            title_map = df_merged[['title', 'movieId', 'average_rating']].drop_duplicates(subset='title')
            display_df = movies_data.merge(title_map, on='title', how='left')
        else:
            display_df = movies_data.copy()
            if 'average_rating' not in display_df.columns:
                rating_map = df_merged[['movieId', 'average_rating']].drop_duplicates()
                display_df = display_df.merge(rating_map, on='movieId', how='left')
    else:
        st.error("Invalid data format")
        return

    display_df = display_df.dropna(subset=['movieId'])
    
    if display_df.empty:
        st.warning("No movies to display.")
        return
    
    # Grid Layout
    cols_per_row = 5
    num_movies = len(display_df)
    
    for row_start in range(0, num_movies, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            movie_idx = row_start + col_idx
            if movie_idx < num_movies:
                row = display_df.iloc[movie_idx]
                with col:
                    poster_url = fetch_poster(int(row['movieId']))
                    title = row['title']
                    
                    # Get rating value safely
                    avg_rating = row.get('average_rating', 0)
                    if pd.isna(avg_rating):
                        avg_rating = 0
                    
                    # Build card HTML
                    if show_match and 'Estimated Rating' in row.index:
                        est_rating = row['Estimated Rating']
                        if pd.notna(est_rating):
                            match_pct = int(est_rating * 20)
                            card_html = f'''<div class="movie-card">
                                <img src="{poster_url}" alt="{title}">
                                <div class="movie-info">
                                    <div class="movie-title" title="{title}">{title}</div>
                                    <span class="match-badge">{match_pct}% Match</span>
                                </div>
                            </div>'''
                        else:
                            card_html = f'''<div class="movie-card">
                                <img src="{poster_url}" alt="{title}">
                                <div class="movie-info">
                                    <div class="movie-title" title="{title}">{title}</div>
                                </div>
                            </div>'''
                    elif avg_rating > 0:
                        card_html = f'''<div class="movie-card">
                            <img src="{poster_url}" alt="{title}">
                            <div class="movie-info">
                                <div class="movie-title" title="{title}">{title}</div>
                                <div class="movie-rating">
                                    <span class="rating-star">★</span>
                                    <span class="rating-value">{avg_rating:.1f}</span>
                                </div>
                            </div>
                        </div>'''
                    else:
                        card_html = f'''<div class="movie-card">
                            <img src="{poster_url}" alt="{title}">
                            <div class="movie-info">
                                <div class="movie-title" title="{title}">{title}</div>
                            </div>
                        </div>'''
                    
                    st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. Sidebar Navigation
# -----------------------------------------------------------------------------

st.sidebar.markdown('<div class="brand-title">CINEMATCH</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Browse",
    ["Top Rated", "Similar Movies", "For You"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="color: #666; font-size: 0.75rem; text-align: center;">
    Powered by TMDB API<br>
    Built with Streamlit
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Page Content
# -----------------------------------------------------------------------------

# --- PAGE 1: TOP RATED ---
if page == "Top Rated":
    st.markdown('<div class="section-header">Top Rated Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Highest rated movies based on weighted score algorithm</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading..."):
        top_movies_df = simple_model.recommend(top_n=50)
    
    display_movies_grid(top_movies_df)

# --- PAGE 2: SIMILAR MOVIES ---
elif page == "Similar Movies":
    st.markdown('<div class="section-header">Find Similar Movies</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Content-based recommendations using genre similarity</div>', unsafe_allow_html=True)
    
    # Search Box
    movie_list = sorted(df_merged['title'].unique())
    selected_movie = st.selectbox("Search for a movie you like", movie_list, index=0)
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_btn = st.button("Get Recommendations")
    
    if search_btn:
        with st.spinner("Finding similar movies..."):
            recommendations_list = content_model.recommend(title=selected_movie, n=15)
        
        if recommendations_list and "sorry" in str(recommendations_list[0]).lower():
            st.error(recommendations_list[0])
        else:
            st.markdown("---")
            st.markdown(f'<div class="section-header">Because You Liked "{selected_movie.split("(")[0].strip()}"</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subheader">Movies with similar genres and themes</div>', unsafe_allow_html=True)
            display_movies_grid(recommendations_list)

# --- PAGE 3: FOR YOU ---
elif page == "For You":
    st.markdown('<div class="section-header">Personalized For You</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">AI-powered recommendations using collaborative filtering (SVD)</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=610, value=1, step=1)
    with col2:
        st.write("")  # Spacer
        st.write("")
        show_btn = st.button("Show Recommendations")
    
    if show_btn:
        with st.spinner("Analyzing your preferences..."):
            try:
                user_recs_df = collab_model.recommend(user_id=user_id, top_n=15)
                st.markdown("---")
                st.markdown(f'<div class="section-header">Recommended For User {user_id}</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-subheader">Based on users with similar taste</div>', unsafe_allow_html=True)
                display_movies_grid(user_recs_df, show_match=True)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
