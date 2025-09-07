import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data():
    """
    Loads and preprocesses the MovieLens 100k dataset.
    
    Returns:
        tuple: A tuple containing:
            - movies_df (pd.DataFrame): DataFrame with movie titles and genres.
            - ratings_df (pd.DataFrame): DataFrame with user ratings.
            - similarity_matrix (np.ndarray): The content-based similarity matrix.
            - indices (pd.Series): Series mapping movie titles to indices.
    """
    # Define column names for the datasets
    movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
    rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    
    # Load the datasets, specifying the separator and header
    try:
        movies_df = pd.read_csv('data/u.item', sep='|', names=movie_cols, encoding='latin-1')
        ratings_df = pd.read_csv('data/u.data', sep='\t', names=rating_cols)
    except FileNotFoundError:
        print("Error: The MovieLens dataset files were not found.")
        print("Please ensure 'u.item' and 'u.data' are in a 'data/' directory.")
        return None, None, None, None
        
    # Drop unnecessary columns from movies_df and clean up the title
    movies_df = movies_df[['movie_id', 'title'] + movie_cols[5:]]
    movies_df['title'] = movies_df['title'].apply(lambda x: x.split(' (')[0].strip())
    
    # Merge datasets
    df = pd.merge(ratings_df, movies_df, on='movie_id')
    
    # --- Content-Based Filtering Setup ---
    # Create a single string of genres for each movie
    movies_df['genres'] = movies_df.iloc[:, 2:].apply(lambda row: ' '.join(row.index[row == 1]), axis=1)
    
    # Create a TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    # Compute the cosine similarity matrix
    similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Create a reverse map of movie titles to indices
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    
    return movies_df, ratings_df, similarity_matrix, indices

def get_movies_list(movies_df):
    """
    Returns a sorted list of unique movie titles.
    """
    return sorted(movies_df['title'].unique())