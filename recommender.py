import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def get_collaborative_recommendations(user_id, ratings_df, movies_df, num_recommendations=5):
    """
    Provides collaborative filtering recommendations using matrix factorization.
    """
    # Create a user-movie matrix
    movie_user_matrix = ratings_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    
    # Apply Truncated SVD for matrix factorization
    SVD = TruncatedSVD(n_components=12, random_state=42)
    matrix = SVD.fit_transform(movie_user_matrix)
    
    # Calculate cosine similarity of the SVD matrix
    corr_matrix = cosine_similarity(matrix)
    
    # Get the correlation with the target user
    user_ratings = movie_user_matrix.loc[:, user_id].to_numpy().reshape(1, -1)
    user_movie_corr = cosine_similarity(user_ratings, corr_matrix)
    
    # Get recommendations based on correlation
    recommendations = pd.Series(user_movie_corr.flatten(), index=movie_user_matrix.index).sort_values(ascending=False)
    
    # Filter out movies the user has already rated
    rated_movies = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].unique()
    recommendations = recommendations.drop(rated_movies, errors='ignore')
    
    # Get the top recommendations and merge with movie titles
    top_recommendations = recommendations.head(num_recommendations).index
    rec_movies = movies_df[movies_df['movie_id'].isin(top_recommendations)]
    
    return rec_movies['title'].tolist()

def get_content_based_recommendations(title, movies_df, similarity_matrix, indices, num_recommendations=5):
    """
    Provides content-based filtering recommendations using genres.
    """
    # Get the index of the movie that matches the title
    if title not in indices:
        return []
    idx = indices[title]
    
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:num_recommendations + 1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 5 most similar movies
    return movies_df['title'].iloc[movie_indices].tolist()