import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_data, get_movies_list
from recommender import get_collaborative_recommendations, get_content_based_recommendations
import random

# Set page config for better appearance
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4ecdc4;
        margin-bottom: 1rem;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Cache the data loading
@st.cache_data
def cached_load_data():
    return load_data()

@st.cache_data
def get_movie_stats(movies_df, ratings_df):
    """Calculate various movie statistics"""
    stats = {
        'total_movies': len(movies_df),
        'total_ratings': len(ratings_df),
        'avg_rating': ratings_df['rating'].mean(),
        'total_users': ratings_df['user_id'].nunique()
    }
    return stats

@st.cache_data
def get_genre_stats(movies_df):
    """Get genre distribution"""
    genre_cols = movies_df.columns[2:-1]  # Exclude movie_id, title, and genres columns
    genre_counts = movies_df[genre_cols].sum().sort_values(ascending=False)
    return genre_counts

def main():
    # Load data
    movies_df, ratings_df, similarity_matrix, indices = cached_load_data()
    
    if movies_df is None:
        st.error("❌ Failed to load data. Please check the 'data/' directory.")
        st.stop()
    
    # Main header
    st.markdown('<h1 class="main-header">🎬 Advanced Movie Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Get statistics
    stats = get_movie_stats(movies_df, ratings_df)
    
    # Display key metrics in the main area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎥 Movies</h3>
            <h2>{stats['total_movies']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⭐ Ratings</h3>
            <h2>{stats['total_ratings']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Users</h3>
            <h2>{stats['total_users']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Avg Rating</h3>
            <h2>{stats['avg_rating']:.1f}/5</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Recommendation Settings")
    
    # Recommendation type selection with emojis
    recommendation_type = st.sidebar.selectbox(
        "🔍 Choose Recommendation Method:",
        ['🤝 Collaborative Filtering', '🎭 Content-Based Filtering', '🎲 Hybrid Approach', '📊 Data Analytics']
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "📈 Number of Recommendations:",
        min_value=3,
        max_value=15,
        value=5
    )
    
    # Advanced options
    st.sidebar.markdown("### 🔧 Advanced Options")
    show_details = st.sidebar.checkbox("📝 Show Movie Details", value=True)
    show_ratings = st.sidebar.checkbox("⭐ Show Rating Info", value=False)
    
    # Main content area
    if recommendation_type == '🤝 Collaborative Filtering':
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("## 🤝 Collaborative Filtering")
        st.markdown("**Find movies loved by users with similar tastes to yours!**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 This method analyzes user behavior patterns to suggest movies.")
            
            # User selection options
            user_input_method = st.radio(
                "Choose how to select user:",
                ["🔢 Enter User ID", "🎲 Random User", "🏆 Top Rated User"]
            )
            
            if user_input_method == "🔢 Enter User ID":
                user_id = st.number_input(
                    "Enter User ID:",
                    min_value=1,
                    max_value=943,
                    value=1,
                    help="User IDs range from 1 to 943"
                )
            elif user_input_method == "🎲 Random User":
                if st.button("🎲 Pick Random User"):
                    user_id = random.randint(1, 943)
                    st.success(f"Selected User ID: {user_id}")
                else:
                    user_id = 1
            else:  # Top Rated User
                # Find user with most ratings
                user_rating_counts = ratings_df['user_id'].value_counts()
                user_id = user_rating_counts.index[0]
                st.info(f"Using most active user (ID: {user_id}) with {user_rating_counts.iloc[0]} ratings")
        
        with col2:
            # Show user's rating history
            if 'user_id' in locals():
                user_ratings = ratings_df[ratings_df['user_id'] == user_id].merge(
                    movies_df[['movie_id', 'title']], on='movie_id'
                )
                if not user_ratings.empty:
                    st.markdown("### 📊 User's Top Ratings")
                    top_user_ratings = user_ratings.nlargest(5, 'rating')[['title', 'rating']]
                    for _, row in top_user_ratings.iterrows():
                        st.markdown(f"⭐ **{row['rating']}/5** - {row['title']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Get Collaborative Recommendations", type="primary"):
            with st.spinner('🔄 Analyzing user preferences...'):
                recommendations = get_collaborative_recommendations(
                    user_id, ratings_df, movies_df, num_recommendations
                )
            
            if recommendations:
                st.markdown(f"## 🎯 Top {len(recommendations)} Recommendations for User {user_id}")
                
                for i, movie in enumerate(recommendations):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{i+1} 🎬 {movie}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if show_details:
                            movie_info = movies_df[movies_df['title'] == movie].iloc[0]
                            genres = movie_info['genres'] if 'genres' in movie_info else "Not specified"
                            st.markdown(f"**Genres:** {genres}")
                        
                        if show_ratings:
                            movie_id = movies_df[movies_df['title'] == movie]['movie_id'].iloc[0]
                            movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]
                            avg_rating = movie_ratings['rating'].mean()
                            rating_count = len(movie_ratings)
                            st.markdown(f"**Average Rating:** ⭐ {avg_rating:.1f}/5 ({rating_count} ratings)")
            else:
                st.warning("⚠️ Could not generate recommendations. Please try a different user ID.")
    
    elif recommendation_type == '🎭 Content-Based Filtering':
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("## 🎭 Content-Based Filtering")
        st.markdown("**Discover movies similar to your favorites based on genres and content!**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 This method finds movies with similar genres and characteristics.")
            
            movie_list = get_movies_list(movies_df)
            
            # Movie selection options
            movie_input_method = st.radio(
                "Choose how to select movie:",
                ["🔍 Search & Select", "🎲 Random Movie", "⭐ Highly Rated Movie"]
            )
            
            if movie_input_method == "🔍 Search & Select":
                selected_movie = st.selectbox(
                    "Select a movie you like:",
                    movie_list,
                    help="Type to search for movies"
                )
            elif movie_input_method == "🎲 Random Movie":
                if st.button("🎲 Pick Random Movie"):
                    selected_movie = random.choice(movie_list)
                    st.success(f"Selected: {selected_movie}")
                else:
                    selected_movie = movie_list[0]
            else:  # Highly Rated Movie
                # Find highly rated movies
                movie_ratings = ratings_df.groupby('movie_id').agg({
                    'rating': ['mean', 'count']
                }).round(2)
                movie_ratings.columns = ['avg_rating', 'rating_count']
                movie_ratings = movie_ratings[movie_ratings['rating_count'] >= 50]  # At least 50 ratings
                top_movies = movie_ratings.nlargest(10, 'avg_rating')
                
                top_movie_titles = movies_df[movies_df['movie_id'].isin(top_movies.index)]['title'].tolist()
                selected_movie = st.selectbox("Choose from top-rated movies:", top_movie_titles)
        
        with col2:
            # Show selected movie details
            if 'selected_movie' in locals():
                st.markdown("### 🎬 Selected Movie Info")
                movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
                st.markdown(f"**🎭 Title:** {selected_movie}")
                
                if 'genres' in movie_info:
                    st.markdown(f"**🏷️ Genres:** {movie_info['genres']}")
                
                # Show rating info for selected movie
                movie_id = movie_info['movie_id']
                movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]
                if not movie_ratings.empty:
                    avg_rating = movie_ratings['rating'].mean()
                    rating_count = len(movie_ratings)
                    st.markdown(f"**⭐ Rating:** {avg_rating:.1f}/5 ({rating_count} reviews)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Get Content-Based Recommendations", type="primary"):
            with st.spinner('🔄 Finding similar movies...'):
                recommendations = get_content_based_recommendations(
                    selected_movie, movies_df, similarity_matrix, indices, num_recommendations
                )
            
            if recommendations:
                st.markdown(f"## 🎯 Top {len(recommendations)} Movies Similar to '{selected_movie}'")
                
                for i, movie in enumerate(recommendations):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{i+1} 🎬 {movie}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if show_details:
                            movie_info = movies_df[movies_df['title'] == movie].iloc[0]
                            genres = movie_info['genres'] if 'genres' in movie_info else "Not specified"
                            st.markdown(f"**Genres:** {genres}")
                        
                        if show_ratings:
                            movie_id = movies_df[movies_df['title'] == movie]['movie_id'].iloc[0]
                            movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]
                            avg_rating = movie_ratings['rating'].mean()
                            rating_count = len(movie_ratings)
                            st.markdown(f"**Average Rating:** ⭐ {avg_rating:.1f}/5 ({rating_count} ratings)")
            else:
                st.warning("⚠️ Could not generate recommendations. Please try a different movie.")
    
    elif recommendation_type == '🎲 Hybrid Approach':
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("## 🎲 Hybrid Recommendation System")
        st.markdown("**Get the best of both worlds - collaborative and content-based recommendations!**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤝 User-Based Component")
            user_id = st.number_input("User ID:", min_value=1, max_value=943, value=1)
        
        with col2:
            st.markdown("### 🎭 Content-Based Component")
            movie_list = get_movies_list(movies_df)
            selected_movie = st.selectbox("Reference Movie:", movie_list)
        
        hybrid_weight = st.slider(
            "⚖️ Balance (0 = Pure Collaborative, 1 = Pure Content-Based):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Get Hybrid Recommendations", type="primary"):
            with st.spinner('🔄 Computing hybrid recommendations...'):
                # Get both types of recommendations
                collab_recs = get_collaborative_recommendations(user_id, ratings_df, movies_df, num_recommendations)
                content_recs = get_content_based_recommendations(selected_movie, movies_df, similarity_matrix, indices, num_recommendations)
                
                # Simple hybrid approach: mix the recommendations based on weight
                hybrid_recs = []
                collab_count = int(num_recommendations * (1 - hybrid_weight))
                content_count = num_recommendations - collab_count
                
                hybrid_recs.extend(collab_recs[:collab_count])
                for rec in content_recs[:content_count]:
                    if rec not in hybrid_recs:
                        hybrid_recs.append(rec)
            
            if hybrid_recs:
                st.markdown(f"## 🎯 Hybrid Recommendations (Weight: {hybrid_weight})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🤝 From Collaborative Filtering:")
                    for i, movie in enumerate(collab_recs[:collab_count]):
                        st.markdown(f"**{i+1}.** {movie}")
                
                with col2:
                    st.markdown("### 🎭 From Content-Based Filtering:")
                    for i, movie in enumerate(content_recs[:content_count]):
                        st.markdown(f"**{i+1}.** {movie}")
                
                st.markdown("### 🎯 Combined Hybrid Results:")
                for i, movie in enumerate(hybrid_recs):
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>#{i+1} 🎬 {movie}</h4>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif recommendation_type == '📊 Data Analytics':
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("## 📊 Movie Data Analytics Dashboard")
        st.markdown("**Explore insights from the movie database!**")
        
        # Analytics options
        analytics_type = st.selectbox(
            "Choose Analytics View:",
            ["📈 Genre Distribution", "⭐ Rating Analysis", "👥 User Activity", "🎬 Movie Popularity"]
        )
        
        if analytics_type == "📈 Genre Distribution":
            genre_stats = get_genre_stats(movies_df)
            
            fig = px.bar(
                x=genre_stats.index[:10],
                y=genre_stats.values[:10],
                title="Top 10 Movie Genres",
                labels={'x': 'Genre', 'y': 'Number of Movies'},
                color=genre_stats.values[:10],
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie chart
            fig_pie = px.pie(
                values=genre_stats.values[:8],
                names=genre_stats.index[:8],
                title="Genre Distribution (Top 8)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        elif analytics_type == "⭐ Rating Analysis":
            # Rating distribution
            fig_hist = px.histogram(
                ratings_df,
                x='rating',
                title="Distribution of Movie Ratings",
                labels={'rating': 'Rating', 'count': 'Number of Ratings'},
                color_discrete_sequence=['#ff6b6b']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Average ratings by genre
            genre_cols = movies_df.columns[2:-1]
            genre_ratings = []
            
            for genre in genre_cols[:10]:  # Top 10 genres
                genre_movies = movies_df[movies_df[genre] == 1]['movie_id']
                genre_movie_ratings = ratings_df[ratings_df['movie_id'].isin(genre_movies)]
                if not genre_movie_ratings.empty:
                    avg_rating = genre_movie_ratings['rating'].mean()
                    genre_ratings.append({'Genre': genre, 'Average Rating': avg_rating})
            
            if genre_ratings:
                genre_df = pd.DataFrame(genre_ratings)
                fig_genre = px.bar(
                    genre_df,
                    x='Genre',
                    y='Average Rating',
                    title="Average Rating by Genre",
                    color='Average Rating',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_genre, use_container_width=True)
        
        elif analytics_type == "👥 User Activity":
            # User rating distribution
            user_activity = ratings_df['user_id'].value_counts()
            
            fig_user = px.histogram(
                x=user_activity.values,
                title="Distribution of User Activity",
                labels={'x': 'Number of Ratings per User', 'y': 'Number of Users'},
                color_discrete_sequence=['#4ecdc4']
            )
            st.plotly_chart(fig_user, use_container_width=True)
            
            # Top active users
            st.markdown("### 🏆 Most Active Users")
            top_users = user_activity.head(10)
            for i, (user_id, rating_count) in enumerate(top_users.items()):
                st.markdown(f"**#{i+1}** User {user_id}: {rating_count} ratings")
        
        elif analytics_type == "🎬 Movie Popularity":
            # Most rated movies
            movie_popularity = ratings_df.groupby('movie_id').agg({
                'rating': ['count', 'mean']
            }).round(2)
            movie_popularity.columns = ['rating_count', 'avg_rating']
            movie_popularity = movie_popularity[movie_popularity['rating_count'] >= 20]  # At least 20 ratings
            
            # Merge with movie titles
            popular_movies = movie_popularity.merge(
                movies_df[['movie_id', 'title']], 
                left_index=True, 
                right_on='movie_id'
            ).sort_values('rating_count', ascending=False)
            
            # Top 15 most rated movies
            top_movies = popular_movies.head(15)
            
            fig_pop = px.scatter(
                top_movies,
                x='rating_count',
                y='avg_rating',
                hover_data=['title'],
                title="Movie Popularity: Rating Count vs Average Rating",
                labels={'rating_count': 'Number of Ratings', 'avg_rating': 'Average Rating'},
                color='avg_rating',
                color_continuous_scale='viridis',
                size='rating_count'
            )
            st.plotly_chart(fig_pop, use_container_width=True)
            
            st.markdown("### 🔥 Most Popular Movies")
            for i, (_, row) in enumerate(top_movies.head(10).iterrows()):
                st.markdown(f"**#{i+1}** {row['title']}: {row['rating_count']} ratings (⭐ {row['avg_rating']}/5)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🎬 Advanced Movie Recommendation System | Built with ❤️ using Streamlit</p>
        <p>Data: MovieLens 100K Dataset | Features: Collaborative Filtering, Content-Based Filtering, Hybrid Approach & Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()