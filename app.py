import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import time

# Page configuration with custom styling
st.set_page_config(
    page_title="Movie Recommender Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for black, white, blue theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        color: white;
    }
    
    /* Hide Streamlit header and menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppViewContainer > .main > div:nth-child(1) {
        padding-top: 2rem;
    }
    
    /* Custom title styling */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #cccccc;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    
    /* Movie card styling */
    .movie-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ffffff, #cccccc);
        border-radius: 15px 15px 0 0;
    }
    
    .movie-rank {
        position: absolute;
        top: 15px;
        right: 20px;
        background: #ffffff;
        color: #000000;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
    }
    
    .movie-title {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 8px;
        padding-right: 60px;
        line-height: 1.4;
    }
    
    .movie-score {
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .score-bar {
        flex: 1;
        height: 6px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #ffffff, #cccccc);
        border-radius: 3px;
        transition: width 0.8s ease;
    }
    
    /* Search section styling */
    .search-container {
        background: rgba(45, 45, 45, 0.8);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
        width: 100%;
        height: 48px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.3);
        background: #f0f0f0;
        color: #000000;
    }
    
    .stButton > button:focus {
        background: #ffffff;
        color: #000000;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.5);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        height: 48px;
        display: flex;
        align-items: center;
    }
    
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Perfect alignment for search row */
    .search-row {
        display: flex;
        align-items: flex-end;
        gap: 15px;
        margin-top: 10px;
    }
    
    .search-input {
        flex: 1;
    }
    
    .search-button {
        min-width: 200px;
        display: flex;
        align-items: flex-end;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ffffff, #cccccc);
    }
    
    /* Warning and info boxes */
    .stWarning, .stInfo {
        background: rgba(45, 45, 45, 0.9);
        border-left: 4px solid #ffffff;
        border-radius: 0 10px 10px 0;
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 30px 0 20px 0;
        text-align: center;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #ffffff, #cccccc);
        border-radius: 2px;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite;
        color: #ffffff;
        text-align: center;
        font-size: 1.1rem;
        margin: 20px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999999;
        padding: 40px 20px;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess data with caching"""
    movie_df = pd.read_csv("movies.csv")
    movie_df['title'] = movie_df['title'].apply(lambda t: t.lower())
    
    rating_df = pd.read_csv("ratings.csv")
    rating_df = rating_df.drop(columns=['timestamp'])
    
    return movie_df, rating_df

@st.cache_data
def build_content_similarity(movie_df):
    """Build content-based similarity matrix with caching"""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(movie_df['genres']).toarray()
    sim_matrix = cosine_similarity(vectors)
    
    simi_cnt_df = pd.DataFrame(sim_matrix, 
                               index=movie_df['title'], 
                               columns=movie_df['title'])
    return simi_cnt_df

@st.cache_data
def build_collaborative_similarity(movie_df, rating_df):
    """Build collaborative filtering similarity matrix with caching"""
    rate_m_df = pd.merge(movie_df, rating_df, on='movieId')[['title','userId','rating']]
    rating_per_user = pd.pivot_table(data=rate_m_df, index='title',
                                     columns='userId', values='rating').fillna(0)
    
    rating_matrix = cosine_similarity(rating_per_user.values)
    simi_colb_df = pd.DataFrame(rating_matrix, 
                                index=rating_per_user.index, 
                                columns=rating_per_user.index)
    return simi_colb_df

def content_recommend(title, simi_cnt_df):
    """Content-based recommendation"""
    title = title.lower()
    if title not in simi_cnt_df.index:
        return pd.DataFrame(columns=["movie","content_score"])
    sim_score = simi_cnt_df[title].sort_values(ascending=False)[1:11]
    return pd.DataFrame({"movie": sim_score.index, "content_score": sim_score.values})

def collaborative_recommend(title, simi_colb_df):
    """Collaborative filtering recommendation"""
    title = title.lower()
    if title not in simi_colb_df.index:
        return pd.DataFrame(columns=["movie","collab_score"])
    simi_scores = simi_colb_df[title].sort_values(ascending=False)[1:11]
    return pd.DataFrame({"movie": simi_scores.index, "collab_score": simi_scores.values})

def hybrid_recommendation(title, simi_cnt_df, simi_colb_df):
    """Hybrid recommendation system"""
    content_df = content_recommend(title, simi_cnt_df)
    collab_df = collaborative_recommend(title, simi_colb_df)

    if content_df.empty and collab_df.empty:
        return pd.DataFrame(columns=["movie","final_score"])
    
    combined = pd.merge(content_df, collab_df, on='movie', how='outer').fillna(0)

    if not combined.empty:
        scaler = MinMaxScaler()
        combined[['content_scaled','collab_scaled']] = scaler.fit_transform(
            combined[['content_score','collab_score']]
        )

        combined['final_score'] = 0.3 * combined['content_scaled'] + 0.7 * combined['collab_scaled']
        final_recommendation = combined.sort_values('final_score', ascending=False)[['movie','final_score']][:10]

        return final_recommendation.reset_index(drop=True)
    
    return pd.DataFrame(columns=["movie","final_score"])

def display_movie_card(rank, title, score):
    """Display a movie recommendation card"""
    score_percentage = score * 100
    
    card_html = f"""
    <div class="movie-card">
        <div class="movie-rank">{rank}</div>
        <div class="movie-title">{title.title()}</div>
        <div class="movie-score">
            <span>Match Score: {score_percentage:.1f}%</span>
            <div class="score-bar">
                <div class="score-fill" style="width: {score_percentage}%;"></div>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-title">🎬 Movie Recommender Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your next favorite movie with our advanced AI-powered recommendation engine</p>', unsafe_allow_html=True)
    
    # Load data
    try:
        with st.spinner("🔄 Loading movie database..."):
            movie_df, rating_df = load_data()
            simi_cnt_df = build_content_similarity(movie_df)
            simi_colb_df = build_collaborative_similarity(movie_df, rating_df)
        
        # Search section
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        # Create a custom layout with proper alignment
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <label style="color: #ffffff; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem; display: block;">
                🎥 Search for a movie:
            </label>
        </div>
        """, unsafe_allow_html=True)
        
        # Use flexbox for perfect alignment
        col1, col2 = st.columns([4, 1], gap="medium")
        
        with col1:
            all_movies = sorted(list(movie_df['title'].unique()))
            movie_input = st.selectbox(
                "Search for a movie",  # Proper label for accessibility
                options=all_movies,
                help="Type or select a movie from our database",
                label_visibility="collapsed"
            )
        
        with col2:
            # Perfect alignment with selectbox
            recommend_button = st.button("🔘 Get Recommendations", key="recommend_btn")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendation results
        if recommend_button:
            if movie_input:
                # Progress bar animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps with progress
                steps = [
                    " Analyzing movie preferences...",
                    " Computing content similarities...",
                    " Finding collaborative patterns...",
                    " Generating hybrid recommendations...",
                    " Finalizing top picks..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.markdown(f'<div class="loading-text">{step}</div>', unsafe_allow_html=True)
                    time.sleep(0.3)  # Realistic loading time
                    progress_bar.progress((i + 1) * 20)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Get recommendations
                recommendations = hybrid_recommendation(movie_input, simi_cnt_df, simi_colb_df)
                
                if recommendations.empty:
                    st.warning(" Sorry, we couldn't find recommendations for this movie. It might not be in our database or have enough data.")
                else:
                    st.markdown(f'<div class="section-header"> Top Recommendations for "{movie_input.title()}"</div>', unsafe_allow_html=True)
                    
                    # Display recommendations in cards
                    for i, row in recommendations.iterrows():
                        display_movie_card(i + 1, row['movie'], row['final_score'])
                    
                    # Additional insights
                    st.markdown('<div class="section-header"> Recommendation Insights</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎬 Total Recommendations", len(recommendations))
                    with col2:
                        avg_score = recommendations['final_score'].mean()
                        st.metric("📈 Average Match Score", f"{avg_score:.2%}")
                    with col3:
                        top_score = recommendations['final_score'].max()
                        st.metric("🏆 Best Match Score", f"{top_score:.2%}")
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>🎬 Movie Recommender Pro | Powered by Hybrid Recommending Technology</p>
            <p>Combining Content-Based & Collaborative Filtering for Better Recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError as e:
        st.error("📁 Error: Could not find the required CSV files (movies.csv, ratings.csv). Please make sure they are in the same directory as this script.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()