# 🎬 Movie Recommender Pro

Movie Recommender Pro is a **hybrid movie recommendation system** built with **Streamlit**, combining **Content-Based Filtering** and **Collaborative Filtering** into a sleek, interactive web app.  

It helps users discover their next favorite movie by leveraging both movie metadata (genres) and user rating patterns.  

---

## 🚀 Features

✅ **Hybrid Recommendation Engine** – Combines **Content-Based** and **Collaborative Filtering**  
✅ **Beautiful UI** – Dark themed, modern, and interactive Streamlit app  
✅ **Search-as-you-type** – Start typing a movie name and select from auto-suggestions  
✅ **Dynamic Movie Cards** – Recommendations displayed as interactive cards  
✅ **Progress Feedback** – Smooth animations & progress bar while generating recommendations  
✅ **Insights Dashboard** – Metrics like average match score, total recommendations, and best match  

---

## 🧠 How It Works

The app uses two different recommendation strategies:

1. **Content-Based Filtering**  
   - Uses **TF-IDF Vectorization** on movie genres  
   - Finds similarity using **Cosine Similarity**  
   - Recommends movies with similar genres  

2. **Collaborative Filtering**  
   - Builds a **user-item rating matrix** from `ratings.csv`  
   - Calculates similarity between movies using **Cosine Similarity**  
   - Suggests movies liked by users with similar preferences  

3. **Hybrid Approach**  
   - Normalizes scores using **MinMaxScaler**  
   - Combines them using a **weighted sum** (30% content + 70% collaborative by default)  
   - Returns the **Top 10 Recommendations**  

---

## 📂 Dataset

This app uses the **MovieLens dataset**.  
You’ll need the following CSV files in your working directory:

- `movies.csv` – Contains movie details (`movieId`, `title`, `genres`)  
- `ratings.csv` – Contains user ratings (`userId`, `movieId`, `rating`, `timestamp`)  

Download dataset here: [MovieLens Dataset](https://grouplens.org/datasets/movielens/)  

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
# Clone this repo
git clone https://github.com/yourusername/movie-recommender-pro.git
cd movie-recommender-pro

# Install dependencies
pip install -r requirements.txt
