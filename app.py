from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the data saved from your notebook
with open('movie_recommender.pkl', 'rb') as file:
    data = pickle.load(file)
    df = data['df']
    features = data['features']
    indices = data['indices']

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def get_best_recommendations(title):
    if title not in indices:
        return None
    
    idx = indices[title]
    # Handle duplicate indices if they exist
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Calculate similarity scores
    from sklearn.metrics.pairwise import cosine_similarity
    target_vector = features[idx].reshape(1, -1)
    similarity_scores = cosine_similarity(target_vector, features).flatten()
    sim_scores = list(enumerate(similarity_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    
    movie_indices = [i[0] for i in sim_scores]
    movies = df.iloc[movie_indices][['title', 'vote_count', 'vote_average']].copy()
    
    # IMDB-style Weighted Rating Logic
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.60)
    
    qualified = movies[(movies['vote_count'] >= m)].copy()
    qualified['wr'] = qualified.apply(weighted_rating, axis=1, args=(m, C))
    
    return qualified.sort_values('wr', ascending=False).head(10)

@app.route('/')
def home():
    # Pass movie titles to the frontend for a dropdown or search hint
    all_movies = sorted(df['title'].unique().tolist())
    return render_template('index.html', movies=all_movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title')
    recommendations = get_best_recommendations(movie_title)
    
    if recommendations is None or recommendations.empty:
        return render_template('index.html', error="Movie not found. Please try another.")
    
    # Convert dataframe to list of dicts for easy rendering in HTML
    results = recommendations.to_dict('records')
    return render_template('index.html', results=results, selected_movie=movie_title)

if __name__ == '__main__':
    app.run(debug=True)