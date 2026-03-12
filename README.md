# Movie-recommender-model

A high-performance movie recommendation system that combines Natural Language Processing (NLP) with a Weighted Rating System. The engine analyzes over 45,000 movies to suggest content based on plot similarity and audience reception.

🔗 Live Demo on Render(Note: Initial loading may take a moment on the free tier due to "cold starts".)
https://movie-recommender-model-hqag.onrender.com

🚀 Key FeaturesContent-Based Filtering: Uses TF-IDF and N-grams to analyze movie descriptions and taglines.
SVD Optimization: Implements Truncated Singular Value Decomposition (SVD) to reduce 19,000+ features into 200 core components, ensuring lightning-fast similarity calculations.
Weighted Ranking: Re-ranks initial results using IMDB's Weighted Rating formula to ensure recommended movies are of high quality.Responsive UI: A clean Flask-based web interface for seamless searching.

🧠 System Architecture1. 
  The Vectorization Pipeline: The system processes text data by merging movie overviews and taglines, then applies:TF-IDF Vectorizer: Captures the importance of          keywords while ignoring common stop words.
  TruncatedSVD: Reduces the dimensionality of the matrix. This was a critical step for deployment, shrinking the model from 1.8GB to ~40MB.2.
  
🛠️ Tech StackBackend: 
Python (Flask, Gunicorn)
Data Science: Scikit-Learn, Pandas, NumPy, SciPyNLP: NLTK (Stemming & Lemmatization)
Frontend: HTML5, CSS3 (Jinaja2 Templates)
