# Movie Recommendation System - Flask Backend
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import os
import io

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATA GENERATION ====================

def generate_movie_data():
    """Generate synthetic movie and rating data"""
    np.random.seed(42)
    
    movies = [
        'Inception', 'The Dark Knight', 'Interstellar', 'Pulp Fiction', 'Fight Club',
        'The Matrix', 'Forrest Gump', 'Shawshank Redemption', 'The Godfather', 'Avatar',
        'Titanic', 'The Avengers', 'Jurassic Park', 'Gladiator', 'The Lion King',
        'Toy Story', 'Inside Out', 'Finding Nemo', 'Frozen', 'Coco',
        'Parasite', 'Joker', 'The Irishman', 'Once Upon a Time', 'Dune',
        'Tenet', 'Black Widow', 'Spider-Man', 'Doctor Strange', 'Thor',
        'Aquaman', 'Wonder Woman', 'The Flash', 'Superman', 'Batman',
        'The Conjuring', 'Insidious', 'The Ring', 'Hereditary', 'A Quiet Place',
        'La La Land', 'The Greatest Showman', 'Bohemian Rhapsody', 'Hairspray', 'Mamma Mia',
        'The Notebook', 'Pride and Prejudice', 'Crazy Rich Asians', 'Arrival', 'Dune'
    ]
    
    n_users = 100
    n_movies = len(movies)
    ratings = np.random.randint(1, 6, (n_users, n_movies))
    
    return movies, ratings

MOVIES, RATINGS_MATRIX = generate_movie_data()

# ==================== RECOMMENDATION ENGINE ====================

def get_recommendations(movie_name, top_n=5):
    """Get movie recommendations using cosine similarity"""
    try:
        movie_index = None
        for i, movie in enumerate(MOVIES):
            if movie.lower() == movie_name.lower():
                movie_index = i
                break
        
        if movie_index is None:
            return {'success': False, 'error': f'Movie \"{movie_name}\" not found'}
        
        similarities = cosine_similarity([RATINGS_MATRIX[:, movie_index]], RATINGS_MATRIX.T)[0]
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'rank': len(recommendations) + 1,
                'name': MOVIES[idx],
                'score': round(float(similarities[idx]), 3),
                'percentage': round(float(similarities[idx]) * 100, 1)
            })
        
        return {
            'success': True,
            'movie': movie_name,
            'recommendations': recommendations
        }
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return {'success': False, 'error': str(e)}

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', movies=MOVIES)

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get all available movies"""
    return jsonify({
        'success': True,
        'total': len(MOVIES),
        'movies': MOVIES
    }), 200

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get recommendations for a movie"""
    try:
        data = request.json
        movie_name = data.get('movie_name', '').strip()
        top_n = int(data.get('top_n', 5))
        
        if not movie_name:
            return jsonify({'success': False, 'error': 'Movie name required'}), 400
        
        result = get_recommendations(movie_name, top_n)
        return jsonify(result), (200 if result['success'] else 404)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by name"""
    query = request.args.get('q', '').lower()
    matches = [m for m in MOVIES if query in m.lower()]
    return jsonify({'success': True, 'query': query, 'results': matches}), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'total_movies': len(MOVIES),
        'total_users': RATINGS_MATRIX.shape[0],
        'total_ratings': int(RATINGS_MATRIX.size),
        'avg_rating': float(np.mean(RATINGS_MATRIX)),
        'algorithm': 'Collaborative Filtering'
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
