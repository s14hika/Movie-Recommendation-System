# Movie Recommendation System

## 📖 Problem Statement

Building a recommendation system is a core machine learning task. This project demonstrates how to implement **collaborative filtering** to recommend movies based on user preferences. The system uses **cosine similarity** to find movies similar to ones a user has watched and rated highly.

## 📊 Dataset

- **Size**: 10,000 user ratings + 50 popular movie titles (synthetic dataset)
- **Rating Scale**: 1-5 (implicit user preferences)
- **Movies**: Popular titles across various genres
- **Challenge**: Building personalized recommendations from sparse user-movie interaction data

## 🔬 Methodology

### 1. Data Generation
- Created synthetic dataset with 50 movie titles
- Generated 10,000 user ratings (1-5 scale)
- Simulated realistic user preference patterns

### 2. Collaborative Filtering Approach
- **Algorithm**: User-based Collaborative Filtering
- **Similarity Metric**: Cosine Similarity
- **Rationale**: Finds users with similar taste and recommends movies they liked

### 3. Implementation Steps
```
1. Create user-movie rating matrix
2. Compute cosine similarity between users
3. Find similar users to target user
4. Recommend movies liked by similar users
5. Rank recommendations by frequency/rating
```

## 🚀 Features

✅ User-based collaborative filtering  
✅ Cosine similarity-based recommendations  
✅ Interactive movie search and recommendation  
✅ Synthetic dataset generation  
✅ Scalable recommendation engine  

## 📈 Results

| Metric | Value |
|--------|-------|
| **Recommendation Accuracy** | Validated on similar user profiles |
| **Number of Movies** | 50 |
| **User Ratings** | 10,000 |
| **Algorithms** | Cosine Similarity |
| **Recommendation Method** | Top-N recommendations |

## 💻 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Usage
```bash
python movie_recommendation.py

# Enter a movie title when prompted
# System returns top 10 similar movies
```

### Example Output
```
Enter a movie: Inception

Top 10 Similar Movies:
1. Interstellar (similarity: 0.89)
2. The Matrix (similarity: 0.85)
3. Dark Knight (similarity: 0.82)
...
```

## 🎓 Key Learnings

✅ Understanding collaborative filtering fundamentals  
✅ Computing similarity metrics (cosine similarity)  
✅ User-based vs item-based filtering trade-offs  
✅ Handling sparse user-item matrices  
✅ Building scalable recommendation engines  

## 🔮 Future Improvements

- [ ] Implement item-based collaborative filtering
- [ ] Add matrix factorization (SVD) for better accuracy
- [ ] Implement hybrid recommendation system (content + collaborative)
- [ ] Add user ratings interface for dynamic recommendations
- [ ] Optimize for large-scale datasets (millions of users)
- [ ] Deploy as REST API

## 📚 Technologies Used

- **Python** - Core language
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Cosine similarity computation
- **Jupyter Notebook** - Development and documentation

## 📝 Files in Repository

```
├── movie_recommendation.py          # Main recommendation engine
├── data_generation.py               # Synthetic dataset generator
├── Movie_Recommendation_System.ipynb # Jupyter notebook with full workflow
└── README.md                        # This file
```

## 🤝 Author

**Shaik Sadhika**  
Final-year B.Tech AI/ML Student | Data Science Enthusiast  
📧 [shaikbushrafathima1926@gmail.com](mailto:shaikbushrafathima1926@gmail.com)  
🔗 [GitHub: s14hika](https://github.com/s14hika)  
🔗 [LinkedIn: sadhika-shaik](https://linkedin.com/in/sadhika-shaik)  

## 📜 License

This project is open source and available under the MIT License.

---

**Made with ❤️ for the ML community**
