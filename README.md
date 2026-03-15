# 📚 Audible Insights — Intelligent Book Recommendation System

An end-to-end ML-powered audiobook recommendation system built on **3,535 Audible titles**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 🗂️ Repository Structure

```
audible_insights/
├── app/
│   ├── app.py                        ← Main Streamlit app
│   ├── recommender.py                ← All 5 recommendation models
│   └── photo_b64.txt                 ← Creator photo (base64)
├── models/
│   ├── cosine_sim.pkl                ← TF-IDF cosine similarity matrix
│   ├── tfidf_model.pkl
│   ├── tfidf_matrix.pkl
│   ├── kmeans_model.pkl
│   └── pca_model.pkl
├── outputs/
│   └── cleaned_data_with_clusters.csv ← Cleaned dataset with cluster labels
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_nlp_clustering.ipynb
│   └── 04_recommendation_models.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Local Setup

```bash
git clone https://github.com/Kavya1245/audible-insights.git
cd audible-insights
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 🤖 5 Recommendation Models

| Model | Technique | Best For |
|-------|-----------|----------|
| Model 1 | Content-Based (TF-IDF + Cosine Similarity) | Similar descriptions |
| Model 2 | Cluster-Based (K-Means) | Thematic groups |
| Model 3 | Popularity-Based (Weighted Score) | New users |
| Model 4 | Genre-Based (Exact match) | Genre fans |
| Model 5 ⭐ | Hybrid (All 4 combined) | Best overall |

---

## 📊 14 Questions Answered

- **5 Easy (EDA):** Genre popularity, top authors, rating distribution, trends, rating vs reviews
- **4 Medium:** Cluster groupings, genre similarity, author popularity effect, best model
- **3 Scenario:** Sci-Fi new user, Thriller fan, Hidden gems discovery

---

## 👩‍💻 Creator

**Kavya S** — Data Science & ML Engineer  
📧 kavya22s145@gmail.com  
💼 [linkedin.com/in/kavya-s1245](https://linkedin.com/in/kavya-s1245)  
🐙 [github.com/Kavya1245](https://github.com/Kavya1245)