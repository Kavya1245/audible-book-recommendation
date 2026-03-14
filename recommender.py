"""
recommender.py — All 5 Recommendation Model Functions
Audible Insights: Intelligent Book Recommendation System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def find_book(df, title):
    """Find book index by partial title match (case-insensitive)."""
    hits = df[df['Book Name'].str.lower().str.contains(title.lower(), na=False)]
    if hits.empty:
        return None, None
    idx = hits.index[0]
    return idx, df.loc[idx, 'Book Name']


def model1_content_based(df, cosine_sim, book_title, top_n=5):
    """Recommend books with similar descriptions using cosine similarity."""
    idx, matched = find_book(df, book_title)
    if idx is None:
        return pd.DataFrame(), None

    sim_scores  = list(enumerate(cosine_sim[idx]))
    sim_scores  = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores if i != idx][:top_n]

    result = df.iloc[top_indices][['Book Name','Author','Rating','Genre',
                                    'Number of Reviews','Description']].copy()
    result.insert(0, 'Rank', range(1, len(result)+1))
    result['Similarity Score'] = [round(cosine_sim[idx][i], 4) for i in top_indices]
    result['Model'] = 'Content-Based'
    return result.reset_index(drop=True), matched


def model2_cluster_based(df, book_title, top_n=5):
    """Recommend top-rated books from the same cluster."""
    idx, matched = find_book(df, book_title)
    if idx is None:
        return pd.DataFrame(), None

    cluster_id    = df.loc[idx, 'cluster']
    cluster_books = df[(df['cluster'] == cluster_id) & (df.index != idx)].copy()
    cluster_books = cluster_books.sort_values('Rating', ascending=False).head(top_n)

    result = cluster_books[['Book Name','Author','Rating','Genre',
                             'Number of Reviews','Description']].copy()
    result.insert(0, 'Rank', range(1, len(result)+1))
    result['Cluster'] = cluster_id
    result['Model']   = 'Cluster-Based'
    return result.reset_index(drop=True), matched


def model3_popularity_based(df, genre_filter=None, top_n=5):
    """Recommend most popular books overall or within a genre."""
    # Compute popularity score if not already present
    if 'popularity_score' not in df.columns:
        scaler = MinMaxScaler()
        df = df.copy()
        df['reviews_norm']     = scaler.fit_transform(df[['Number of Reviews']])
        df['popularity_score'] = (df['Rating'] * 0.7) + (df['reviews_norm'] * 0.3)

    pool = df.copy()
    label = 'All Books'
    if genre_filter and genre_filter != 'All':
        pool  = df[df['Genre'].str.lower().str.contains(genre_filter.lower(), na=False)]
        label = genre_filter

    if pool.empty:
        return pd.DataFrame(), label

    result = (pool.sort_values('popularity_score', ascending=False)
                  .head(top_n)[['Book Name','Author','Rating',
                                'Number of Reviews','Genre',
                                'popularity_score','Description']]
                  .copy())
    result.insert(0, 'Rank', range(1, len(result)+1))
    result.rename(columns={'popularity_score': 'Popularity Score'}, inplace=True)
    result['Model'] = 'Popularity-Based'
    return result.reset_index(drop=True), label


def model4_genre_based(df, book_title, top_n=5):
    """Recommend top-rated books from the same genre."""
    idx, matched = find_book(df, book_title)
    if idx is None:
        return pd.DataFrame(), None

    genre = df.loc[idx, 'Genre']
    if genre == 'Unknown':
        cid  = df.loc[idx, 'cluster']
        pool = df[(df['cluster'] == cid) & (df.index != idx)].copy()
        note = f'Cluster {cid}'
    else:
        pool = df[(df['Genre'] == genre) & (df.index != idx)].copy()
        note = genre

    result = (pool.sort_values('Rating', ascending=False)
                  .head(top_n)[['Book Name','Author','Rating',
                                'Genre','Number of Reviews','Description']]
                  .copy())
    result.insert(0, 'Rank', range(1, len(result)+1))
    result['Filter'] = note
    result['Model']  = 'Genre-Based'
    return result.reset_index(drop=True), matched


def model5_hybrid(df, cosine_sim, book_title, top_n=5,
                  w_content=0.40, w_cluster=0.25,
                  w_popular=0.20, w_genre=0.15):
    """Hybrid recommendation combining all 4 models."""
    idx, matched = find_book(df, book_title)
    if idx is None:
        return pd.DataFrame(), None

    cluster_id = df.loc[idx, 'cluster']
    genre      = df.loc[idx, 'Genre']

    candidates = df[df.index != idx].copy()

    candidates['content_score'] = cosine_sim[idx][candidates.index]
    candidates['cluster_score'] = (candidates['cluster'] == cluster_id).astype(float)

    pop_range = df['popularity_score'].max() - df['popularity_score'].min() + 1e-9
    candidates['pop_score'] = (
        (candidates['popularity_score'] - df['popularity_score'].min()) / pop_range
    )

    if genre != 'Unknown':
        candidates['genre_score'] = (candidates['Genre'] == genre).astype(float)
    else:
        candidates['genre_score'] = 0.0

    candidates['hybrid_score'] = (
        candidates['content_score'] * w_content +
        candidates['cluster_score'] * w_cluster +
        candidates['pop_score']     * w_popular +
        candidates['genre_score']   * w_genre
    )

    result = (candidates.sort_values('hybrid_score', ascending=False)
                        .head(top_n)[['Book Name','Author','Rating','Genre',
                                      'Number of Reviews','Description',
                                      'content_score','cluster_score',
                                      'genre_score','hybrid_score']]
                        .copy())
    result = result.round(4)
    result.insert(0, 'Rank', range(1, len(result)+1))
    result['Model'] = 'Hybrid'
    return result.reset_index(drop=True), matched


def get_hidden_gems(df, min_rating=4.5, max_reviews_pct=0.25):
    """Return highly rated books with low review counts (hidden gems)."""
    reviews_threshold = df['Number of Reviews'].quantile(max_reviews_pct)
    gems = df[
        (df['Rating'] >= min_rating) &
        (df['Number of Reviews'] <= reviews_threshold)
    ].sort_values('Rating', ascending=False)

    result = gems[['Book Name','Author','Rating',
                   'Number of Reviews','Genre','Description']].copy()
    result.insert(0, 'Rank', range(1, len(result)+1))
    return result.reset_index(drop=True), reviews_threshold


def get_scifi_recommendations(df, top_n=5):
    """Scenario 1 — Recommend Sci-Fi books for a new user."""
    pool = df[df['Genre'].str.lower().str.contains(
        'science|fiction|fantasy|sci-fi|dystopian|futur', na=False
    )].copy()
    if pool.empty:
        pool = df.copy()
    result = (pool.sort_values('popularity_score', ascending=False)
                  .head(top_n)[['Book Name','Author','Rating',
                                'Number of Reviews','Genre','Description']]
                  .copy())
    result.insert(0, 'Rank', range(1, len(result)+1))
    return result.reset_index(drop=True)


def get_thriller_recommendations(df, cosine_sim, top_n=5):
    """Scenario 2 — Recommend books similar to top thriller."""
    pool = df[df['Genre'].str.lower().str.contains(
        'thriller|mystery|crime|suspense|detective', na=False
    )]
    if not pool.empty:
        seed = pool.sort_values('Rating', ascending=False).iloc[0]['Book Name']
    else:
        seed = df.sort_values('popularity_score', ascending=False).iloc[0]['Book Name']

    result, matched = model5_hybrid(df, cosine_sim, seed[:40], top_n=top_n)
    return result, seed