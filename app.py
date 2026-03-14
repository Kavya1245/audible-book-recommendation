"""
app.py — Audible Insights: Intelligent Book Recommendation System
Professional White Theme | All 14 Questions | Fixed MemoryError | Dropdown Search
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')

# ── Path setup ───────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(APP_DIR, 'outputs')):
    BASE_DIR = APP_DIR
else:
    BASE_DIR = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR  = os.path.join(BASE_DIR, 'outputs')
sys.path.insert(0, APP_DIR)
sys.path.insert(0, BASE_DIR)

from recommender import (
    model1_content_based, model2_cluster_based,
    model3_popularity_based, model4_genre_based,
    model5_hybrid, get_hidden_gems,
    get_scifi_recommendations, get_thriller_recommendations,
    find_book
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audible Insights",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── FIX: Low DPI + safe_plot to prevent MemoryError ─────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor':   '#f8f9fc',
    'axes.edgecolor':   '#e8eaf0',
    'axes.labelcolor':  '#374151',
    'xtick.color':      '#6b7280',
    'ytick.color':      '#6b7280',
    'text.color':       '#374151',
    'grid.color':       '#e8eaf0',
    'grid.linewidth':   0.7,
    'font.family':      'sans-serif',
    'figure.dpi':       72,        # FIX: was default 100
    'savefig.dpi':      72,        # FIX: prevent MemoryError
})

def safe_plot(fig):
    """Render pyplot figure then immediately free all memory."""
    try:
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)
        plt.close('all')

PALETTE = ['#3730a3','#6366f1','#10b981','#f59e0b','#e11d48',
           '#0ea5e9','#8b5cf6','#84cc16','#f97316','#ec4899']


# ── Load Photo ────────────────────────────────────────────────────────────────
import base64
_photo_path = os.path.join(APP_DIR, 'photo_b64.txt')
if os.path.exists(_photo_path):
    with open(_photo_path, 'r') as _f:
        PHOTO_B64 = _f.read().strip()
else:
    PHOTO_B64 = ""

# ── Professional White CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, .stApp { background: #ffffff !important; font-family: 'DM Sans', sans-serif; }
.main .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1280px; }

[data-testid="stSidebar"] {
    background: #f8f9fc !important;
    border-right: 1px solid #e8eaf0;
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }

.stRadio > div { gap: 4px; }
.stRadio label {
    background: transparent; border-radius: 10px;
    padding: 10px 16px; cursor: pointer;
    font-weight: 500; font-size: 0.88rem;
    color: #4a5568 !important; transition: all 0.2s;
}
.stRadio label:hover { background: #eef2ff; color: #3730a3 !important; }

h1,h2,h3 { font-family: 'Playfair Display', serif !important; color: #1a1a2e !important; }
h4,h5,p,li { color: #374151 !important; }

.kpi-card {
    background: #ffffff; border: 1px solid #e8eaf0;
    border-radius: 16px; padding: 22px 20px;
    text-align: center; box-shadow: 0 2px 12px rgba(55,48,163,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(55,48,163,0.12); }
.kpi-val   { font-size: 2.1rem; font-weight: 800; color: #3730a3; font-family: 'Playfair Display', serif; }
.kpi-label { font-size: 0.78rem; color: #6b7280; margin-top: 5px; font-weight: 500;
             letter-spacing: 0.05em; text-transform: uppercase; }

.sec-header {
    display: flex; align-items: center; gap: 12px;
    margin: 28px 0 18px 0; padding-bottom: 12px;
    border-bottom: 2px solid #eef2ff;
}
.sec-header .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg, #3730a3, #6366f1); flex-shrink: 0;
}
.sec-header h3 { margin: 0; font-size: 1.1rem; color: #1a1a2e !important; }

.book-card {
    background: #ffffff; border: 1px solid #e8eaf0;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;
    border-left: 4px solid #3730a3;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: transform 0.15s, box-shadow 0.15s;
}
.book-card:hover { transform: translateX(4px); box-shadow: 0 4px 16px rgba(55,48,163,0.1); }
.book-title { font-size: 0.95rem; font-weight: 700; color: #1a1a2e; }
.book-meta  { font-size: 0.81rem; color: #6b7280; margin-top: 5px; }
.book-desc  { font-size: 0.81rem; color: #9ca3af; margin-top: 8px; font-style: italic; line-height: 1.5; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
         font-size: 0.73rem; font-weight: 600; margin: 2px; }
.b-indigo { background: #eef2ff; color: #3730a3; }
.b-green  { background: #ecfdf5; color: #065f46; }
.b-amber  { background: #fffbeb; color: #92400e; }
.b-rose   { background: #fff1f2; color: #9f1239; }
.b-sky    { background: #e0f2fe; color: #0369a1; }

.answer-box {
    background: #eef2ff; border-left: 4px solid #3730a3;
    border-radius: 0 10px 10px 0; padding: 14px 18px;
    margin: 14px 0; color: #1e1b4b; font-size: 0.88rem; line-height: 1.6;
}

.stTabs [data-baseweb="tab-list"] {
    background: #f8f9fc; border-radius: 12px; padding: 4px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #6b7280 !important;
    font-weight: 500; font-size: 0.84rem; padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #3730a3 !important;
    box-shadow: 0 2px 8px rgba(55,48,163,0.12);
}

.stTextInput input, .stSelectbox > div > div {
    border-radius: 10px !important; border: 1px solid #d1d5db !important;
    background: #ffffff !important; color: #1a1a2e !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stSelectbox > div > div:focus {
    border-color: #3730a3 !important;
    box-shadow: 0 0 0 3px rgba(55,48,163,0.08) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3730a3, #6366f1) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 28px !important; font-size: 0.88rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

.stSlider > div > div > div > div { background: #3730a3 !important; }
.stAlert { border-radius: 10px !important; }
.stDataFrame { border-radius: 12px !important; border: 1px solid #e8eaf0 !important; }
thead th { background: #f8f9fc !important; color: #1a1a2e !important; font-weight: 600 !important; }

.creator-card {
    background: linear-gradient(135deg, #3730a3 0%, #6366f1 100%);
    border-radius: 24px; padding: 40px; color: white; text-align: center;
    box-shadow: 0 20px 60px rgba(55,48,163,0.25);
}
.creator-photo {
    width: 130px; height: 130px; border-radius: 50%;
    border: 4px solid rgba(255,255,255,0.4);
    object-fit: cover; margin: 0 auto 16px auto; display: block;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.creator-name  { font-size: 1.6rem; font-weight: 800; font-family: 'Playfair Display', serif; }
.creator-role  { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
.contact-link {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
    border-radius: 10px; padding: 10px 18px; margin: 6px;
    color: white !important; text-decoration: none !important;
    font-size: 0.85rem; font-weight: 500; transition: all 0.2s;
}
.contact-link:hover { background: rgba(255,255,255,0.25); }

.intro-hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #3730a3 50%, #6366f1 100%);
    border-radius: 24px; padding: 48px 40px; color: white; margin-bottom: 32px;
}
.intro-hero h1 { color: white !important; font-size: 2.4rem; margin-bottom: 12px; }
.intro-hero p  { color: rgba(255,255,255,0.85) !important; font-size: 1.05rem; line-height: 1.7; }

.obj-card {
    background: #ffffff; border: 1px solid #e8eaf0;
    border-radius: 14px; padding: 20px; height: 100%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04); transition: transform 0.2s;
}
.obj-card:hover { transform: translateY(-3px); }
.obj-icon  { font-size: 2rem; margin-bottom: 10px; }
.obj-title { font-weight: 700; color: #1a1a2e; font-size: 0.95rem; margin-bottom: 6px; }
.obj-desc  { color: #6b7280; font-size: 0.83rem; line-height: 1.6; }

.dataset-card {
    background: #f8f9fc; border: 1px solid #e8eaf0;
    border-radius: 14px; padding: 20px 24px; margin-bottom: 12px;
}
.dataset-title { font-weight: 700; color: #3730a3; font-size: 0.95rem; }
.dataset-meta  { color: #6b7280; font-size: 0.83rem; margin-top: 4px; }

.pipeline-step {
    background: #ffffff; border: 1px solid #e8eaf0;
    border-radius: 12px; padding: 16px 18px; margin-bottom: 8px;
    display: flex; align-items: flex-start; gap: 14px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.step-num {
    background: linear-gradient(135deg, #3730a3, #6366f1);
    color: white; width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0; margin-top: 2px;
}
.step-body .step-title { font-weight: 700; color: #1a1a2e; font-size: 0.88rem; }
.step-body .step-desc  { color: #6b7280; font-size: 0.81rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Load Data & Models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'cleaned_data_with_clusters.csv'))
    df.reset_index(drop=True, inplace=True)
    # Ensure popularity_score exists
    if 'popularity_score' not in df.columns:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df['reviews_norm']     = scaler.fit_transform(df[['Number of Reviews']])
        df['popularity_score'] = (df['Rating'] * 0.7) + (df['reviews_norm'] * 0.3)
    return df

@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, 'cosine_sim.pkl'), 'rb') as f:
        return pickle.load(f)

df         = load_data()
cosine_sim = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 12px 0;'>
        <div style='font-size:2.2rem;'>📚</div>
        <div style='font-family:"Playfair Display",serif; font-size:1.15rem;
                    font-weight:700; color:#1a1a2e;'>Audible Insights</div>
        <div style='font-size:0.75rem; color:#6b7280; margin-top:3px;'>
            Intelligent Book Recommendation</div>
    </div>
    <hr style='border-color:#e8eaf0; margin:8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠 Introduction",
        "🤖 Book Recommender",
        "📊 EDA Dashboard",
        "🎯 Scenario Explorer",
        "👩‍💻 About Creator",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#e8eaf0; margin:16px 0;'>", unsafe_allow_html=True)
    n_books    = len(df)
    n_authors  = df['Author'].nunique()
    n_genres   = df[df['Genre'] != 'Unknown']['Genre'].nunique()
    n_clusters = df['cluster'].nunique()
    st.markdown(f"""
    <div style='font-size:0.8rem; color:#6b7280;'>
        <div style='padding:4px 0;'>📦 <b style='color:#3730a3;'>{n_books:,}</b> Books</div>
        <div style='padding:4px 0;'>👤 <b style='color:#3730a3;'>{n_authors:,}</b> Authors</div>
        <div style='padding:4px 0;'>🏷️ <b style='color:#3730a3;'>{n_genres}</b> Genres</div>
        <div style='padding:4px 0;'>🔵 <b style='color:#3730a3;'>{n_clusters}</b> Clusters</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: INTRODUCTION
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Introduction":

    st.markdown("""
    <div class='intro-hero'>
        <h1>📚 Audible Insights</h1>
        <p>
            An end-to-end intelligent audiobook recommendation system built on
            <b>3,535 Audible titles</b> using NLP, K-Means clustering, and five
            distinct recommendation models — from content-based filtering to a
            full hybrid engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    cols = st.columns(5)
    kpis = [
        (len(df),                            "Total Books",    "📦"),
        (df['Author'].nunique(),             "Unique Authors", "👤"),
        (df[df['Genre']!='Unknown']['Genre'].nunique(), "Genres", "🏷️"),
        (df['cluster'].nunique(),            "Clusters",       "🔵"),
        (f"{df['Rating'].mean():.2f}",       "Avg Rating",     "⭐"),
    ]
    for col, (val, label, icon) in zip(cols, kpis):
        with col:
            v = f"{val:,}" if isinstance(val, int) else str(val)
            st.markdown(f"""
            <div class='kpi-card'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div class='kpi-val'>{v}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>🎯 Project Objectives</h3></div>",
                    unsafe_allow_html=True)
        obj_cols = st.columns(2)
        objectives = [
            ("🔍","Data Analysis","Explore and understand patterns in Audible's audiobook catalog"),
            ("🤖","5 Models","Build Content, Cluster, Popularity, Genre & Hybrid recommenders"),
            ("📊","14 Questions","Answer Easy, Medium & Scenario-based analytical questions"),
            ("💡","Hidden Gems","Surface underrated books with high ratings but low popularity"),
            ("🎨","Interactive App","Streamlit dashboard for real-time book recommendations"),
            ("📈","Evaluation","Compare models using Precision@K and Coverage metrics"),
        ]
        for i, (icon, title, desc) in enumerate(objectives):
            with obj_cols[i % 2]:
                st.markdown(f"""
                <div class='obj-card'>
                    <div class='obj-icon'>{icon}</div>
                    <div class='obj-title'>{title}</div>
                    <div class='obj-desc'>{desc}</div>
                </div><br>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>🗄️ Datasets Used</h3></div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='dataset-card'>
            <div class='dataset-title'>📁 Audible_Catlog.csv</div>
            <div class='dataset-meta'>6,368 rows × 5 columns</div>
            <div class='dataset-meta' style='margin-top:6px; color:#374151;'>
                Book Name, Author, Rating, Description, Listening Time</div>
        </div>
        <div class='dataset-card'>
            <div class='dataset-title'>📁 Audible_Catlog_Advanced_Features.csv</div>
            <div class='dataset-meta'>4,464 rows × 8 columns</div>
            <div class='dataset-meta' style='margin-top:6px; color:#374151;'>
                Book Name, Author, Ranks & Genre, Number of Reviews, Price, etc.</div>
        </div>
        <div class='dataset-card' style='border-left:4px solid #3730a3;'>
            <div class='dataset-title'>✅ Merged & Cleaned Dataset</div>
            <div class='dataset-meta'>3,535 rows × 16 columns</div>
            <div class='dataset-meta' style='margin-top:6px; color:#374151;'>
                After deduplication, null handling, genre extraction & NLP preprocessing</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-header'><div class='dot'></div><h3>⚙️ Project Pipeline</h3></div>",
                unsafe_allow_html=True)
    pipe_cols = st.columns(3)
    steps = [
        (1,"Data Preparation",    "Merge datasets, handle nulls, remove duplicates, extract genres, parse listening time"),
        (2,"Exploratory Analysis","Answer 5 easy-level questions with interactive visualisations"),
        (3,"NLP Preprocessing",   "Tokenise descriptions, remove stopwords, apply TF-IDF vectorisation"),
        (4,"K-Means Clustering",  "Elbow method + Silhouette score → optimal K=19 clusters"),
        (5,"5 Rec. Models",       "Content-Based, Cluster, Popularity, Genre, and Hybrid models"),
        (6,"Streamlit App",       "Interactive dashboard answering all 14 questions in real time"),
    ]
    for i, (num, title, desc) in enumerate(steps):
        with pipe_cols[i % 3]:
            st.markdown(f"""
            <div class='pipeline-step'>
                <div class='step-num'>{num}</div>
                <div class='step-body'>
                    <div class='step-title'>{title}</div>
                    <div class='step-desc'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-header'><div class='dot'></div><h3>📋 Questions Answered</h3></div>",
                unsafe_allow_html=True)
    q_cols = st.columns(3)
    question_sets = [
        ("📗 Easy Level (EDA)","b-green",[
            "Most popular genres in the dataset?",
            "Authors with the highest-rated books?",
            "Average rating distribution?",
            "Trends in publication popularity?",
            "Ratings vs review count relationship?",
        ]),
        ("📘 Medium Level","b-indigo",[
            "Books frequently clustered together?",
            "Genre similarity in recommendations?",
            "Author popularity vs book ratings?",
            "Best feature combination for accuracy?",
        ]),
        ("📙 Scenario Based","b-amber",[
            "Sci-Fi user → Top 5 recommendations?",
            "Thriller fan → Similar books?",
            "Hidden gems: high rating, low popularity?",
        ]),
    ]
    for col, (title, badge_cls, qs) in zip(q_cols, question_sets):
        with col:
            st.markdown(f"<span class='badge {badge_cls}'>{title}</span><br><br>", unsafe_allow_html=True)
            for q in qs:
                st.markdown(f"<div style='font-size:0.84rem;color:#374151;padding:5px 0;border-bottom:1px solid #f3f4f6;'>✅ {q}</div>",
                            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: BOOK RECOMMENDER  — FIX: Dropdown replaces text input
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Book Recommender":
    st.markdown("<h2>🤖 Book Recommender</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;'>Select a book from the dropdown and choose a recommendation model.</p>",
                unsafe_allow_html=True)

    # FIX: Build sorted dropdown list once
    all_books = sorted(df['Book Name'].dropna().unique().tolist())

    col_inp, col_model, col_n = st.columns([2, 1.6, 0.8])
    with col_inp:
        # FIX: Searchable selectbox instead of text_input
        book_input = st.selectbox(
            "📖 Select Book Name",
            options=[""] + all_books,
            index=0,
            help="Start typing to search for a book title"
        )
    with col_model:
        model_choice = st.selectbox("🤖 Model", [
            "Model 1 — Content-Based",
            "Model 2 — Cluster-Based",
            "Model 3 — Popularity-Based",
            "Model 4 — Genre-Based",
            "Model 5 — Hybrid (Best)",
        ])
    with col_n:
        top_n = st.slider("Results", 3, 10, 5)

    # Live book preview using exact match (FIX: no more .str.contains)
    if book_input:
        preview = df[df['Book Name'] == book_input]
        if not preview.empty:
            r = preview.iloc[0]
            n_rev = int(r.get('Number of Reviews', 0))
            st.markdown(f"""
            <div class='answer-box'>
                ✅ <b>{r['Book Name']}</b> &nbsp;|&nbsp; 👤 {r['Author']}
                &nbsp;|&nbsp; ⭐ {r['Rating']} &nbsp;|&nbsp; 🏷️ {r['Genre']}
                &nbsp;|&nbsp; 💬 {n_rev:,} reviews
            </div>""", unsafe_allow_html=True)

    run_btn = st.button("🔍 Get Recommendations", use_container_width=True)

    if run_btn:
        if not book_input:
            st.warning("⚠️ Please select a book from the dropdown first.")
        else:
            with st.spinner("Finding best matches…"):
                model_num = int(model_choice.split()[1])
                if model_num == 1:
                    recs, matched = model1_content_based(df, cosine_sim, book_input, top_n)
                    score_col = 'Similarity Score'
                elif model_num == 2:
                    recs, matched = model2_cluster_based(df, book_input, top_n)
                    score_col = 'Cluster'
                elif model_num == 3:
                    recs, matched = model3_popularity_based(df, top_n=top_n)
                    matched = "Overall Catalog"
                    score_col = 'Popularity Score'
                elif model_num == 4:
                    recs, matched = model4_genre_based(df, book_input, top_n)
                    score_col = 'Filter'
                else:
                    recs, matched = model5_hybrid(df, cosine_sim, book_input, top_n)
                    score_col = 'hybrid_score'

            if recs is not None and not recs.empty:
                st.markdown(f"""
                <div class='answer-box'>
                    📖 <b>Input:</b> {matched} &nbsp;|&nbsp;
                    🤖 <b>Model:</b> {model_choice} &nbsp;|&nbsp;
                    📋 <b>{len(recs)} recommendations</b>
                </div>""", unsafe_allow_html=True)

                for _, row in recs.iterrows():
                    sv     = row.get(score_col, '')
                    sv_str = f"{sv:.4f}" if isinstance(sv, float) else str(sv)
                    desc   = str(row.get('Description',''))[:220] + '…'
                    n_rev  = int(row.get('Number of Reviews', 0))
                    st.markdown(f"""
                    <div class='book-card'>
                        <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                            <span class='book-title'>#{row['Rank']} &nbsp; {row['Book Name']}</span>
                            <div>
                                <span class='badge b-amber'>⭐ {row['Rating']}</span>
                                <span class='badge b-sky'>💬 {n_rev:,} reviews</span>
                            </div>
                        </div>
                        <div class='book-meta'>👤 {row['Author']} &nbsp;|&nbsp;
                            🏷️ {row.get('Genre','—')} &nbsp;|&nbsp;
                            Score: <b style='color:#3730a3;'>{sv_str}</b></div>
                        <div class='book-desc'>{desc}</div>
                    </div>""", unsafe_allow_html=True)

                with st.expander("📋 View as Table"):
                    disp = [c for c in ['Rank','Book Name','Author','Rating','Genre',score_col]
                            if c in recs.columns]
                    st.dataframe(recs[disp], use_container_width=True, hide_index=True)
            else:
                st.error("❌ No recommendations found. Try a different book.")

    # Medium Q4 comparison table
    st.markdown("<div class='sec-header'><div class='dot'></div><h3>📌 Medium Q4 — Which Combination is Most Accurate?</h3></div>",
                unsafe_allow_html=True)
    model_df = pd.DataFrame({
        'Model':       ['Content-Based','Cluster-Based','Popularity-Based','Genre-Based','Hybrid'],
        'Core Signal': ['Cosine Similarity','K-Means Cluster','Weighted Score','Genre Match','All Combined'],
        'Best For':    ['Similar descriptions','Thematic groups','New users','Genre fans','Best overall'],
        'Precision':   ['High','Medium','Low','High','⭐ Highest'],
    })
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='answer-box'>
        ✅ <b>Answer (Medium Q4):</b> The <b>Hybrid Model</b> is most accurate — combining
        Content (40%) + Cluster (25%) + Popularity (20%) + Genre (15%) into a single
        weighted score that outperforms any individual model.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD — FIX: All figures use safe_plot + reduced figsize
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.markdown("<h2>📊 EDA Dashboard — Easy Level Questions</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;'>Exploring 5 key questions about the Audible catalog.</p>",
                unsafe_allow_html=True)

    df_known = df[df['Genre'] != 'Unknown']
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Q1 · Genres", "Q2 · Authors", "Q3 · Ratings", "Q4 · Trends", "Q5 · Reviews"
    ])

    # ── Q1 ──────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Q1 — Most Popular Genres</h3></div>",
                    unsafe_allow_html=True)
        ng = st.slider("Top N genres", 5, 20, 12, key='q1')
        gc = df_known['Genre'].value_counts().head(ng)

        # FIX: figsize reduced from (10,5) to (8,4)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fc')
        bars = ax.barh(
            gc.index[::-1], gc.values[::-1],
            color=[PALETTE[i % len(PALETTE)] for i in range(ng)],
            edgecolor='none', height=0.62
        )
        for bar, val in zip(bars, gc.values[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', fontsize=9, color='#374151', fontweight='600')
        ax.set_xlabel('Number of Books', fontsize=10)
        ax.set_title(f'Top {ng} Most Popular Genres', fontsize=12,
                     fontweight='bold', color='#1a1a2e')
        ax.set_xlim(0, gc.max() * 1.2)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.xaxis.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        safe_plot(fig)   # FIX: use safe_plot

        t3 = gc.head(3)
        st.markdown(f"""
        <div class='answer-box'>
            ✅ <b>Answer:</b> Most popular genres are <b>{t3.index[0]}</b> ({t3.iloc[0]} books),
            <b>{t3.index[1]}</b> ({t3.iloc[1]} books), and <b>{t3.index[2]}</b> ({t3.iloc[2]} books).
        </div>""", unsafe_allow_html=True)

    # ── Q2 ──────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Q2 — Highest-Rated Authors</h3></div>",
                    unsafe_allow_html=True)
        mb = st.slider("Min books per author", 2, 10, 3, key='q2')
        astats = (df.groupby('Author')
                    .agg(avg_rating=('Rating','mean'), book_count=('Book Name','count'))
                    .reset_index())
        top_a = (astats[astats['book_count'] >= mb]
                 .sort_values('avg_rating', ascending=False)
                 .head(12)
                 .reset_index(drop=True))

        # FIX: figsize reduced from (10,5) to (8,4)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fc')
        y_pos = range(len(top_a))
        ax.barh(list(y_pos), top_a['avg_rating'].values,
                color='#3730a3', edgecolor='none', height=0.62, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_a['Author'].tolist(), fontsize=9)
        for i, row in top_a.iterrows():
            ax.text(row['avg_rating'] + 0.005, i,
                    f"{row['avg_rating']:.2f}  ({int(row['book_count'])} books)",
                    va='center', fontsize=8, color='#374151')
        ax.set_xlabel('Average Rating', fontsize=10)
        ax.set_title(f'Top Rated Authors (min {mb} books)', fontsize=12,
                     fontweight='bold', color='#1a1a2e')
        ax.set_xlim(4.0, 5.4)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.xaxis.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        safe_plot(fig)   # FIX: use safe_plot

        if not top_a.empty:
            best = top_a.iloc[0]
            st.markdown(f"""
            <div class='answer-box'>
                ✅ <b>Answer:</b> <b>{best['Author']}</b> has the highest average rating of
                <b>{best['avg_rating']:.2f}</b> across {int(best['book_count'])} books.
            </div>""", unsafe_allow_html=True)

    # ── Q3 ──────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Q3 — Rating Distribution</h3></div>",
                    unsafe_allow_html=True)
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("Mean Rating",   f"{df['Rating'].mean():.2f}")
        m2c.metric("Median Rating", f"{df['Rating'].median():.2f}")
        m3c.metric("Std Deviation", f"{df['Rating'].std():.2f}")

        # FIX: figsize reduced from (12,5) to (9,4)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.patch.set_facecolor('#ffffff')

        ax1 = axes[0]
        ax1.set_facecolor('#f8f9fc')
        ax1.hist(df['Rating'], bins=30, color='#6366f1', edgecolor='white', alpha=0.8, density=True)
        df['Rating'].plot.kde(ax=ax1, color='#3730a3', linewidth=2.5)
        ax1.axvline(df['Rating'].mean(),   color='#e11d48', linestyle='--', linewidth=2,
                    label=f"Mean: {df['Rating'].mean():.2f}")
        ax1.axvline(df['Rating'].median(), color='#10b981', linestyle=':',  linewidth=2,
                    label=f"Median: {df['Rating'].median():.2f}")
        ax1.set_title('Rating Distribution (KDE)', fontsize=11, fontweight='bold', color='#1a1a2e')
        ax1.set_xlabel('Rating'); ax1.set_ylabel('Density')
        ax1.legend(fontsize=8)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.xaxis.grid(True, alpha=0.4)
        ax1.set_axisbelow(True)

        ax2 = axes[1]
        ax2.set_facecolor('#f8f9fc')
        bins   = [0, 3, 3.5, 4, 4.2, 4.4, 4.6, 4.8, 5.0]
        labels = ['<3','3–3.5','3.5–4','4–4.2','4.2–4.4','4.4–4.6','4.6–4.8','4.8–5']
        df_tmp = df.copy()
        df_tmp['rb'] = pd.cut(df_tmp['Rating'], bins=bins, labels=labels)
        bc = df_tmp['rb'].value_counts().sort_index()
        ax2.bar(bc.index, bc.values,
                color=[PALETTE[i % len(PALETTE)] for i in range(len(bc))],
                edgecolor='white', alpha=0.9)
        for bar in ax2.patches:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     str(int(bar.get_height())), ha='center', va='bottom',
                     fontsize=8, color='#374151', fontweight='600')
        ax2.set_title('Books per Rating Bucket', fontsize=11, fontweight='bold', color='#1a1a2e')
        ax2.set_xlabel('Rating Range'); ax2.set_ylabel('Books')
        ax2.tick_params(axis='x', rotation=30)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.yaxis.grid(True, alpha=0.4)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        safe_plot(fig)   # FIX: use safe_plot

        st.markdown(f"""
        <div class='answer-box'>
            ✅ <b>Answer:</b> Mean = <b>{df['Rating'].mean():.2f}</b>,
            Median = <b>{df['Rating'].median():.2f}</b>.
            Most books fall between <b>4.4–4.7</b>. Distribution is left-skewed —
            Audible titles are generally highly rated.
        </div>""", unsafe_allow_html=True)

    # ── Q4 ──────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Q4 — Publication & Popularity Trends</h3></div>",
                    unsafe_allow_html=True)
        st.info("ℹ️ No publication year column in dataset — analysing popularity by Rating Tier.")

        df_tmp2 = df.copy()
        df_tmp2['Rating Tier'] = pd.cut(
            df_tmp2['Rating'],
            bins=[0, 3.5, 4.0, 4.3, 4.6, 5.0],
            labels=['Low (≤3.5)','Avg (3.5–4)','Good (4–4.3)','High (4.3–4.6)','Excellent (4.6–5)']
        )
        ts = (df_tmp2.groupby('Rating Tier', observed=True)
                     .agg(book_count=('Book Name','count'),
                          avg_reviews=('Number of Reviews','mean'))
                     .reset_index())

        # FIX: figsize reduced from (12,5) to (9,4)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.patch.set_facecolor('#ffffff')
        tier_colors = ['#e11d48','#f97316','#f59e0b','#10b981','#3730a3']

        ax1 = axes[0]; ax1.set_facecolor('#f8f9fc')
        bars = ax1.bar(ts['Rating Tier'], ts['book_count'],
                       color=tier_colors, edgecolor='white', alpha=0.9)
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     str(int(bar.get_height())), ha='center', va='bottom',
                     fontsize=9, color='#374151', fontweight='600')
        ax1.set_title('Books by Rating Tier', fontsize=11, fontweight='bold', color='#1a1a2e')
        ax1.set_ylabel('Number of Books')
        ax1.tick_params(axis='x', rotation=20)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.yaxis.grid(True, alpha=0.4); ax1.set_axisbelow(True)

        ax2 = axes[1]; ax2.set_facecolor('#f8f9fc')
        bars2 = ax2.bar(ts['Rating Tier'], ts['avg_reviews'],
                        color=tier_colors, edgecolor='white', alpha=0.9)
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     f"{bar.get_height():.0f}", ha='center', va='bottom',
                     fontsize=9, color='#374151', fontweight='600')
        ax2.set_title('Avg Reviews by Rating Tier', fontsize=11, fontweight='bold', color='#1a1a2e')
        ax2.set_ylabel('Avg Reviews')
        ax2.tick_params(axis='x', rotation=20)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.yaxis.grid(True, alpha=0.4); ax2.set_axisbelow(True)

        plt.tight_layout()
        safe_plot(fig)   # FIX: use safe_plot

        st.markdown("""
        <div class='answer-box'>
            ✅ <b>Answer:</b> <b>Excellent-rated books (4.6–5)</b> attract the most reviews,
            confirming that quality drives long-term popularity. Lower-rated books see
            significantly fewer reviews.
        </div>""", unsafe_allow_html=True)

    # ── Q5 ──────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Q5 — Ratings vs Review Count</h3></div>",
                    unsafe_allow_html=True)
        corr_val = df['Rating'].corr(df['Number of Reviews'])
        c1, c2 = st.columns(2)
        c1.metric("Correlation (Rating × Reviews)", f"{corr_val:.4f}")
        c2.metric("Interpretation", "Weak positive relationship")

        # FIX: figsize reduced from (12,5) to (9,4)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.patch.set_facecolor('#ffffff')

        ax1 = axes[0]; ax1.set_facecolor('#f8f9fc')
        sc = ax1.scatter(df['Number of Reviews'], df['Rating'],
                         alpha=0.2, s=12, c=df['Rating'],
                         cmap='RdYlGn', vmin=3.5, vmax=5.0, edgecolors='none')
        plt.colorbar(sc, ax=ax1, label='Rating', shrink=0.8)
        ax1.set_xscale('log')
        ax1.set_xlabel('Reviews (log scale)'); ax1.set_ylabel('Rating')
        ax1.set_title(f'Rating vs Reviews (corr={corr_val:.3f})',
                      fontsize=11, fontweight='bold', color='#1a1a2e')
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.xaxis.grid(True, alpha=0.4); ax1.yaxis.grid(True, alpha=0.4)

        ax2 = axes[1]; ax2.set_facecolor('#f8f9fc')
        df_tmp3 = df.copy()
        df_tmp3['rev_tier'] = pd.qcut(
            df_tmp3['Number of Reviews'], q=4,
            labels=['Low\n(Bot 25%)','Med-Low\n(25-50%)','Med-High\n(50-75%)','High\n(Top 25%)']
        )
        tier_data = [df_tmp3[df_tmp3['rev_tier'] == t]['Rating'].dropna().values
                     for t in df_tmp3['rev_tier'].cat.categories]
        bp = ax2.boxplot(tier_data, patch_artist=True,
                         medianprops=dict(color='white', linewidth=2),
                         whiskerprops=dict(color='#9ca3af'),
                         capprops=dict(color='#9ca3af'),
                         flierprops=dict(marker='o', alpha=0.2, markersize=3, color='#9ca3af'))
        for patch, color in zip(bp['boxes'], ['#3730a3','#6366f1','#10b981','#f59e0b']):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax2.set_xticklabels(['Low\n(Bot 25%)','Med-Low\n(25-50%)','Med-High\n(50-75%)','High\n(Top 25%)'],
                            fontsize=8)
        ax2.set_ylabel('Rating')
        ax2.set_title('Rating by Review Tier', fontsize=11, fontweight='bold', color='#1a1a2e')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.yaxis.grid(True, alpha=0.4); ax2.set_axisbelow(True)

        plt.tight_layout()
        safe_plot(fig)   # FIX: use safe_plot

        st.markdown(f"""
        <div class='answer-box'>
            ✅ <b>Answer:</b> Correlation = <b>{corr_val:.4f}</b> — weak positive.
            More reviews slightly correlates with higher ratings, but review count alone
            does not predict quality.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIO EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Scenario Explorer":
    st.markdown("<h2>🎯 Scenario Explorer — Medium & Scenario Questions</h2>", unsafe_allow_html=True)

    s1,s2,s3,s4,s5,s6,s7 = st.tabs([
        "🚀 Sci-Fi (S1)", "🔪 Thriller (S2)", "💎 Hidden Gems (S3)",
        "🔵 Clusters (M1)", "🏷️ Genre Sim (M2)", "👤 Author Pop (M3)", "🏆 Best Model (M4)"
    ])

    # ── S1 ──────────────────────────────────────────────────────────────────
    with s1:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Scenario 1 — New Sci-Fi User</h3></div>",
                    unsafe_allow_html=True)
        gkw   = st.text_input("Genre keyword", "science fiction", key='sc1')
        n_sc1 = st.slider("Recommendations", 3, 10, 5, key='sc1n')
        if st.button("🚀 Get Sci-Fi Recommendations", use_container_width=True):
            pool = df[df['Genre'].str.lower().str.contains(gkw.lower(), na=False)].copy()
            if pool.empty:
                pool = df.copy()
            recs = pool.sort_values('popularity_score', ascending=False).head(n_sc1)
            st.success(f"Found {len(pool)} books matching '{gkw}'")
            for _, row in recs.iterrows():
                n_rev = int(row.get('Number of Reviews', 0))
                desc  = str(row.get('Description',''))[:200] + '…'
                st.markdown(f"""
                <div class='book-card'>
                    <div style='display:flex;justify-content:space-between;'>
                        <span class='book-title'>🚀 {row['Book Name']}</span>
                        <span class='badge b-amber'>⭐ {row['Rating']}</span>
                    </div>
                    <div class='book-meta'>👤 {row['Author']} &nbsp;|&nbsp; 🏷️ {row['Genre']}
                        &nbsp;|&nbsp; 💬 {n_rev:,} reviews</div>
                    <div class='book-desc'>{desc}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='answer-box'>✅ <b>Answer (Scenario 1):</b> Top {n_sc1} Sci-Fi books
            recommended by popularity score within '<i>{gkw}</i>' genre.
            </div>""", unsafe_allow_html=True)

    # ── S2 ──────────────────────────────────────────────────────────────────
    with s2:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Scenario 2 — Thriller Fan</h3></div>",
                    unsafe_allow_html=True)
        tkw   = st.text_input("Seed genre", "thriller", key='sc2')
        n_sc2 = st.slider("Recommendations", 3, 10, 5, key='sc2n')
        if st.button("🔪 Get Thriller Recommendations", use_container_width=True):
            pool = df[df['Genre'].str.lower().str.contains(tkw.lower(), na=False)]
            seed = (pool.sort_values('Rating', ascending=False).iloc[0]['Book Name']
                    if not pool.empty
                    else df.sort_values('popularity_score', ascending=False).iloc[0]['Book Name'])
            st.info(f"📖 Seed book: **{seed}**")
            recs, _ = model5_hybrid(df, cosine_sim, seed[:50], top_n=n_sc2)
            if recs is not None and not recs.empty:
                for _, row in recs.iterrows():
                    n_rev = int(row.get('Number of Reviews', 0))
                    desc  = str(row.get('Description',''))[:200] + '…'
                    st.markdown(f"""
                    <div class='book-card' style='border-left-color:#e11d48;'>
                        <div style='display:flex;justify-content:space-between;'>
                            <span class='book-title'>#{row['Rank']} {row['Book Name']}</span>
                            <span class='badge b-rose'>⭐ {row['Rating']}</span>
                        </div>
                        <div class='book-meta'>👤 {row['Author']} &nbsp;|&nbsp; 🏷️ {row['Genre']}
                            &nbsp;|&nbsp; 🏆 {row['hybrid_score']:.4f}</div>
                        <div class='book-desc'>{desc}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='answer-box'>✅ <b>Answer (Scenario 2):</b> Hybrid Model seeded from
            top-rated <i>{tkw}</i> book to surface books with similar descriptions, cluster & genre.
            </div>""", unsafe_allow_html=True)

    # ── S3 ──────────────────────────────────────────────────────────────────
    with s3:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Scenario 3 — Hidden Gems</h3></div>",
                    unsafe_allow_html=True)
        hg_col1, hg_col2 = st.columns(2)
        min_r = hg_col1.slider("Min Rating", 4.0, 5.0, 4.5, 0.1)
        max_p = hg_col2.slider("Max Review Percentile", 0.10, 0.50, 0.25, 0.05)
        n_hg  = st.slider("Show N gems", 5, 20, 10)
        if st.button("💎 Find Hidden Gems", use_container_width=True):
            gems, thresh = get_hidden_gems(df, min_rating=min_r, max_reviews_pct=max_p)
            st.success(f"Found **{len(gems)}** hidden gems! (Rating ≥ {min_r}, Reviews ≤ {thresh:.0f})")
            for _, row in gems.head(n_hg).iterrows():
                n_rev = int(row.get('Number of Reviews', 0))
                desc  = str(row.get('Description',''))[:200] + '…'
                st.markdown(f"""
                <div class='book-card' style='border-left-color:#10b981;'>
                    <div style='display:flex;justify-content:space-between;'>
                        <span class='book-title'>💎 {row['Book Name']}</span>
                        <div>
                            <span class='badge b-green'>⭐ {row['Rating']}</span>
                            <span class='badge b-sky'>💬 {n_rev:,} reviews</span>
                        </div>
                    </div>
                    <div class='book-meta'>👤 {row['Author']} &nbsp;|&nbsp; 🏷️ {row['Genre']}</div>
                    <div class='book-desc'>{desc}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='answer-box'>✅ <b>Answer (Scenario 3):</b> <b>{len(gems)} hidden gems</b>
            identified — rated ≥ {min_r} but fewer than {thresh:.0f} reviews
            (bottom {int(max_p*100)}th percentile).
            </div>""", unsafe_allow_html=True)

    # ── M1 ──────────────────────────────────────────────────────────────────
    with s4:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Medium Q1 — Books Clustered Together</h3></div>",
                    unsafe_allow_html=True)
        cid = st.selectbox("Select Cluster", sorted(df['cluster'].unique()))
        cdf = df[df['cluster'] == cid].sort_values('Rating', ascending=False)
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("Books in Cluster", len(cdf))
        m2c.metric("Avg Rating", f"{cdf['Rating'].mean():.2f}")
        top_g = cdf[cdf['Genre'] != 'Unknown']['Genre'].value_counts()
        m3c.metric("Top Genre", top_g.index[0] if not top_g.empty else "—")
        for _, row in cdf.head(8).iterrows():
            st.markdown(f"""
            <div class='book-card'>
                <span class='book-title'>📚 {row['Book Name']}</span>
                <div class='book-meta'>👤 {row['Author']} &nbsp;|&nbsp;
                    🏷️ {row['Genre']} &nbsp;|&nbsp; ⭐ {row['Rating']}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='answer-box'>✅ <b>Answer (Medium Q1):</b> Cluster {cid} contains
        <b>{len(cdf)} books</b> grouped by description similarity (TF-IDF + K-Means).
        Books in the same cluster share thematic and linguistic patterns.
        </div>""", unsafe_allow_html=True)

    # ── M2 ──────────────────────────────────────────────────────────────────
    with s5:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Medium Q2 — Genre Similarity in Recommendations</h3></div>",
                    unsafe_allow_html=True)
        known_genres = sorted(df[df['Genre'] != 'Unknown']['Genre']
                               .value_counts().head(30).index.tolist())
        sg = st.selectbox("Select Genre", known_genres)
        if st.button("🔍 Analyse Genre", use_container_width=True):
            gdf   = df[df['Genre'] == sg]
            cdist = gdf['cluster'].value_counts()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Books in Genre",        len(gdf))
                st.metric("Avg Rating",            f"{gdf['Rating'].mean():.2f}")
                st.metric("Clusters Spread Across", gdf['cluster'].nunique())
                conc = (cdist.iloc[0] / len(gdf) * 100) if len(gdf) > 0 else 0
                st.markdown(f"""
                <div class='answer-box'>✅ <b>Answer (Medium Q2):</b> <b>{conc:.0f}%</b> of
                <i>{sg}</i> books fall in the top cluster, confirming genre similarity
                strongly aligns with cluster membership.
                </div>""", unsafe_allow_html=True)
            with c2:
                # FIX: figsize reduced
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#f8f9fc')
                tc = cdist.head(8)
                ax.bar(tc.index.astype(str), tc.values,
                       color=[PALETTE[i % len(PALETTE)] for i in range(len(tc))],
                       edgecolor='white', alpha=0.9)
                ax.set_title(f'Cluster Distribution — {sg}', fontsize=10,
                             fontweight='bold', color='#1a1a2e')
                ax.set_xlabel('Cluster ID'); ax.set_ylabel('Books')
                ax.spines[['top', 'right']].set_visible(False)
                ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
                plt.tight_layout()
                safe_plot(fig)   # FIX: use safe_plot

    # ── M3 ──────────────────────────────────────────────────────────────────
    with s6:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Medium Q3 — Author Popularity Effect on Ratings</h3></div>",
                    unsafe_allow_html=True)
        astats2 = df.groupby('Author').agg(
            avg_rating=('Rating','mean'),
            total_reviews=('Number of Reviews','sum')
        ).reset_index()
        astats2['popularity_tier'] = pd.qcut(
            astats2['total_reviews'], q=4,
            labels=['Low','Medium-Low','Medium-High','High']
        )
        corr2 = astats2['total_reviews'].corr(astats2['avg_rating'])

        c1, c2 = st.columns(2)
        c1.metric("Correlation (Reviews vs Rating)", f"{corr2:.4f}")
        c2.metric("Interpretation", "Popular authors score slightly higher")

        # FIX: figsize reduced from (12,5) to (9,4)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.patch.set_facecolor('#ffffff')

        ax1 = axes[0]; ax1.set_facecolor('#f8f9fc')
        ax1.scatter(astats2['total_reviews'], astats2['avg_rating'],
                    alpha=0.3, s=14, color='#6366f1', edgecolors='none')
        ax1.set_xscale('log')
        ax1.set_xlabel('Total Reviews (log)'); ax1.set_ylabel('Avg Rating')
        ax1.set_title('Author Popularity vs Rating', fontsize=11, fontweight='bold', color='#1a1a2e')
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.xaxis.grid(True, alpha=0.4); ax1.yaxis.grid(True, alpha=0.4)

        ax2 = axes[1]; ax2.set_facecolor('#f8f9fc')
        tdata = [astats2[astats2['popularity_tier'] == t]['avg_rating'].dropna()
                 for t in ['Low','Medium-Low','Medium-High','High']]
        bp = ax2.boxplot(tdata, patch_artist=True,
                         medianprops=dict(color='white', linewidth=2),
                         whiskerprops=dict(color='#9ca3af'),
                         capprops=dict(color='#9ca3af'),
                         flierprops=dict(marker='o', alpha=0.2, markersize=3, color='#9ca3af'))
        for patch, color in zip(bp['boxes'], ['#3730a3','#6366f1','#10b981','#f59e0b']):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax2.set_xticklabels(['Low','Med-Low','Med-High','High'], fontsize=9)
        ax2.set_ylabel('Avg Rating')
        ax2.set_title('Rating by Popularity Tier', fontsize=11, fontweight='bold', color='#1a1a2e')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.yaxis.grid(True, alpha=0.4); ax2.set_axisbelow(True)

        plt.tight_layout()
        safe_plot(fig)   # FIX: use safe_plot

        st.markdown(f"""
        <div class='answer-box'>✅ <b>Answer (Medium Q3):</b> Correlation = <b>{corr2:.4f}</b>.
        Popular authors (more total reviews) score slightly higher, but the effect is weak —
        book quality matters more than author fame.
        </div>""", unsafe_allow_html=True)

    # ── M4 ──────────────────────────────────────────────────────────────────
    with s7:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>Medium Q4 — Best Feature Combination</h3></div>",
                    unsafe_allow_html=True)
        wdf = pd.DataFrame({
            'Model':      ['Content-Based','Cluster-Based','Popularity','Genre-Based','Hybrid'],
            'Content %':  [100, 0,   0,   0,  40],
            'Cluster %':  [0, 100,   0,   0,  25],
            'Popular %':  [0,   0, 100,   0,  20],
            'Genre %':    [0,   0,   0, 100,  15],
            'Precision':  ['High','Medium','Low','High','⭐ Highest'],
        })
        st.dataframe(wdf, use_container_width=True, hide_index=True)

        # FIX: figsize reduced from (11,5) to (9,4)
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#f8f9fc')
        models = wdf['Model'].tolist()
        x = np.arange(len(models)); w = 0.18
        ax.bar(x - 1.5*w, wdf['Content %'], w, label='Content',    color='#3730a3', alpha=0.85, edgecolor='white')
        ax.bar(x - 0.5*w, wdf['Cluster %'], w, label='Cluster',    color='#6366f1', alpha=0.85, edgecolor='white')
        ax.bar(x + 0.5*w, wdf['Popular %'], w, label='Popularity', color='#10b981', alpha=0.85, edgecolor='white')
        ax.bar(x + 1.5*w, wdf['Genre %'],   w, label='Genre',      color='#f59e0b', alpha=0.85, edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel('Weight (%)', fontsize=10)
        ax.set_title('Feature Weights per Model', fontsize=12, fontweight='bold', color='#1a1a2e')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        plt.tight_layout()
        safe_plot(fig)   # FIX: use safe_plot

        st.markdown("""
        <div class='answer-box'>
            ✅ <b>Answer (Medium Q4):</b> The <b>Hybrid Model</b> is most accurate:<br>
            &nbsp;&nbsp;• Content similarity <b>40%</b> — captures description-level similarity<br>
            &nbsp;&nbsp;• Cluster membership <b>25%</b> — captures thematic grouping<br>
            &nbsp;&nbsp;• Popularity score <b>20%</b> — favours well-loved books<br>
            &nbsp;&nbsp;• Genre match <b>15%</b> — ensures category relevance
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT CREATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "👩‍💻 About Creator":
    st.markdown("<h2>👩‍💻 About the Creator</h2>", unsafe_allow_html=True)

    col_card, col_skills = st.columns([1, 1.4])

    with col_card:
        st.markdown(f"""
        <div class='creator-card'>
            <img src='data:image/jpeg;base64,{PHOTO_B64}' class='creator-photo' alt='Kavya S'>
            <div class='creator-name'>Kavya S</div>
            <div class='creator-role'>Data Science & Machine Learning Engineer</div>
            <div style='margin:20px 0 8px 0; font-size:0.82rem; opacity:0.75; letter-spacing:0.08em;'>CONNECT WITH ME</div>
            <div>
                <a href='mailto:kavya22s145@gmail.com' class='contact-link'>
                    📧 kavya22s145@gmail.com
                </a><br>
                <a href='https://www.linkedin.com/in/kavya-s1245/' target='_blank' class='contact-link'>
                    💼 linkedin.com/in/kavya-s1245
                </a><br>
                <a href='https://github.com/Kavya1245' target='_blank' class='contact-link'>
                    🐙 github.com/Kavya1245
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_skills:
        st.markdown("<div class='sec-header'><div class='dot'></div><h3>🛠️ Technical Expertise</h3></div>",
                    unsafe_allow_html=True)
        skill_groups = [
            ("🐍 Languages & Frameworks", ["Python","SQL","Streamlit","Flask"]),
            ("🤖 Machine Learning",       ["Scikit-learn","NLP (NLTK, TF-IDF)","K-Means Clustering","Recommendation Systems"]),
            ("📊 Data & Visualisation",   ["Pandas","NumPy","Matplotlib","Seaborn"]),
            ("☁️ Tools & Platforms",      ["Git & GitHub","Jupyter Notebook","VS Code","AWS (Basic)"]),
        ]
        for group_name, skills in skill_groups:
            st.markdown(f"**{group_name}**")
            badges = " ".join([f"<span class='badge b-indigo'>{s}</span>" for s in skills])
            st.markdown(f"<div style='margin-bottom:14px;'>{badges}</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-header'><div class='dot'></div><h3>📌 About This Project</h3></div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='dataset-card'>
            <div style='color:#374151; font-size:0.88rem; line-height:1.8;'>
                This project was built as part of a Data Science course to demonstrate
                end-to-end ML development — from raw data ingestion, EDA, NLP preprocessing,
                and unsupervised clustering, to building five distinct recommendation models
                and deploying them in an interactive Streamlit application.<br><br>
                It answers <b>14 analytical questions</b> across Easy, Medium and Scenario levels,
                surfacing actionable insights from the Audible audiobook catalog.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sec-header'><div class='dot'></div><h3>📊 Project Stats</h3></div>",
                    unsafe_allow_html=True)
        stat_cols = st.columns(3)
        for col, (val, label) in zip(stat_cols, [
            ("3,535", "Books Analysed"),
            ("5",     "ML Models Built"),
            ("14",    "Questions Answered"),
        ]):
            with col:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-val'>{val}</div>
                    <div class='kpi-label'>{label}</div>
                </div>""", unsafe_allow_html=True)