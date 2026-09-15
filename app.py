"""
Email Spam Classification App
Author: Eman Fatima | BS-AI @ PAF-IAST | ML Intern @ ProSensia
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, re, os, json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #6366F1, #4F46E5);
        padding: 2rem; border-radius: 12px; text-align: center;
        color: white; margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.9; margin: 0.5rem 0 0; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #6366F1;
    }
    .metric-card h3 { font-size: 1.8rem; color: #6366F1; margin: 0; }
    .metric-card p  { color: #64748B; margin: 0; font-size: 0.9rem; }
    .result-spam {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border: 2px solid #DC2626; border-radius: 12px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .result-ham {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        border: 2px solid #059669; border-radius: 12px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .example-btn {
        background: #EEF2FF; border-radius: 8px;
        padding: 0.8rem; margin: 0.3rem 0;
        cursor: pointer; font-size: 0.85rem;
        color: #4338CA; border-left: 3px solid #6366F1;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load(os.path.join(OUTPUT_DIR, "spam_model.pkl"))
    tfidf  = joblib.load(os.path.join(OUTPUT_DIR, "tfidf.pkl"))
    mpath  = os.path.join(OUTPUT_DIR, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    return model, tfidf, metrics

model, tfidf, metrics = load_model()

# ── Preprocessing ─────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text   = re.sub(r'http\S+|www\S+', 'URL', text)
    text   = re.sub(r'\d+', 'NUM', text)
    text   = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text   = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def predict(message):
    cleaned = preprocess(message)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return pred, proba, cleaned

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📧 Email Spam Classifier</h1>
    <p>AI-powered spam detection using advanced NLP — SVM + TF-IDF with bigrams</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><h3>{metrics.get("accuracy","98.45")}%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><h3>{metrics.get("roc_auc","0.992")}</h3><p>ROC-AUC</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><h3>{metrics.get("total_msgs","5157")}</h3><p>Messages Trained</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><h3>5</h3><p>Models Compared</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📧 Classify Message", "📊 Model Insights", "ℹ️ About"])

with tab1:
    col_main, col_examples = st.columns([2, 1])

    with col_examples:
        st.markdown("### 💡 Try these examples")
        examples = {
            "🚨 Spam 1": "WINNER!! Congratulations! You have won a 1000 cash prize. Call now to claim your reward!",
            "🚨 Spam 2": "FREE entry in 2 a weekly competition to win FA Cup final tkts! Text FA to 87121",
            "🚨 Spam 3": "URGENT! Your Mobile No has been awarded a £2,000 Bonus Caller Prize. Call 09066660802",
            "✅ Ham 1":  "Hey, are we still meeting for lunch tomorrow? Let me know!",
            "✅ Ham 2":  "Can you pick up some groceries on your way home? We need milk and eggs.",
            "✅ Ham 3":  "The meeting has been rescheduled to 3pm. Please update your calendar.",
        }
        for label, msg in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state['example_msg'] = msg

    with col_main:
        st.markdown("### ✍️ Enter a message to classify")
        default_msg = st.session_state.get('example_msg', '')
        message = st.text_area(
            "Paste or type your email/SMS message here:",
            value=default_msg,
            height=160,
            placeholder="Type or paste a message here..."
        )
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        if clear_btn:
            st.session_state['example_msg'] = ''
            st.rerun()

    if classify_btn and message.strip():
        pred, proba, cleaned = predict(message)
        spam_pct = proba[1] * 100
        ham_pct  = proba[0] * 100

        st.markdown("---")
        res_col, detail_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="result-spam">
                    <h1>🚨 SPAM</h1>
                    <h2 style="color:#DC2626;font-size:2.5rem;margin:0">{spam_pct:.1f}%</h2>
                    <p style="color:#7F1D1D;">Probability of being Spam</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ham">
                    <h1>✅ HAM (Legitimate)</h1>
                    <h2 style="color:#059669;font-size:2.5rem;margin:0">{ham_pct:.1f}%</h2>
                    <p style="color:#064E3B;">Probability of being Legitimate</p>
                </div>""", unsafe_allow_html=True)

            # Confidence bar
            fig_b, ax_b = plt.subplots(figsize=(5, 1.8))
            ax_b.barh([''], [spam_pct], color='#DC2626', height=0.4, label='Spam')
            ax_b.barh([''], [ham_pct], left=[spam_pct], color='#059669', height=0.4, label='Ham')
            ax_b.set_xlim(0, 100)
            ax_b.set_title("Confidence Split", fontweight='bold', fontsize=10)
            ax_b.legend(loc='upper right', fontsize=8)
            ax_b.text(spam_pct/2, 0, f"{spam_pct:.1f}%", ha='center', va='center',
                      color='white', fontweight='bold', fontsize=10)
            ax_b.text(spam_pct + ham_pct/2, 0, f"{ham_pct:.1f}%", ha='center', va='center',
                      color='white', fontweight='bold', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close()

        with detail_col:
            st.markdown("#### 📝 Message Analysis")
            st.markdown(f"**Original length:** {len(message)} characters")
            st.markdown(f"**Word count:** {len(message.split())} words")
            st.markdown(f"**Has URL:** {'Yes ⚠️' if bool(re.search(r'http|www|\.com', message, re.I)) else 'No ✅'}")
            st.markdown(f"**Has phone number:** {'Yes ⚠️' if bool(re.search(r'\d{{10,}}', message)) else 'No ✅'}")
            st.markdown(f"**Uppercase ratio:** {sum(c.isupper() for c in message)/max(len(message),1)*100:.1f}%")
            st.markdown(f"**Exclamation marks:** {message.count('!')}")
            st.markdown("**Cleaned tokens:**")
            st.code(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)

    elif classify_btn:
        st.warning("Please enter a message to classify.")

with tab2:
    st.markdown("### 📊 Model Performance & Analysis")
    c1, c2 = st.columns(2)
    with c1:
        for img, cap in [("model_comparison.png","Model Comparison"),
                          ("confusion_matrix.png","Confusion Matrix")]:
            path = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(path):
                st.image(path, caption=cap, use_container_width=True)
    with c2:
        for img, cap in [("roc_curves.png","ROC Curves"),
                          ("spam_keywords.png","Top Spam Keywords")]:
            path = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(path):
                st.image(path, caption=cap, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        path = os.path.join(OUTPUT_DIR, "wordclouds.png")
        if os.path.exists(path):
            st.image(path, caption="Word Clouds — Ham vs Spam", use_container_width=True)
    with c4:
        path = os.path.join(OUTPUT_DIR, "length_distribution.png")
        if os.path.exists(path):
            st.image(path, caption="Message Length Distribution", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Model Results Summary")
    perf = {
        "Model":     ["Naive Bayes","Logistic Regression","SVM ✅","Random Forest","Gradient Boosting"],
        "Accuracy":  ["98.06%","98.16%","98.45%","98.16%","97.48%"],
        "Precision": ["0.9426","0.9739","0.9746","1.0000","0.9811"],
        "Recall":    ["0.8984","0.8750","0.8984","0.8516","0.8125"],
        "F1 Score":  ["0.9200","0.9218","0.9350","0.9198","0.8889"],
        "ROC-AUC":   ["0.9862","0.9924","0.9920","0.9927","0.9786"],
    }
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 🔬 About This Project
        Advanced NLP spam detection system trained on 5,157
        real-world SMS/email messages from the UCI SMS Spam Collection Dataset.

        **Pipeline Highlights:**
        - ✅ 5 ML models trained and compared
        - ✅ TF-IDF with bigrams (5,504 features)
        - ✅ NLTK lemmatization + stopword removal
        - ✅ URL and number normalization
        - ✅ 8 engineered text features
        - ✅ 5-fold stratified cross-validation
        - ✅ Final: SVM (Accuracy: 98.45%, ROC-AUC: 0.9920)

        **Dataset:** UCI SMS Spam Collection
        - 5,157 messages (after deduplication)
        - 87.6% Ham | 12.4% Spam
        """)
    with c2:
        st.markdown("""
        ### 👩‍💻 Developer

        **Eman Fatima**
        BS Artificial Intelligence — Semester 6 | PAF-IAST

        - 🏢 ML Intern @ ProSensia
        - 🎓 Dean's List — SGPA 3.72
        - 🤖 HR Manager @ CtrlAltCrew

        **Tech Stack:**
        Python · Scikit-learn · NLTK · WordCloud · Streamlit · TF-IDF · SVM

        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/eman-fatima-99962230b)
        [![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/EmanFatima00)
        """)
