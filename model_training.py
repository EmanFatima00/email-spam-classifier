"""
Email Spam Classification — Advanced NLP Pipeline
Author: Eman Fatima | BS-AI @ PAF-IAST | ML Intern @ ProSensia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re, os, json, joblib, warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "spam.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  EMAIL SPAM CLASSIFICATION — ADVANCED NLP PIPELINE")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df = df[['Category', 'Message']].copy()
df.columns = ['label', 'message']
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n✅ Dataset loaded: {len(df)} messages")
print(f"   Ham  (legit): {(df['label_num']==0).sum()} ({(df['label_num']==0).mean()*100:.1f}%)")
print(f"   Spam:         {(df['label_num']==1).sum()} ({(df['label_num']==1).mean()*100:.1f}%)")

# ── 2. Feature Engineering ────────────────────────────────────
df['msg_length']     = df['message'].apply(len)
df['word_count']     = df['message'].apply(lambda x: len(x.split()))
df['num_digits']     = df['message'].apply(lambda x: sum(c.isdigit() for c in x))
df['num_uppercase']  = df['message'].apply(lambda x: sum(c.isupper() for c in x))
df['num_special']    = df['message'].apply(lambda x: len(re.findall(r'[!?$£€]', x)))
df['has_url']        = df['message'].apply(lambda x: int(bool(re.search(r'http|www|\.com', x, re.I))))
df['has_phone']      = df['message'].apply(lambda x: int(bool(re.search(r'\d{10,}', x))))
df['uppercase_ratio']= df['num_uppercase'] / (df['msg_length'] + 1)
print("\n✅ Feature engineering — 8 new text features added")

# ── 3. Text Preprocessing ─────────────────────────────────────
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

def preprocess(text):
    text   = re.sub(r'http\S+|www\S+', 'URL', text)
    text   = re.sub(r'\d+', 'NUM', text)
    text   = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text   = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(preprocess)
print("✅ Text preprocessing complete (lemmatization + stopwords + URL/NUM handling)")

# ── 4. Train/Test Split ───────────────────────────────────────
X_text = df['clean_message']
y      = df['label_num']
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42)

# ── 5. TF-IDF Vectorization ───────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),       # unigrams + bigrams
    sublinear_tf=True,
    min_df=2,
)
X_train = tfidf.fit_transform(X_train_text)
X_test  = tfidf.transform(X_test_text)
print(f"✅ TF-IDF vectorization — vocab size: {len(tfidf.vocabulary_):,} features")

# ── 6. Train Models ───────────────────────────────────────────
models = {
    'Naive Bayes':         MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(C=10, max_iter=1000, random_state=42),
    'SVM':                 CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42)),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)

results = {}
cv_skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    proba    = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_skf, scoring='f1')
    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall':    recall_score(y_test, preds),
        'f1':        f1_score(y_test, preds),
        'roc_auc':   roc_auc_score(y_test, proba),
        'cv_f1':     cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'preds':     preds,
        'proba':     proba,
    }
    print(f"\n📌 {name}")
    print(f"   Accuracy  : {results[name]['accuracy']*100:.2f}%")
    print(f"   Precision : {results[name]['precision']:.4f}")
    print(f"   Recall    : {results[name]['recall']:.4f}")
    print(f"   F1        : {results[name]['f1']:.4f}")
    print(f"   ROC-AUC   : {results[name]['roc_auc']:.4f}")
    print(f"   CV F1     : {results[name]['cv_f1']:.4f} ± {results[name]['cv_std']:.4f}")

# ── 7. Best Model ─────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['f1'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name}")
print(f"   Accuracy  : {best['accuracy']*100:.2f}%")
print(f"   F1 Score  : {best['f1']:.4f}")
print(f"   ROC-AUC   : {best['roc_auc']:.4f}")
print(f"\n{classification_report(y_test, best['preds'], target_names=['Ham','Spam'])}")

# ── 8. Save Model ─────────────────────────────────────────────
joblib.dump(best['model'], os.path.join(OUTPUT_DIR, "spam_model.pkl"))
joblib.dump(tfidf,         os.path.join(OUTPUT_DIR, "tfidf.pkl"))

metrics = {
    "accuracy":    round(best['accuracy']*100, 2),
    "precision":   round(best['precision']*100, 2),
    "recall":      round(best['recall']*100, 2),
    "f1":          round(best['f1']*100, 2),
    "roc_auc":     round(best['roc_auc'], 4),
    "best_model":  best_name,
    "total_msgs":  len(df),
    "spam_count":  int((df['label_num']==1).sum()),
    "ham_count":   int((df['label_num']==0).sum()),
}
json.dump(metrics, open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"))
print("✅ Model, vectorizer and metrics saved to outputs/")

# ── 9. Visualizations ─────────────────────────────────────────
print("\n📊 Generating visualizations...")
COLORS = ['#059669','#DC2626']

# A. Model Comparison
names_ = list(results.keys())
accs_  = [results[n]['accuracy']*100 for n in names_]
f1s_   = [results[n]['f1']           for n in names_]
aucs_  = [results[n]['roc_auc']      for n in names_]
cols_  = ['#6366F1','#059669','#D97706','#0891B2','#DC2626']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')
for ax, vals, title, xlim in zip(
        axes, [accs_, f1s_, aucs_],
        ["Accuracy (%)","F1 Score","ROC-AUC"],
        [(90,100),(0.9,1.0),(0.95,1.0)]):
    ax.barh(names_, vals, color=cols_)
    ax.set_title(title); ax.set_xlim(*xlim)
    for i, v in enumerate(vals):
        lbl = f"{v:.1f}%" if "Accuracy" in title else f"{v:.4f}"
        ax.text(v+0.001 if "Accuracy" not in title else v+0.02,
                i, lbl, va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# B. ROC Curves
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})", linewidth=2)
plt.plot([0,1],[0,1],'gray',linestyle=':')
plt.title("ROC Curves — All Models", fontsize=13, fontweight='bold')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc='lower right', fontsize=9); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

# C. Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.title(f"Confusion Matrix — {best_name}", fontweight='bold')
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()

# D. Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, label, title, color in zip(
        axes, [0, 1], ['Ham Messages', 'Spam Messages'],
        ['Blues', 'Reds']):
    text = ' '.join(df[df['label_num']==label]['clean_message'])
    wc   = WordCloud(width=600, height=300, background_color='white',
                     colormap=color, max_words=100).generate(text)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off'); ax.set_title(title, fontsize=13, fontweight='bold')
plt.suptitle("Most Common Words — Ham vs Spam", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds.png"), dpi=150, bbox_inches='tight')
plt.close()

# E. Message Length Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for col, ax, title in zip(
        ['msg_length','word_count'], axes,
        ['Message Length (chars)','Word Count']):
    for label, color, lname in zip([0,1], COLORS, ['Ham','Spam']):
        data = df[df['label_num']==label][col]
        ax.hist(data, bins=40, alpha=0.6, color=color, label=lname, density=True)
    ax.set_title(f'{title} Distribution', fontweight='bold')
    ax.set_xlabel(title); ax.set_ylabel('Density')
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "length_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()

# F. Top Spam Keywords
spam_text  = ' '.join(df[df['label_num']==1]['clean_message'])
spam_words = Counter(spam_text.split()).most_common(15)
words, counts = zip(*spam_words)
plt.figure(figsize=(10, 5))
bars = plt.barh(list(reversed(words)), list(reversed(counts)), color='#DC2626')
plt.title("Top 15 Spam Keywords", fontsize=13, fontweight='bold')
plt.xlabel("Frequency")
for bar, count in zip(bars, list(reversed(counts))):
    plt.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
             str(count), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "spam_keywords.png"), dpi=150, bbox_inches='tight')
plt.close()

print("✅ All visualizations saved to outputs/")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {best['accuracy']*100:.2f}%")
print(f"  F1 Score   : {best['f1']:.4f}")
print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
print("=" * 60)
