# 📧 Email Spam Classification System

> **Advanced NLP-powered spam detection with 98.45% accuracy**
> Built by Eman Fatima | BS-AI @ PAF-IAST | ML Intern @ ProSensia

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Live Demo
**[👉 Click here to try the live app](YOUR_STREAMLIT_URL_HERE)**

---

## 📌 Project Overview

This project builds an advanced NLP pipeline to classify SMS and email messages as spam
or legitimate (ham). The system compares 5 machine learning models with TF-IDF bigram
features, NLTK lemmatization, and 8 engineered text features, then deploys the best
model as an interactive Streamlit web application.

---

## 🏆 Results

| Model                  | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Naive Bayes            | 98.06%   | 0.9426    | 0.8984 | 0.9200   | 0.9862  |
| Logistic Regression    | 98.16%   | 0.9739    | 0.8750 | 0.9218   | 0.9924  |
| **SVM ✅**             | **98.45%** | **0.9746** | **0.8984** | **0.9350** | **0.9920** |
| Random Forest          | 98.16%   | 1.0000    | 0.8516 | 0.9198   | 0.9927  |
| Gradient Boosting      | 97.48%   | 0.9811    | 0.8125 | 0.8889   | 0.9786  |

**Final Model: SVM | Accuracy: 98.45% | F1: 0.9350 | ROC-AUC: 0.9920**

---

## 🔬 NLP Pipeline

```
Raw Messages (5,157 SMS/Email)
       ↓
Text Cleaning (URL → URL token, numbers → NUM token)
       ↓
NLTK Lemmatization + Stopword Removal
       ↓
Feature Engineering (8 new text features)
       ↓
TF-IDF Vectorization (unigrams + bigrams, 5,504 features)
       ↓
Train/Test Split (80/20, Stratified)
       ↓
5 Models Trained + 5-Fold Cross Validation
       ↓
Best Model by F1 Score → SVM Deployed on Streamlit
```

---

## ✨ Key Features

- **5 ML models** trained and compared
- **TF-IDF with bigrams** — captures two-word patterns like "free prize", "call now"
- **NLTK preprocessing** — lemmatization, stopword removal, URL/number normalization
- **8 engineered features** — message length, word count, digit count, uppercase ratio, special chars, URL flag, phone number flag, uppercase ratio
- **Interactive Streamlit app** with real-time prediction, confidence split bar, message analysis panel, and 6 example messages
- **6 visualizations** — word clouds, spam keywords, ROC curves, confusion matrix, length distributions

---

## 📊 Dataset

- **Source:** UCI SMS Spam Collection Dataset
- **Size:** 5,157 messages (after deduplication)
- **Split:** Ham: 87.6% | Spam: 12.4%

---

## ⚙️ Run Locally

```bash
git clone https://github.com/EmanFatima00/email-spam-classifier.git
cd email-spam-classifier
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

---

## 📁 Project Structure

```
email-spam-classifier/
│
├── app.py                  # Streamlit web application
├── model_training.py       # Full NLP + ML pipeline
├── spam.csv                # Dataset
├── requirements.txt        # Dependencies
├── README.md               # This file
│
└── outputs/
    ├── spam_model.pkl           # Trained SVM model
    ├── tfidf.pkl                # TF-IDF vectorizer
    ├── metrics.json             # Model performance metrics
    ├── model_comparison.png     # Accuracy/F1/ROC-AUC comparison
    ├── roc_curves.png           # ROC curves for all models
    ├── confusion_matrix.png     # Confusion matrix
    ├── wordclouds.png           # Ham vs Spam word clouds
    ├── spam_keywords.png        # Top 15 spam keywords
    └── length_distribution.png  # Message length analysis
```

---

## 👩‍💻 Author

**Eman Fatima**
BS Artificial Intelligence — Semester 6 | PAF-IAST, Pakistan
ML Intern @ ProSensia | HR Manager @ CtrlAltCrew | Dean's List (SGPA: 3.72)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/eman-fatima-99962230b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/EmanFatima00)

---

*⭐ Star this repo if you found it useful!*
