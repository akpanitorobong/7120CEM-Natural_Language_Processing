
# 🧠 Natural Language Processing Coursework: Temporal and Context-Aware Sentiment Analysis on Twitter

This repository presents two NLP projects exploring **sentiment analysis** on Twitter data through different lenses:

1. **📆 Part 1 – Cross-Temporal Domain Adaptation:**  
   How well do classic ML models generalize sentiment classification across time (2013–2017)?

2. **🎭 Part 2 – Sarcasm-Aware Sentiment Analysis:**  
   How can we improve sentiment analysis by incorporating sarcasm detection using deep learning?

---

## 🔍 Overview

| Part | Focus | Techniques | Models |
|------|-------|------------|--------|
| **1** | Domain Adaptation Over Time | TF-IDF, BoW, ML classifiers | Random Forest, SVM, Logistic Regression, Naïve Bayes |
| **2** | Sarcasm-Aware Deep Learning | Multi-task Learning, Transformers | BERTweet, Multi-task BERT, Sentiment & Sarcasm classifiers |

---

## 📂 Project Contents

| File | Description |
|------|-------------|
| [`7120_CW.ipynb`](https://github.com/akpanitorobong/7120CEM-Natural_Language_Processing/blob/main/7120_CW.ipynb) | Part 1: Cross-temporal sentiment analysis with traditional ML |
| [`7120_CW2.ipynb`](https://github.com/akpanitorobong/7120CEM-Natural_Language_Processing/blob/main/7120_CW2.ipynb) | Part 2: Sarcasm-aware sentiment analysis using BERTweet |


---

## 📆 Part 1: Domain Adaptation in Sentiment Analysis

### 🧪 Objective
Assess how sentiment classifiers degrade over time using Twitter data from 2013, 2016 (train), and 2017 (test).

### 🔧 Methods
- **Feature Representations:** Bag of Words (BoW), TF-IDF
- **Models:** Random Forest, Gradient Boosting, SVM, Logistic Regression, Naïve Bayes
- **Data Source:** SemEval 2017 Task 4
- **Preprocessing:** Cleaning, normalization, emoji/text conversion, lemmatization

### 📊 Key Results

| Model (TF-IDF) | F1 Score 2013 | F1 Score 2016 | Performance Drop |
|----------------|---------------|---------------|------------------|
| Logistic Regression | 32.7 | 53.13 | **-20.47** |
| Gradient Boosting   | 26.1 | 49.27 | **-23.17** |
| SVM                 | 32.5 | 53.02 | **-20.52** |

### 💡 Insights
- Logistic Regression is most **stable over time**
- Tree-based models (Random Forest, GB) **perform best on newer data**
- Feature choice (BoW vs. TF-IDF) has **minor impact**

---

## 🎭 Part 2: Sarcasm-Aware Sentiment Analysis (Deep Learning)

### 🎯 Objective
Build a deep learning framework that **jointly detects sarcasm and sentiment** using Twitter data.

### 🤖 Models
- **Model 1 – Sarcasm Detector:** BERTweet + Binary Classifier (F1 = 0.96)
- **Model 2 – Baseline Sentiment Classifier:** BERTweet + 3-class Sentiment Classifier (F1 = 0.72)
- **Model 3 – Multi-Task Classifier:** Shared BERTweet encoder, joint sentiment/sarcasm heads (F1 = 0.63 / 0.99)

### 📦 Dataset
- SemEval-2017 (Sentiment-labeled tweets)
- Additional sarcastic tweets with sentiment labels
- Custom combined dataset with `text`, `sentiment`, `sarcastic`

### 📈 Key Metrics

| Model | Sentiment F1 | Sarcasm F1 | Notes |
|-------|--------------|------------|-------|
| Sarcasm Detector | N/A | 0.96 | Binary classification |
| Baseline Sentiment | 0.72 | 0.88 | No sarcasm input |
| Sarcasm-Aware (Multi-Task) | 0.63 | **0.99** | Improved on sarcastic tweets |

### 💡 Insights
- Sarcasm detection improves overall **robustness**
- Joint learning offers **better interpretability** on ambiguous inputs
- BERTweet performs well due to **Twitter pretraining**

---

## 🧠 Key Takeaways

- Static ML models degrade across time, highlighting need for **frequent retraining** or **adaptive architectures**
- Incorporating linguistic nuance (sarcasm) can significantly boost **context-aware performance**
- Transformer-based models like BERTweet outperform traditional ML on complex tasks like sarcasm + sentiment

---

## 📌 Technologies Used

- Python (Jupyter Notebook)
- Scikit-learn, Pandas, Numpy
- HuggingFace Transformers
- PyTorch
- NLTK / Text Processing
- Google Colab

---
