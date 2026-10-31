# Comparative Analysis of ML, Deep Learning & BERT for Sentiment Classification

Comparative evaluation of machine learning, deep learning, and transformer-based models for binary sentiment classification of IMDB movie reviews.

## Dataset
- **Dataset:** IMDB movie reviews (`data_mr.csv`)
- **Samples:** 15,000
- **Train-test split:** 80:20
- **Classes:** Positive (1), Negative (0)
- **Evaluation metric:** F1-score

## Preprocessing & Feature Extraction
- Text cleaning using Gensim and custom regular expressions
- Bag-of-Words (BoW) and TF-IDF
- Word2Vec embeddings
- BERT subword tokenization

## Models Implemented

**Traditional Machine Learning**
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)

**Deep Learning**
- Feed-Forward Neural Network
- 1D Convolutional Neural Network (CNN)
- Gated Recurrent Unit (GRU)
- Bidirectional LSTM (BiLSTM)

**Transformer**
- Fine-tuned BERT

## Experimental Results

| Rank | Model | F1-score |
|---:|---|---:|
| 1 | BERT | **0.9135** |
| 2 | BoW + Multinomial Naive Bayes | 0.9119 |
| 3 | TF-IDF + SVM | 0.9048 |
| 4 | BoW + Feed-Forward Neural Network | 0.9002 |
| 5 | TF-IDF + Multinomial Naive Bayes | 0.8993 |
| 6 | BoW + Logistic Regression | 0.8935 |
| 7 | TF-IDF + Logistic Regression | 0.8922 |
| 8 | Tokenizer + BiLSTM | 0.8926 |
| 9 | Tokenizer + CNN | 0.8841 |
| 10 | TF-IDF + KNN | 0.8818 |
| 11 | BoW + SVM | 0.8764 |
| 12 | Tokenizer + GRU | 0.8721 |
| 13 | BoW + Random Forest | 0.8483 |
| 14 | TF-IDF + Random Forest | 0.8260 |
| 15 | Word2Vec + Feed-Forward Neural Network | 0.7654 |
| 16 | BoW + KNN | 0.6928 |
| 17 | Word2Vec + Logistic Regression | 0.6820 |
| 18 | Word2Vec + Random Forest | 0.6557 |
| 19 | Word2Vec + SVM | 0.6274 |
| 20 | Word2Vec + KNN | 0.5934 |

## Technology Stack
- Python
- pandas
- Gensim
- scikit-learn
- TensorFlow / Keras
- Hugging Face Transformers
