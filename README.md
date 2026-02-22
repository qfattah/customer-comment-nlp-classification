# End-to-End NLP & Customer Feedback Classification

## Project Overview

This project builds an end-to-end Natural Language Processing (NLP) and predictive modeling pipeline to analyze customer comments from a fuel retail company operating 250+ gas stations across the U.S.

Customer phone feedback is merged with structured numeric data using loyalty card identifiers. The goal is to extract meaningful themes from text data and improve prediction of a business target variable using both unstructured and structured data.

The project integrates:

- Text preprocessing & cleaning
- Word frequency analysis
- Word embeddings (Word2Vec)
- Topic modeling (LDA)
- TF-IDF + SVD dimensionality reduction
- Decision Tree classification
- Structured + text feature fusion

---

## Business Context

The company collects customer comments via phone surveys and stores them alongside numeric operational data.

Key columns:
- **Cust_ID** – Unique customer identifier
- **Comment** – Unstructured text feedback
- **Target** – Business outcome variable (binary classification)

Objective:
- Extract meaningful themes from customer comments
- Integrate textual insights with structured features
- Improve prediction of the target variable

---

## Technical Approach

### 1. Text Preprocessing

Performed standard NLP cleaning steps:

- Lowercasing
- Stopword removal (NLTK)
- Custom stopword refinement
- Punctuation removal
- Stemming (PorterStemmer)

Dimensionality of the potential document-feature matrix (DFM) was examined before and after cleaning.

---

### 2. Exploratory Text Analysis

- Frequency analysis (Top 20 words)
- WordCloud visualization to identify dominant themes

---

### 3. Document-Term Matrix Construction

- Tokenization of corpus
- Gensim dictionary creation
- Filtered rare and overly frequent terms (`no_below=5`, `no_above=0.75`)
- Created Bag-of-Words DFM

---

### 4. Word Embeddings

Trained a **Word2Vec (Skip-gram)** model to identify semantic similarity between terms.

Example:
- Most similar words to “service” were extracted to analyze contextual patterns.

---

### 5. Topic Modeling (LDA)

Applied Latent Dirichlet Allocation:

- Number of topics: 3
- 40 training passes

Used **pyLDAvis** for interactive topic visualization.

---

### 6. Predictive Modeling — Baseline (Structured Data Only)

Removed `Cust_ID` and `Comment`.

Trained a Decision Tree classifier:

- `max_depth=4`
- `min_samples_split=30`

Evaluated validation accuracy.

---

### 7. Feature Engineering — TF-IDF + SVD

- Applied TF-IDF weighting to DFM
- Reduced dimensionality using Latent Semantic Indexing (LSI/SVD)
- Extracted 6 latent semantic components
- Combined SVD components with structured numeric features

---

### 8. Predictive Modeling — Text + Structured Features

Trained a second Decision Tree model using:

- Structured features
- Latent semantic components (SVD)

Compared performance against baseline model.

---

## Model Evaluation

Models were evaluated using:

- Validation Accuracy
- Decision Tree structure interpretation

The hybrid model incorporating text-derived semantic features demonstrated worse predictive capability compared to structured-only modeling.

---

## Key Skills Demonstrated

- Natural Language Processing
- Text Cleaning & Preprocessing
- Word Embeddings (Word2Vec)
- Topic Modeling (LDA)
- TF-IDF & Dimensionality Reduction (SVD / LSI)
- Structured + Unstructured Data Integration
- Decision Tree Classification
- Model Comparison & Validation

---

## Impact & Practical Application

This project demonstrates how unstructured customer feedback can be transformed into structured analytical signals and integrated into predictive modeling frameworks.

The methodology can be extended to:

- Customer satisfaction modeling
- Sentiment-driven forecasting
- Operational quality monitoring
- Retail performance analytics

---

## Tools & Libraries

- Python
- Pandas
- NumPy
- NLTK
- Gensim
- WordCloud
- Scikit-learn
- pyLDAvis
- Matplotlib

---

## Author

Qusai Fattah  
Data Analytics | NLP | Predictive Modeling | Applied Machine Learning
