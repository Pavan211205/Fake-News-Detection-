Project Overview
* This project implements a deep learning solution for classifying news articles as REAL or FAKE using TensorFlow and Natural Language Processing techniques. The model uses LSTM networks to analyze text content and detect patterns indicative of misinformation.

Key Features
* Text Preprocessing Pipeline: Comprehensive cleaning, tokenization, stopword removal, and lemmatization

* Deep Learning Architecture: LSTM-based neural network with embedding layers for text understanding

* Performance Metrics: Complete evaluation with accuracy, precision, recall, and F1-score

* Visualization: Training history plots and confusion matrix for model interpretation

* Model Persistence: Saved model and tokenizer for deployment and future use

Technical Implementation
* Data Preprocessing
* Text normalization (lowercasing, special character removal)

* Tokenization and stopword elimination using NLTK

* Lemmatization for word standardization

* Label encoding (FAKE → 0, REAL → 1)

Model Architecture
* Embedding Layer: Converts words to dense vectors

* LSTM Layers: Capture sequential patterns in text (64 and 32 units)

* Dense Layers: Feature learning with ReLU activation

* Dropout: Regularization to prevent overfitting (0.5 rate)

* Output Layer: Sigmoid activation for binary classification

Training Configuration
* Loss Function: Binary Crossentropy

* Optimizer: Adam

* Early Stopping: Prevents overfitting by monitoring validation loss

* Batch Size: 32 samples

* Epochs: 10 (with early stopping)

Performance Metrics
* The model achieves:

* Accuracy: 90%

* Precision: 92% for FAKE, 89% for REAL

* Recall: 92% for FAKE, 89% for REAL

* F1-score: 0.92 for FAKE, 0.89 for REAL

Dataset Information
* The dataset contains 20 news articles with:

* 12 FAKE news samples

* 8 REAL news samples

Sample articles cover political topics, international events, and social issues, providing diverse text patterns for the model to learn.
