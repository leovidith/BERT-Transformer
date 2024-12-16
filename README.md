# BERT Model for Text Classification

## Overview

This project implements a **BERT-based** model for text classification, utilizing **TensorFlow** and **HuggingFace Transformers**. BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer-based model developed by Google, which improves the understanding of natural language by capturing context from both directions (left-to-right and right-to-left) in text.

## Results

The following confusion matrix shows the performance of the BERT model in classifying text into different categories:

<img src="https://github.com/leovidith/BERT-Transformer/blob/main/images/Confusion%20matrix.png" alt="Confusion Matrix" width="600"/>

## Features

- **Preprocessing Steps**:
  - **Tokenization**: Splits text into individual words or subwords.
  - **Lowercasing**: Converts all text to lowercase.
  - **Removing Punctuation**: Strips unnecessary punctuation marks.
  - **Stopword Removal**: Removes common but unimportant words (e.g., "the", "is").
  - **Stemming/Lemmatization**: Reduces words to their root or base form.
  - **Padding**: Ensures uniform sequence length.
  - **Truncation**: Shortens text sequences exceeding the maximum length.
  - **Encoding**: Converts text into numerical format (e.g., word indices or embeddings).

- **Model Architecture**:
  - **BERT Layer**: A pre-trained BERT transformer for feature extraction.
  - **Dense Layer**: Classifies the extracted features into three categories (positive, negative, neutral).
  - **Dropout Layer**: Helps prevent overfitting by randomly dropping connections during training.
  
- **Total Parameters**: 2,307 (9.01 KB), all of which are trainable.

## Sprint Features

### Sprint 1: Data Preprocessing
- Implement preprocessing steps like tokenization, lowercasing, stopword removal, and padding.
- **Deliverable**: A cleaned and tokenized dataset ready for training.

### Sprint 2: Model Architecture Design
- Design the BERT-based model for text classification.
- **Deliverable**: A complete model architecture with pre-trained BERT and a classification head.

### Sprint 3: Model Training
- Train the BERT model using the processed dataset and evaluate performance.
- **Deliverable**: A trained model with evaluation metrics like accuracy, precision, recall, and F1 score.

### Sprint 4: Model Evaluation
- Evaluate the model using a classification report and confusion matrix.
- **Deliverable**: Performance metrics to assess the effectiveness of the trained model.

## Conclusion

The BERT model for text classification successfully classifies text into predefined categories. With an accuracy of 51% and balanced precision and recall across categories, the model shows promise for NLP tasks. Future improvements can include fine-tuning the model further or experimenting with other transformer-based architectures for better performance.

Let me know if you need any further adjustments!
