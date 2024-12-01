# BERT Model for Text Classification
A BERT-based model for text classification, implemented using TensorFlow and HuggingFace Transformers.
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer-based model developed by Google. It is designed to improve the understanding of natural language processing (NLP) tasks by learning context from both directions (left-to-right and right-to-left) in a text.

## Text - Preprocessing Steps:
1. **Tokenization**: Split text into individual words or subwords using a tokenizer.
2. **Lowercasing**: Convert all text to lowercase to maintain uniformity.
3. **Removing Punctuation**: Strip unnecessary punctuation marks from the text.
4. **Stopword Removal**: Remove common but unimportant words (e.g., "the", "is").
5. **Stemming/Lemmatization**: Reduce words to their root or base form.
6. **Padding**: Ensure text sequences are of consistent length, often done with zeros.
7. **Truncation**: Shorten text sequences that exceed the maximum allowed length.
8. **Encoding**: Convert text into numerical format (e.g., word indices or embeddings).

## Model Architecture:

| **Layer (type)**      | **Output Shape**     | **Param #** | **Connected to**               |
|-----------------------|----------------------|-------------|--------------------------------|
| input_ids (InputLayer) | (None, 128)          | 0           | -                              |
| attention_mask (InputLayer) | (None, 128)     | 0           | -                              |
| token_type_ids (InputLayer) | (None, 128)    | 0           | -                              |
| bert_layer (BertLayer) | (None, 128, 768)     | 0           | input_ids[0][0], attention_mask[0][0], token_type_ids[0][0] |
| get_item (GetItem)     | (None, 768)          | 0           | bert_layer[0][0]               |
| dropout (Dropout)      | (None, 768)          | 0           | get_item[0][0]                 |
| dense (Dense)          | (None, 3)            | 2,307       | dropout[0][0]                  |

- **Total params**: 2,307 (9.01 KB)
- **Trainable params**: 2,307 (9.01 KB)
- **Non-trainable params**: 0 (0.00 B)
  
## Classification Report:
```
               precision    recall  f1-score   support

           0       0.33      0.27      0.29      1103
           1       0.29      0.60      0.39      1001
           2       0.84      0.57      0.68      2711

    accuracy                           0.51      4815
   macro avg       0.49      0.48      0.45      4815
weighted avg       0.61      0.51      0.53      4815
```

## Confusion Matrix:
<img src="https://github.com/leovidith/BERT-Transformer/blob/main/images/Confusion%20matrix.png" alt="Model Architecture" width="600"/>

## Prediction:
```python
Enter the string: this is one excellent product
The predicted sentiment is: Positive
```
