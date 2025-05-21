# bert-base-uncased
This is a BERT-based sentiment analysis model fine-tuned on a binary classification task (positive or negative sentiment). It is built on top of the bert-base-uncased model from Hugging Face Transformers and trained on movie reviews (e.g., IMDb dataset).
# BERT Fine-Tuned for Sentiment Analysis

This is a `bert-base-uncased` model fine-tuned on a sentiment analysis dataset. It predicts whether an input sentence expresses a **positive** or **negative** sentiment.

## Model Details

- **Base model**: `bert-base-uncased`
- **Task**: Binary Sentiment Classification
- **Framework**: PyTorch + Hugging Face Transformers
- **Labels**: 
  - `0` → Negative  
  - `1` → Positive

## Example Usage

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
model = BertForSequenceClassification.from_pretrained("your-username/bert-finetuned-imdb")
tokenizer = BertTokenizer.from_pretrained("your-username/bert-finetuned-imdb")

# Predict
inputs = tokenizer("This movie was fantastic!", return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()

print("Sentiment:", "Positive" if prediction == 1 else "Negative")
