# Amazon Polarity Sentiment Classifier using Hugging Face Transformers

This repository implements a sentiment analysis pipeline using Hugging Face Transformers and the [Amazon Polarity Dataset](https://huggingface.co/datasets/amazon_polarity). The project includes data preprocessing, model fine-tuning with DistilBERT, and evaluation using metrics such as accuracy, precision, recall, and F1-score. Additionally, it demonstrates how to classify new text inputs with the fine-tuned model.

---

## Features

- Fine-tunes a `DistilBERT` model for binary sentiment classification (`positive` or `negative`).
- Preprocesses the Amazon Polarity dataset for training and testing.
- Evaluates the model using accuracy, precision, recall, and F1-score.
- Provides an inference pipeline for classifying new review texts.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/amazon-polarity-sentiment-classifier.git
   cd amazon-polarity-sentiment-classifier
   ```

2. Install dependencies:
   ```bash
   pip install transformers datasets scikit-learn
   ```

---

## Usage

### 1. Run the Script
Execute the main script to train the model and perform sentiment analysis:
```bash
python sentiment_classifier.py
```

### 2. Customize Input Texts
Modify the `texts` list in the script to test the model with your own input reviews:
```python
texts = [
    "This product exceeded my expectations!",
    "The quality was disappointing and not worth the price."
]
```

### 3. Analyze Sentiments
The output will display each text with its predicted sentiment (`positive` or `negative`) and confidence score.

---

## Example Output

```plaintext
Review 1: This product exceeded my expectations!
Sentiment: Positive
Confidence: 0.97

Review 2: The quality was disappointing and not worth the price.
Sentiment: Negative
Confidence: 0.85
```

---

## Project Structure

```
amazon-polarity-sentiment-classifier/
├── sentiment_classifier.py   # Main script for training and inference
├── README.md                 # Project documentation
```

---

## Key Functions

- **`load_data()`**: Loads the Amazon Polarity dataset.
- **`preprocess_data()`**: Tokenizes and prepares the dataset for training.
- **`train_model()`**: Fine-tunes DistilBERT and evaluates the model.
- **`analyze_sentiments()`**: Analyzes custom input reviews and predicts sentiments.

---

## Dataset

The [Amazon Polarity Dataset](https://huggingface.co/datasets/amazon_polarity) consists of customer reviews labeled as:
- **Positive (1)**: Indicates favorable reviews.
- **Negative (0)**: Indicates unfavorable reviews.

---

## Model Details

- **Base Model**: `distilbert-base-uncased`
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Optimizer**: AdamW
