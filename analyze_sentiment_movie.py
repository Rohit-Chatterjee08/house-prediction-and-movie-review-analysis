# analyze_sentiment_kaggle.py

import pandas as pd
from transformers import pipeline

# --- 1. Load the Pre-trained NLP Model ---
print("--- Loading Sentiment Analysis Model ---")
# This is the same as before; it downloads a powerful pre-trained model.
sentiment_classifier = pipeline("sentiment-analysis")
print("Model loaded successfully.")
print("-" * 20)

# --- 2. Load and Sample the Data ---
print("\n--- Loading and Sampling IMDb Data ---")
try:
    df = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found. Please download it from Kaggle and place it in the same directory.")
    exit()

# The dataset has 50,000 reviews. Let's analyze a small random sample of 5.
# You can increase this number to test more.
sample_df = df.sample(100, random_state=50)

texts_to_analyze = sample_df['review'].tolist()
true_labels = sample_df['sentiment'].tolist()

print(f"Analyzing {len(texts_to_analyze)} random reviews...")
print("-" * 20)

# --- 3. Make Predictions ---
print("\n--- Analyzing Sentiments ---")
results = sentiment_classifier(texts_to_analyze)

# --- 4. Display the Results and Compare with True Labels ---
for i in range(len(texts_to_analyze)):
    review_text = texts_to_analyze[i]
    true_label = true_labels[i]
    pred_label = results[i]['label']
    pred_score = results[i]['score']

    # Truncate the review for cleaner printing
    short_review = (review_text[:100] + '...') if len(review_text) > 100 else review_text

    print(f"Review: '{short_review}'")
    print(f"  -> True Label:      {true_label.upper()}")
    print(f"  -> Predicted Label: {pred_label} (Confidence: {pred_score:.2%})")
    print("-" * 10)