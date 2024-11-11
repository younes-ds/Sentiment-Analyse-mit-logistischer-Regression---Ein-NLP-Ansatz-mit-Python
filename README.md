# Sentiment Analysis Project

## Description

This project implements a sentiment analysis model using natural language processing (NLP) techniques, specifically a Logistic Regression model. It preprocesses text data, converts it into a TF-IDF representation, and trains a model to predict sentiments.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Results](#results)

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis.git
    ```

2. Navigate into the project directory:
    ```bash
    cd sentiment-analysis
    ```

3. Install the dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download the dataset and place it in the project folder (make sure the file is named `Sentiment Analysis Dataset.csv`).

2. Run the sentiment analysis script:
    ```bash
    python sentiment_analysis.py
    ```

3. The output will display the training and test accuracy, classification report, and confusion matrix.

## Preprocessing

The preprocessing steps include:
- Removal of HTML tags and URLs.
- Lowercasing the text.
- Removal of non-alphabetic characters.
- Tokenization, stopwords removal, and lemmatization.

## Model

The sentiment analysis model is built using:
- **Logistic Regression** for classification.
- **TF-IDF** for text feature extraction.

## Results

The model achieves the following performance:

- **Training Accuracy**: 83.81%
- **Test Accuracy**: 70.90%

### Training Classification Report

Classification Report on Training Data:
              precision    recall  f1-score   support

           0       0.85      0.80      0.83      3817
           1       0.83      0.87      0.85      4183

    accuracy                           0.84      8000
   macro avg       0.84      0.84      0.84      8000
weighted avg       0.84      0.84      0.84      8000

### Test Classification Report

Classification Report on Test Data:
              precision    recall  f1-score   support

           0       0.71      0.63      0.67       923
           1       0.71      0.78      0.74      1077

    accuracy                           0.71      2000
   macro avg       0.71      0.70      0.70      2000
weighted avg       0.71      0.71      0.71      2000