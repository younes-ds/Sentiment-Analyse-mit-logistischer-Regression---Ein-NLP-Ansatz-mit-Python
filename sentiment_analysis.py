import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re  # Import re module for regex operations
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Import evaluation metrics

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# Define the preprocessing function
def preprocess_text(text):
    # Remove HTML tags using re.sub()
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs using re.sub()
    text = re.sub(r'http\S+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove non-alphabetic characters using re.sub()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    return " ".join(tokens)

# Load dataset with the specified encoding
data = pd.read_csv(r"C:\Users\chemi\Documents\Python\Sentiment Analysis Dataset.csv", encoding='ISO-8859-1')

# Drop the 'ItemID' column as it's not needed
data.drop(columns=['ItemID'], inplace=True)

# Increase the dataset size to 10,000 rows (adjust this as needed)
large_data = data.sample(10000, random_state=random_seed)

# Apply preprocessing to each text in 'SentimentText' column
large_data['Cleaned_SentimentText'] = large_data['SentimentText'].apply(preprocess_text)

# Print the first few rows with cleaned text
print(large_data[['SentimentText', 'Cleaned_SentimentText']].head())

# ------------------------------ Feature Engineering with TF-IDF ------------------------------
# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to the top 5000 features for efficiency

# Fit and transform the cleaned text to get the TF-IDF feature matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(large_data['Cleaned_SentimentText'])

# Convert the TF-IDF matrix to a DataFrame for easier inspection
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Print the shape of the TF-IDF feature matrix and the first few rows
print("TF-IDF feature matrix shape:", tfidf_df.shape)
print(tfidf_df.head())

# ------------------------------ Train-Test Split ------------------------------
# Splitting the data into training and test sets
X = tfidf_df  # Features (TF-IDF values)
y = large_data['Sentiment']  # Target variable (sentiment labels)

# Perform the train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Print the shape of the resulting train and test sets
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# ------------------------------ Logistic Regression ------------------------------
# Initialize the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000, random_state=random_seed)

# Train the Logistic Regression model
logreg_model.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = logreg_model.predict(X_train)

# Evaluate the model performance on the training data
accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("Classification Report on Training Data:")
print(classification_report(y_train, y_train_pred))

# Print confusion matrix
print("Confusion Matrix on Training Data:")
print(confusion_matrix(y_train, y_train_pred))

# Make predictions on the test data
y_test_pred = logreg_model.predict(X_test)

# Evaluate the model performance on the test data
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print detailed classification report for test data
print("Classification Report on Test Data:")
print(classification_report(y_test, y_test_pred))

# Print confusion matrix for test data
print("Confusion Matrix on Test Data:")
print(confusion_matrix(y_test, y_test_pred))
