# Benötigte Bibliotheken importieren
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK-Ressourcen herunterladen (falls noch nicht vorhanden)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialisieren von Lemmatizer und Stoppwort-Liste
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Zufallskeim setzen für Reproduzierbarkeit
np.random.seed(42)

# Textvorverarbeitungsfunktion definieren
def preprocess_text(text):
    # HTML-Tags entfernen
    text = re.sub(r'<[^>]+>', '', text)

    # URLs entfernen
    text = re.sub(r'http\S+', '', text)

    # In Kleinbuchstaben umwandeln
    text = text.lower()

    # Nicht-alphabetische Zeichen entfernen
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenisierung
    tokens = word_tokenize(text)

    # Stoppwörter entfernen
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatisierung
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Tokens wieder zu einem String verbinden
    return " ".join(tokens)

# Datensatz laden
data = pd.read_csv(r"C:\Users\chemi\Documents\Python\Projects\Sentiment-Analyse mit logistischer Regression - Ein NLP-Ansatz mit Python\Sentiment Analysis Dataset.csv", encoding='ISO-8859-1')

# Nicht benötigte Spalte entfernen
data.drop(columns=['ItemID'], inplace=True)

# Arbeitsdatensatz auf 10.000 Zeilen begrenzen
data_sample = data.sample(10000, random_state=42)

# Vorverarbeitung auf den Text anwenden
data_sample['Cleaned_SentimentText'] = data_sample['SentimentText'].apply(preprocess_text)

# Vorverarbeitete Daten anzeigen
print(data_sample[['SentimentText', 'Cleaned_SentimentText']].head())

# ------------------------------ Feature Engineering mit TF-IDF ------------------------------
# TF-IDF-Vectorizer initialisieren und auf den Text anwenden
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data_sample['Cleaned_SentimentText'])

# ------------------------------ Train-Test Split ------------------------------
# Daten in Merkmale und Zielwerte trennen
X = tfidf_matrix
y = data_sample['Sentiment']

# Train-Test-Split (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------ Logistische Regression ------------------------------
# Modell initialisieren
logreg_model = LogisticRegression(max_iter=1000)

# Modell trainieren
logreg_model.fit(X_train, y_train)

# Vorhersagen für den Trainings- und Testdatensatz
y_train_pred = logreg_model.predict(X_train)
y_test_pred = logreg_model.predict(X_test)

# Genauigkeit berechnen und ausgeben
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Klassifikationsbericht für den Testdatensatz
print("Classification Report on Test Data:")
print(classification_report(y_test, y_test_pred))

# Konfusionsmatrix für den Testdatensatz
print("Confusion Matrix on Test Data:")
print(confusion_matrix(y_test, y_test_pred))
