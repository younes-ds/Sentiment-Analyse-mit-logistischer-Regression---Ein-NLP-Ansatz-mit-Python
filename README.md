# Sentimentanalyse-Projekt

In diesem Projekt geht es darum, die Stimmung von Textdaten zu analysieren, indem wir maschinelles Lernen und Textvorverarbeitung verwenden. Die Analyse basiert auf einem großen Datensatz von Textnachrichten, und das Modell wird mit logistischer Regression trainiert, um positive und negative Stimmungen zu klassifizieren.

**Quelle**: Der Datensatz wird lokal geladen und enthält Stimmungsbewertungen von Textnachrichten.

## **Ziel**:
Vorhersagen, ob eine Nachricht positiv oder negativ ist, basierend auf der Textanalyse und maschinellem Lernen.

## **Schritte**:
1. **Datenvorverarbeitung**: Bereinigung von Textdaten, Tokenisierung, Stopwort-Entfernung und Lemmatisierung.
2. **Feature Engineering**: Erstellung eines TF-IDF-Merkmalsraums für die Textdaten.
3. **Modelltraining**: Trainieren eines Modells der logistischen Regression auf einem Trainingsdatensatz.
4. **Modellbewertung**: Bewertung des Modells durch Genauigkeit, Klassifikationsbericht und Konfusionsmatrix.

## **Merkmale**:
- `SentimentText`: Textinhalt der Nachricht
- `Sentiment`: Stimmung (0 = negativ, 1 = positiv)

## **Datenvorverarbeitung**:
- HTML-Tags und URLs entfernen
- Alle Zeichen in Kleinbuchstaben umwandeln
- Nicht-alphabetische Zeichen entfernen
- Tokenisierung, Stopwort-Entfernung und Lemmatisierung der Wörter

## **Feature Engineering**:
- **TF-IDF**: Erstellen eines TF-IDF-Vektors mit den häufigsten 5000 Wörtern, um Textdaten in numerische Werte umzuwandeln.

## **Modellierung**:
- **Logistische Regression**: Modell wird mit einem Train-Test-Split von 80:20 trainiert und getestet.

## **Bewertung**:
- **Genauigkeit**: Die Gesamtgenauigkeit des Modells für Trainings- und Testdaten.
- **Klassifikationsbericht**: Darstellung von Präzision, Recall und F1-Score für das Testset.
- **Konfusionsmatrix**: Zeigt die wahren Positiven, falschen Positiven, wahren Negativen und falschen Negativen.

## **Abhängigkeiten**:
- `pandas`, `nltk`, `re`, `numpy`, `scikit-learn`

### **Installation**:
Um die benötigten Abhängigkeiten zu installieren, führen Sie aus:
```bash
pip install pandas nltk numpy scikit-learn
