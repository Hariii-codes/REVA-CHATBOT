"""
REVA University Chatbot - NLP Training Script

This script trains a simple NLP model using NLTK and scikit-learn
for intent classification on the REVA University chatbot dataset.

Usage:
    python train.py
"""

import json
import pickle
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets if not already present."""
    nltk_data = ['punkt', 'wordnet', 'omw-1.4']
    for data in nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            try:
                nltk.download(data, quiet=True)
            except:
                pass
        try:
            nltk.data.find(f'corpora/{data}')
        except LookupError:
            try:
                nltk.download(data, quiet=True)
            except:
                pass

download_nltk_data()


class ChatbotTrainer:
    """Handles training of the REVA University chatbot model."""

    def __init__(self, intents_file='intents.json'):
        """
        Initialize the trainer.

        Args:
            intents_file: Path to the intents JSON file
        """
        self.intents_file = intents_file
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.intents = None
        self.model = None
        self.vectorizer = None

    def load_intents(self):
        """Load intents from JSON file."""
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

    def preprocess_training_data(self):
        """
        Preprocess training data from intents.

        Creates:
        - words: vocabulary list
        - classes: intent tags
        - documents: (words, tag) pairs
        """
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word in pattern
                word_list = nltk.word_tokenize(pattern)
                # Add to words list (lemmatized)
                self.words.extend([self.lemmatizer.lemmatize(w.lower()) for w in word_list])
                # Add to documents with tag
                self.documents.append((word_list, intent['tag']))

            # Add tag to classes if not exists
            if intent['tag'] not in self.classes:
                self.classes.append(intent['tag'])

        # Remove duplicates and sort
        self.words = sorted(list(set([w for w in self.words if w.isalpha() or w.replace('.', '').replace('?', '').replace('!', '').replace(',', '').isalnum()])))
        self.classes = sorted(list(set(self.classes)))

        print(f"Vocabulary size: {len(self.words)}")
        print(f"Number of intents: {len(self.classes)}")
        print(f"Training documents: {len(self.documents)}")

    def create_bag_of_words(self, document_words):
        """
        Create bag of words representation.

        Args:
            document_words: List of words in a document

        Returns:
            numpy array representing bag of words
        """
        bag = [0] * len(self.words)
        document_words = [self.lemmatizer.lemmatize(w.lower()) for w in document_words]

        for i, word in enumerate(self.words):
            if word in document_words:
                bag[i] = 1

        return np.array(bag)

    def prepare_training_data(self):
        """Prepare training data for model training."""
        training_data = []
        output_labels = []

        for document in self.documents:
            bag = self.create_bag_of_words(document[0])
            training_data.append(bag)
            output_labels.append(document[1])

        # Create label encoder
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(output_labels)

        return np.array(training_data), np.array(encoded_labels)

    def train_model(self):
        """Train the classification model."""
        # Prepare data
        X_train, y_train = self.prepare_training_data()

        # Train Logistic Regression model
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        )

        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Model training completed!")

        # Calculate training accuracy
        train_accuracy = self.model.score(X_train, y_train)
        print(f"Training accuracy: {train_accuracy:.2f}")

    def save_model(self):
        """Save trained model and related files."""
        # Save words vocabulary
        with open('words.pkl', 'wb') as f:
            pickle.dump(self.words, f)

        # Save classes
        with open('classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)

        # Save model and label encoder
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        with open('model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print("Model files saved successfully!")

    def train(self):
        """Execute complete training pipeline."""
        print("=" * 50)
        print("REVA University Chatbot - Training")
        print("=" * 50)

        # Load intents
        print("Loading intents...")
        self.load_intents()

        # Preprocess data
        print("Preprocessing training data...")
        self.preprocess_training_data()

        # Train model
        self.train_model()

        # Save model
        self.save_model()

        print("\nTraining completed successfully!")
        print(f"Model files created: words.pkl, classes.pkl, model.pkl")


def main():
    """Main function to run training."""
    trainer = ChatbotTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
