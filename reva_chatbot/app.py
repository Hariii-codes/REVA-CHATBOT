"""
REVA University Chatbot - Flask Backend Server

This Flask application serves the chatbot API and frontend interface.
It loads the trained NLP model and processes user messages.

Usage:
    python app.py

Then visit: http://127.0.0.1:5000
"""

import os
import json
import pickle
import random
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app)


class REVAChatbot:
    """REVA University Chatbot class for NLP processing."""

    def __init__(self):
        """Initialize the chatbot with trained model."""
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.model = None
        self.label_encoder = None
        self.intents = None
        self.load_model()

    def load_model(self):
        """Load the trained model and vocabulary."""
        try:
            # Load words vocabulary
            with open('words.pkl', 'rb') as f:
                self.words = pickle.load(f)

            # Load classes
            with open('classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)

            # Load model and label encoder
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']

            # Load intents for responses
            with open('intents.json', 'r', encoding='utf-8') as f:
                self.intents = json.load(f)

            print("Chatbot model loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error: Model files not found. {e}")
            print("Please run train.py first to train the model.")
            raise

    def clean_up_sentence(self, sentence):
        """
        Tokenize and lemmatize user input sentence.

        Args:
            sentence: User input string

        Returns:
            List of lemmatized words
        """
        sentence_words = nltk.word_tokenize(sentence)
        return [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    def bag_of_words(self, sentence):
        """
        Create bag of words representation for input.

        Args:
            sentence: User input string

        Returns:
            numpy array representing bag of words
        """
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)

        for word in sentence_words:
            for i, w in enumerate(self.words):
                if w == word:
                    bag[i] = 1

        return np.array(bag)

    def predict_class(self, sentence):
        """
        Predict the intent class for user input.

        Args:
            sentence: User input string

        Returns:
            List of predictions with intent tag and probability
        """
        # Create bag of words
        bow = self.bag_of_words(sentence)

        # Get prediction probabilities
        if bow.sum() == 0:
            # No words matched, return default low confidence
            return [{'intent': 'unknown', 'probability': 0.0}]

        # Predict
        try:
            probabilities = self.model.predict_proba([bow])[0]
            predictions = []

            for i, probability in enumerate(probabilities):
                predictions.append({
                    'intent': self.classes[i],
                    'probability': float(probability)
                })

            # Sort by probability
            predictions.sort(key=lambda x: x['probability'], reverse=True)
            return predictions[:1]  # Return top prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return [{'intent': 'unknown', 'probability': 0.0}]

    def get_response(self, intents_list):
        """
        Get appropriate response based on predicted intent.

        Args:
            intents_list: List of predicted intents with probabilities

        Returns:
            Response string
        """
        if not intents_list or intents_list[0]['probability'] < 0.1:
            return self.get_fallback_response()

        tag = intents_list[0]['intent']

        # Find matching intent and return random response
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

        return self.get_fallback_response()

    def get_fallback_response(self):
        """Return fallback response for unknown queries."""
        fallback_responses = [
            "I apologize, but I'm not sure how to help with that. I can answer questions about REVA University academics, campus facilities, admissions, and student life.",
            "I'm not certain about that. Feel free to ask me about REVA University courses, exams, facilities, or admissions!",
            "I didn't quite understand that. Try asking about REVA University academics, campus, events, or contact information.",
            "I'm still learning! Please ask questions about REVA University, and I'll do my best to help."
        ]
        return random.choice(fallback_responses)

    def chat(self, message):
        """
        Process user message and return response.

        Args:
            message: User input string

        Returns:
            Chatbot response string
        """
        intents = self.predict_class(message)
        print(f"DEBUG: Message='{message}', Predicted={intents}")  # Debug log
        response = self.get_response(intents)
        return response


# Initialize chatbot instance
chatbot = REVAChatbot()


@app.route('/')
def home():
    """Render the home page with chatbot interface."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint for chatbot interaction.

    Expected JSON input:
        {"message": "user message here"}

    Returns:
        JSON response with bot reply
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'Please provide a message.'
            }), 400

        # Get chatbot response
        bot_response = chatbot.chat(user_message)

        return jsonify({
            'status': 'success',
            'response': bot_response
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'chatbot': 'REVA University Assistant',
        'version': '1.0.0'
    })

@app.route('/logo')
def serve_logo():
    """Serve the logo image."""
    from flask import send_from_directory, current_app
    import os
    logo_path = os.path.join(os.path.dirname(current_app.root_path), 'images.png')
    return send_from_directory(os.path.dirname(logo_path), 'images.png')


def main():
    """Main function to run the Flask application."""
    print("\n" + "=" * 50)
    print("REVA University Chatbot Server")
    print("=" * 50)
    print("\nServer starting at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server\n")

    app.run(debug=True, host='0.0.0.0', port=5001)


if __name__ == '__main__':
    main()
