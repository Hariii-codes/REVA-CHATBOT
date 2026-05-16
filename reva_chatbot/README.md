# REVA University Chatbot

A intelligent NLP-powered chatbot assistant for REVA University students, built with Python, NLTK, Flask, and modern web technologies.

![REVA Chatbot](https://img.shields.io/badge/REVA-Chatbot-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-red)

## Features

- **🎓 Smart Intent Classification**: Uses NLTK and machine learning for accurate question understanding
- **💬 Natural Conversations**: Handles queries about academics, campus facilities, admissions, and student life
- **🎨 Modern UI**: Clean, responsive chatbot interface with smooth animations
- **⚡ Real-time Responses**: Fast API powered by Flask backend
- **📱 Mobile Friendly**: Fully responsive design that works on all devices
- **🎯 Quick Suggestions**: Pre-built question buttons for common queries
- **🎤 Voice Input**: Optional speech recognition support (in compatible browsers)

## Chatbot Capabilities

The REVA Assistant can help with:

### 📚 Academics
- Course information and syllabus details
- Exam schedules and internal assessments
- Attendance requirements and policies
- Assignment details and mark distribution

### 🏛️ Campus Information
- Department locations and HOD information
- Library timings and digital resources
- Campus facilities and amenities

### 🏠 Student Life
- Hostel availability, rules, and fees
- Student clubs and organizations
- Hackathons and tech events
- Cultural festivals and activities

### 📝 Admissions
- Admission process and eligibility
- Required documents
- Contact information
- Office locations

## Project Structure

```
reva_chatbot/
│
├── app.py              # Flask backend server
├── train.py            # NLP model training script
├── intents.json        # Training dataset with intents
├── words.pkl           # Vocabulary file (generated after training)
├── classes.pkl         # Intent classes (generated after training)
├── model.pkl           # Trained model (generated after training)
│
├── templates/
│ └── index.html        # Frontend HTML template
│
├── static/
│ ├── style.css         # Styling
│ └── script.js         # Frontend JavaScript
│
└── README.md           # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install flask flask-cors nltk scikit-learn numpy
```

Or use the requirements.txt file:

```bash
pip install -r requirements.txt
```

Create `requirements.txt` with:
```
flask>=2.0.0
flask-cors>=3.0.0
nltk>=3.8.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

### Step 2: Train the Model

Run the training script to generate the NLP model:

```bash
python train.py
```

This will:
- Load intents from `intents.json`
- Tokenize and preprocess the text data
- Train a Logistic Regression classifier
- Save model files (`words.pkl`, `classes.pkl`, `model.pkl`)

**Expected Output:**
```
==================================================
REVA University Chatbot - Training
==================================================
Loading intents...
Preprocessing training data...
Vocabulary size: XXX
Number of intents: XX
Training documents: XXX
Training model...
Model training completed!
Training accuracy: X.XX
Model files saved successfully!
```

### Step 3: Run the Application

Start the Flask server:

```bash
python app.py
```

### Step 4: Access the Chatbot

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

The chatbot interface will load, and you can start asking questions!

## Usage Examples

### Greeting
```
You: Hello!
Bot: Hello! I am the REVA University Assistant. How can I help you today?
```

### Academics
```
You: What subjects are in semester 6?
Bot: Semester 6 subjects include Natural Language Processing, Computer Vision and Applications, Advanced IoT, AI with Data Cloud Master, Software Development through Experiential Learning, and Indian Knowledge System.
```

### Campus Facilities
```
You: What are the library timings?
Bot: The central library is open from 8 AM to 12 midnight on regular days and 9 AM to 11 PM on holidays.
```

### Admissions
```
You: What is the admission process?
Bot: Students must apply through the REVA admission portal, take REVA CET or other accepted entrance exams and attend counseling.
```

## API Documentation

### Chat Endpoint

**URL:** `/chat`

**Method:** `POST`

**Request Body:**
```json
{
  "message": "What is REVA University known for?"
}
```

**Response:**
```json
{
  "status": "success",
  "response": "REVA University is known for its modern infrastructure, AI-driven education, strong engineering and management programs, and excellent placement opportunities."
}
```

### Health Check Endpoint

**URL:** `/health`

**Method:** `GET`

**Response:**
```json
{
  "status": "healthy",
  "chatbot": "REVA University Assistant",
  "version": "1.0.0"
}
```

## Customization

### Adding New Intents

To add new questions and responses, edit `intents.json`:

```json
{
  "intents": [
    {
      "tag": "new_intent_tag",
      "patterns": ["Question pattern 1", "Question pattern 2"],
      "responses": ["Response 1", "Response 2"]
    }
  ]
}
```

After modifying `intents.json`, retrain the model:

```bash
python train.py
```

### Styling Customization

Edit `static/style.css` to customize colors and appearance:

```css
:root {
    --primary-color: #0066cc;    /* Main brand color */
    --primary-dark: #0052a3;     /* Darker shade */
    --secondary-color: #00d4aa;  /* Accent color */
}
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| NLP | NLTK, Scikit-learn |
| Frontend | HTML5, CSS3, JavaScript |
| Styling | Custom CSS with CSS Variables |
| Icons | Font Awesome |

## Model Architecture

The chatbot uses a simple but effective NLP pipeline:

1. **Tokenization**: NLTK word tokenization
2. **Lemmatization**: Word reduction to base forms
3. **Bag of Words**: Feature extraction
4. **Classification**: Logistic Regression with multinomial solver
5. **Response Selection**: Random response from matched intent

## Troubleshooting

### Model Files Not Found

**Error:** `FileNotFoundError: Model files not found`

**Solution:** Run `python train.py` first to generate the model files.

### NLTK Data Missing

**Error:** `LookupError: Resource punkt not found`

**Solution:** The training script automatically downloads required NLTK data. If issues persist:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Port Already in Use

**Error:** `Address already in use`

**Solution:** Change the port in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## Future Enhancements

Potential improvements for the chatbot:

- [ ] Integration with REVA University database
- [ ] User authentication and personalized responses
- [ ] Multi-language support
- [ ] Context-aware conversations
- [ ] Integration with university APIs
- [ ] Student portal integration
- [ ] Advanced NLP with transformers
- [ ] Admin dashboard for analytics

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is created for educational purposes for REVA University.

## Credits

- **Developed for:** REVA University NLP Course Project
- **Technologies:** Python, NLTK, Flask, HTML, CSS, JavaScript

## Support

For issues or questions, please contact the development team or raise an issue in the repository.

---

**Built with ❤️ for REVA University Students**
