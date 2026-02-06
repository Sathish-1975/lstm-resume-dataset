from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Re-download stopwords if needed
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Configuration
MODEL_PATH = 'resume_lstm_model.h5'
TOKENizer_PATH = 'tokenizer_lstm.pkl'
ENCODER_PATH = 'label_encoder_lstm.pkl'
MAX_LEN = 500

# Global variables to store loaded objects
model = None
tokenizer = None
label_encoder = None

def load_objects():
    global model, tokenizer, label_encoder
    print("Loading model and preprocessing objects...")
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENizer_PATH) and os.path.exists(ENCODER_PATH):
        model = load_model(MODEL_PATH)
        with open(TOKENizer_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Model and objects loaded successfully.")
    else:
        print("Error: Missing model or preprocessing files!")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500
    
    data = request.get_json()
    resume_text = data.get('resume', '')
    
    if not resume_text:
        return jsonify({'error': 'No resume text provided.'}), 400
    
    # Preprocess
    cleaned_input = clean_text(resume_text)
    seq = tokenizer.texts_to_sequences([cleaned_input])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    
    # Predict
    prediction = model.predict(padded)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    category = label_encoder.classes_[class_idx]
    
    return jsonify({
        'category': category,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    load_objects()
    app.run(debug=True, port=5000)
