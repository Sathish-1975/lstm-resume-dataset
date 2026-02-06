import numpy as np
import re
import nltk
import pickle
import sys
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Re-download stopwords if needed
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def predict_category(resume_text):
    # Load preprocessing objects
    try:
        with open('tokenizer_lstm.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder_lstm.pkl', 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        return "Error: Tokenizer or Label Encoder file not found. Please run training first.", 0

    # Load Model
    try:
        model = load_model('resume_lstm_model.h5')
    except Exception as e:
        return f"Error loading model: {e}", 0
    
    # Preprocess Input
    max_len = 500
    cleaned_input = clean_text(resume_text)
    seq = tokenizer.texts_to_sequences([cleaned_input])
    padded = pad_sequences(seq, maxlen=max_len)
    
    # Predict
    prediction = model.predict(padded)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    
    return le.classes_[class_idx], confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_resume = " ".join(sys.argv[1:])
    else:
        sample_resume = """
        Experienced Data Scientist with expertise in Python, SQL, and Machine Learning. 
        Deep understanding of neural networks, NLP, and data visualization. 
        Previously worked at a tech startup building recommendation engines.
        """
    
    print("Predicting...")
    category, conf = predict_category(sample_resume)
    print(f"\nPredicted Category: {category}")
    print(f"Confidence: {conf:.4f}")
