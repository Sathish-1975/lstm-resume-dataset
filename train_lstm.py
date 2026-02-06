import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('Resume/Resume.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Subsample (commented out for full training)
    # df = df.sample(n=min(len(df), 1000), random_state=42) 
    
    print(f"Dataset size: {len(df)} rows")
    print("Preprocessing text...")
    df['cleaned_text'] = df['Resume_str'].apply(clean_text)
    
    # Label encoding
    print("Encoding labels...")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])
    num_classes = len(le.classes_)
    
    # Tokenization
    print("Tokenizing...")
    max_words = 10000
    max_len = 500
    tokenizer = Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(df['cleaned_text'].values)
    
    X = tokenizer.texts_to_sequences(df['cleaned_text'].values)
    X = pad_sequences(X, maxlen=max_len)
    
    y = to_categorical(df['label'], num_classes=num_classes)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build Model
    print("Building model...")
    embedding_dim = 128
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # Train Model
    print("Training model...")
    # Increase epochs for better results
    epochs = 10 
    batch_size = 64
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print(f"Accuracy: {accuracy_score(y_test_classes, y_pred_classes):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))
    
    # Save model and preprocessing objects
    print("Saving model and objects...")
    model.save('resume_lstm_model.h5')
    with open('tokenizer_lstm.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('label_encoder_lstm.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Finished! Saved as resume_lstm_model.h5, tokenizer_lstm.pkl, label_encoder_lstm.pkl")

if __name__ == "__main__":
    main()
