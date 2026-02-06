import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Remove HTML tags if any (the dataset has Resume_html)
    # But we use Resume_str which is mostly text
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
    df = pd.read_csv('Resume/Resume.csv')
    
    # We use Resume_str as it's the raw text
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
    max_len = 300
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
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train Model
    print("Training model...")
    epochs = 20
    batch_size = 64
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print(f"Accuracy: {accuracy_score(y_test_classes, y_pred_classes)}")
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))
    
    # Save model and preprocessing objects
    model.save('resume_rnn_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Model and preprocessing objects saved (resume_rnn_model.h5, tokenizer.pkl, label_encoder.pkl)")

if __name__ == "__main__":
    main()
