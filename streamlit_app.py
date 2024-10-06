import streamlit as st
import numpy as np
import tensorflow as tf
from utils import extract_audio_features, reshape_audio_features, get_word2vec_embeddings
from gensim.models import Word2Vec

# Load models
audio_model = tf.keras.models.load_model('audio_model.h5')
lyrics_model = tf.keras.models.load_model('lyrics_model.h5')

# Load Word2Vec model (for lyrics processing)
word2vec_model = Word2Vec.load('word2vec_model.bin')

# Streamlit app
st.title("Music Genre Classifier")

# Section 1: Audio-based Classification
st.header("Classify by Audio Features")

# File uploader for audio
uploaded_audio = st.file_uploader("Upload an audio file (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_audio is not None:
    # Save uploaded audio file temporarily
    audio_path = "temp_audio." + uploaded_audio.name.split('.')[-1]
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.getbuffer())

    # Extract and reshape audio features
    st.write("Extracting audio features...")
    audio_features = extract_audio_features(audio_path)
    reshaped_features = reshape_audio_features(audio_features)

    if reshaped_features is not None:
        # Predict genre using the audio model
        audio_probs = audio_model.predict(reshaped_features)
        st.write("Audio-based prediction:")
        st.bar_chart(audio_probs)
    else:
        st.write("Error: Unable to extract features from the audio.")

# Section 2: Lyrics-based Classification
st.header("Classify by Lyrics")

# Text input for lyrics
lyrics_input = st.text_area("Enter song lyrics:")

if lyrics_input:
    # Process and predict genre from lyrics
    st.write("Processing lyrics...")
    lyrics_embeddings = get_word2vec_embeddings(lyrics_input, word2vec_model)
    
    # Predict genre using the lyrics model
    lyrics_probs = lyrics_model.predict(np.expand_dims(lyrics_embeddings, axis=0))
    st.write("Lyrics-based prediction:")
    st.bar_chart(lyrics_probs)

# Clean up temporary audio file if created
import os
if uploaded_audio:
    os.remove(audio_path)
