import streamlit as st
import numpy as np
import tensorflow as tf
import os
import speech_recognition as sr
from pydub import AudioSegment
from utils import extract_audio_features, get_word2vec_embeddings
from gensim.models import Word2Vec
import librosa

# Load models
audio_model = tf.keras.models.load_model('audio_model.h5')
lyrics_model = tf.keras.models.load_model('lyrics_model.h5')

# Load Word2Vec model (for lyrics processing)
word2vec_model = Word2Vec.load('word2vec_model.bin')

# Streamlit app
st.title("Music Genre Classifier")

# File uploader for audio
uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if uploaded_file is not None:
    # Save uploaded MP3 file temporarily
    audio_path = "temp_audio.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert MP3 to WAV for speech recognition
    audio_segment = AudioSegment.from_mp3(audio_path)
    audio_segment.export("temp_audio.wav", format="wav")

    # Extract lyrics using speech recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = recognizer.record(source)
        try:
            lyrics = recognizer.recognize_google(audio_data)
            st.write("Extracted Lyrics:")
            st.write(lyrics)
        except sr.UnknownValueError:
            st.write("Could not understand audio.")
            lyrics = ""
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
            lyrics = ""

    if lyrics:
        # Process audio features
        st.write("Extracting audio features...")
        audio_features = extract_audio_features(audio_path)
        
        # Predict genre using the audio model
        audio_probs = audio_model.predict(np.expand_dims(audio_features, axis=0))
        st.write("Audio-based prediction:")
        st.bar_chart(audio_probs)

        # Process and predict genre from lyrics
        st.write("Processing lyrics...")
        lyrics_embeddings = get_word2vec_embeddings(lyrics, word2vec_model)
        
        # Predict genre using the lyrics model
        lyrics_probs = lyrics_model.predict(np.expand_dims(lyrics_embeddings, axis=0))
        st.write("Lyrics-based prediction:")
        st.bar_chart(lyrics_probs)

        # Combine predictions (3:1 ratio for audio:lyrics)
        combined_probs = 0.75 * audio_probs + 0.25 * lyrics_probs
        st.write("Combined prediction (75% audio, 25% lyrics):")
        st.bar_chart(combined_probs)

    # Clean up temporary files
    os.remove(audio_path)
    os.remove("temp_audio.wav")
