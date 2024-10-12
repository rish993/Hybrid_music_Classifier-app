import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import get_word2vec_embeddings, extract_audio_features, reshape_audio_features
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import speech_recognition as sr
from pydub import AudioSegment  # Import pydub for audio conversion
import os
from moviepy.editor import AudioFileClip

# Load your models
audio_model = load_model('audio_model.h5')
lyrics_model = load_model('lyrics_model.h5')

# Load Word2Vec model
word2vec_model = Word2Vec.load('word2vec_model.bin')

# Define genre labels for both models
audio_model_genres = ['blues' 'classical' 'country' 'disco' 'hiphop' 'jazz' 'metal' 'pop''reggae' 'rock']
lyrics_model_genres = ['blues', 'classical', 'country', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Speech Recognition function
def extract_lyrics(file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_path)
    with audio_file as source:
        audio = recognizer.record(source)
    
    try:
        lyrics = recognizer.recognize_google(audio)
        return lyrics
    except sr.UnknownValueError:
        st.error("Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError:
        st.error("Error connecting to the Speech Recognition API.")
        return None

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file_path):
    wav_file_path = mp3_file_path.replace('.mp3', '.wav')
    audio_clip = AudioFileClip(mp3_file_path)
    audio_clip.write_audiofile(wav_file_path, codec='pcm_s16le')  # Exporting as WAV
    audio_clip.close()
    return wav_file_path

# Align predictions
def align_predictions(audio_preds, lyrics_preds):
    aligned_lyrics_preds = np.zeros_like(lyrics_preds)
    genre_mapping = {genre: audio_model_genres.index(genre) for genre in lyrics_model_genres}
    for i, genre in enumerate(lyrics_model_genres):
        aligned_lyrics_preds[:, genre_mapping[genre]] = lyrics_preds[:, i]
    return aligned_lyrics_preds

# Plotting
def plot_genre_probabilities(probabilities, genres):
    fig, ax = plt.subplots()
    ax.barh(genres, probabilities)
    ax.set_xlabel('Probability')
    ax.set_title('Genre Prediction')
    st.pyplot(fig)

# Streamlit app UI
st.title("Hybrid Music Genre Classifier")
uploaded_file = st.file_uploader("Upload an mp3 file", type=["mp3"])

if uploaded_file is not None:
    with open("temp.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert MP3 to WAV
    wav_file_path = convert_mp3_to_wav("temp.mp3")
    
    # Extract audio features
    audio_features = extract_audio_features(wav_file_path)
    if audio_features is not None:
        reshaped_audio_features = reshape_audio_features(audio_features)
        audio_genre_prediction = audio_model.predict(reshaped_audio_features)
        
        # Speech-to-text for lyrics
        lyrics = extract_lyrics(wav_file_path)
        if lyrics:
            st.write(f"Recognized lyrics: {lyrics}")
            lyrics_embedding = get_word2vec_embeddings(lyrics, word2vec_model)
            lyrics_embedding = np.expand_dims(lyrics_embedding, axis=0)
            lyrics_genre_prediction = lyrics_model.predict(lyrics_embedding)
            
            # Align lyrics model predictions to audio model genre order
            aligned_lyrics_genre_prediction = align_predictions(audio_genre_prediction, lyrics_genre_prediction)
            
            # Average the predictions
            combined_genre_probs = (audio_genre_prediction + aligned_lyrics_genre_prediction) / 2
            
            # Plot the averaged probabilities
            plot_genre_probabilities(combined_genre_probs[0], audio_model_genres)
    
    # Clean up temporary files
    os.remove("temp.mp3")
    os.remove(wav_file_path)