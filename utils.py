import numpy as np
from gensim.models import Word2Vec
import librosa

def get_word2vec_embeddings(lyrics, word2vec_model, embedding_dim=100):
    """
    Converts lyrics into a Word2Vec embedding by averaging the word vectors.
    
    Args:
        lyrics (str): Input song lyrics.
        word2vec_model (gensim.models.Word2Vec): Pre-trained Word2Vec model.
        embedding_dim (int): The dimension of the word vectors (depends on your Word2Vec model).
        
    Returns:
        np.array: Averaged Word2Vec embedding for the lyrics.
    """
    # Tokenize the lyrics (simple split by spaces, but can be improved with better tokenization)
    words = lyrics.split()
    
    # Initialize an array for the embeddings
    embedding_matrix = np.zeros((len(words), embedding_dim))
    
    # Create the embedding matrix by averaging the embeddings of the words present in Word2Vec model
    valid_words = 0
    for idx, word in enumerate(words):
        if word in word2vec_model.wv:
            embedding_matrix[idx] = word2vec_model.wv[word]
            valid_words += 1
    
    # If no valid words are found, return an array of zeros
    if valid_words == 0:
        return np.zeros((embedding_dim,))
    
    # Average the embeddings
    return np.mean(embedding_matrix[:valid_words], axis=0)

def extract_audio_features(file_path):
    """
    Extracts MFCC, chroma, and mel spectrogram features from an audio file and concatenates them.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        np.array: Concatenated audio features (MFCCs, chroma, mel spectrogram).
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCCs, chroma, and mel spectrogram features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Compute the mean of each feature across time
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel_mean = np.mean(mel.T, axis=0)
        
        # Concatenate the features into a single array
        return np.concatenate([mfccs_mean, chroma_mean, mel_mean])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None  # Return None if an error occurs during feature extraction

def reshape_audio_features(features):
    """
    Reshapes the audio features to match the input shape required by the RCNN model.
    
    Args:
        features (np.array): Extracted audio features.
        
    Returns:
        np.array: Reshaped audio features.
    """
    try:
        # Define timesteps and feature size based on the model's expected input
        timesteps = 1  # Adjust as per the model's structure
        n_features = features.shape[0]  # This is the number of features extracted
        
        # Reshape the features to (1, timesteps, n_features)
        return features.reshape(1, timesteps, n_features)
    except Exception as e:
        print(f"Error reshaping features: {e}")
        return None
