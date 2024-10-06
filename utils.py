import librosa
import numpy as np
import gensim

# Audio Feature Extraction Function
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel_mean = np.mean(mel.T, axis=0)
        
        return np.concatenate([mfccs_mean, chroma_mean, mel_mean])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None 

# Word2Vec Embedding Function for Lyrics
def get_word2vec_embeddings(lyrics, word2vec_model):
    tokens = lyrics.split()
    embeddings = [word2vec_model[word] for word in tokens if word in word2vec_model]
    return np.mean(embeddings, axis=0)
