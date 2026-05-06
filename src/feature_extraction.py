import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract MFCC features from an audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        np.array: Mean of MFCC features
    """
    # Load audio file, standardizing to a duration of 3 seconds
    audio, sr = librosa.load(file_path, duration=3)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    #(40,T)
    
    # Return the mean of features over time
    return np.mean(mfcc.T, axis=0)