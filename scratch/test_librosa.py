import librosa
import numpy as np
import sys

try:
    print(f"Librosa version: {librosa.__version__}")
    print("Numpy version:", np.__version__)
    # Try to create a dummy signal and extract MFCC
    sr = 22050
    y = np.random.randn(sr * 3) # 3 seconds of noise
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    print("MFCC shape:", mfcc.shape)
    features = np.mean(mfcc.T, axis=0)
    print("Features shape:", features.shape)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
