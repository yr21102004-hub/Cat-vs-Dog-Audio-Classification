import sys
import os
import joblib
import warnings
from feature_extraction import extract_features

# Suppress librosa warnings for clean output
warnings.filterwarnings('ignore')

def predict(file_path):
    model_path = os.path.join("..", "model", "trained_model.pkl")
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please run train.py first.")
        return

    try:
        # Extract features
        features = extract_features(file_path)
        
        # Predict
        prediction = model.predict(features.reshape(1, -1))
        
        # Map label
        classes = {0: 'Cat 🐱', 1: 'Dog 🐶'}
        print(f"\n🎧 Audio: {file_path}")
        print(f"🧠 Prediction: {classes[prediction[0]]}\n")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file.wav>")
    else:
        predict(sys.argv[1])
