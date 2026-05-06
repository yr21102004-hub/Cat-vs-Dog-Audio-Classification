import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from feature_extraction import extract_features

def load_data(dataset_path):
    """
    Load audio files from dataset path, extract features and return (X, y).
    """
    X = []
    y = []
    classes = {'cat': 0, 'dog': 1}
    
    for cls, label in classes.items():
        cls_dir = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory {cls_dir} does not exist.")
            continue
            
        for file in os.listdir(cls_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(cls_dir, file)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    dataset_path = os.path.join("..", "dataset")
    print("Loading and processing data...")
    X, y = load_data(dataset_path)
    
    if len(X) == 0:
        print("No data found! Please add .wav files to dataset/cat and dataset/dog directories.")
    else:
        print(f"Data loaded: {len(X)} samples.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        print("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Save model
        model_dir = os.path.join("..", "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "trained_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
