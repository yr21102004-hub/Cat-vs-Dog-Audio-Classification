# 🐱🐶 Cat vs Dog Audio Classification

A machine learning project to classify audio recordings of cats and dogs using MFCC feature extraction and a Random Forest Classifier. The project includes a training pipeline and a web-based interface built with Streamlit.

## 🚀 Overview

This project provides a complete end-to-end pipeline for audio classification:
1.  **Feature Extraction**: Extracting MFCCs (Mel-frequency cepstral coefficients) from `.wav` files using `librosa`.
2.  **Training**: Training a Random Forest model on the extracted features.
3.  **Deployment**: A user-friendly Streamlit app to upload audio and get instant predictions.

## 📂 Project Structure

```text
Cat_Dog_Audio_Classification/
├── app.py                # Streamlit Web Application
├── src/
│   ├── feature_extraction.py  # Audio processing logic
│   ├── train.py              # Model training script
│   └── predict.py            # Local inference script
├── model/
│   └── trained_model.pkl      # Saved Random Forest model
├── dataset/                   # (Ignored) Training data directory
└── .gitignore                 # Files to ignore in Git
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd Cat_Dog_Audio_Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install streamlit librosa scikit-learn joblib numpy
   ```

## 💻 Usage

### 1. Training the Model
If you want to retrain the model with your own data, place your `.wav` files in `dataset/cat` and `dataset/dog` folders, then run:
```bash
python src/train.py
```

### 2. Running the Web App
To start the Streamlit interface:
```bash
streamlit run app.py
```

### 3. Local Prediction
To test a single file via terminal:
```bash
python src/predict.py
```

## 🧠 Technical Details
- **Features**: 40 MFCC coefficients averaged over time.
- **Model**: Random Forest Classifier with 100 estimators.
- **Preprocessing**: Audio files are standardized to a 3-second duration during feature extraction.

## 📝 License
This project is for educational purposes. Feel free to use and modify!
