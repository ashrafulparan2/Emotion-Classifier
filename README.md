# Bangla Emotion and Intensity Detection App

A Streamlit web application for detecting emotions and their intensity in Bangla text using a fine-tuned BERT model.

## Model

This app uses the `ashrafulparan/Emotion-BERT` model from Hugging Face, which is a fine-tuned BERT model for Bangla emotion classification.

### Supported Emotions
- **Angry** 😠
- **Disgust** 🤢  
- **Fear** 😨
- **Happy** 😊
- **Sad** 😢
- **Surprise** 😮

### Intensity Levels
- **Low** (0.0) 🔻
- **Medium** (1.0) 🔸
- **High** (2.0) 🔺

## Features

- 🎭 **Single Text Analysis**: Analyze emotion and intensity for individual texts
- 📊 **Batch Analysis**: Process multiple texts at once
- 📈 **Probability Visualization**: Interactive charts showing confidence scores
- 💾 **Export Results**: Download analysis results as CSV
- 🔍 **Detailed Analysis**: View all emotion and intensity probabilities
- 🎯 **Quick Examples**: Pre-loaded sample texts for testing

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app_improved.py
   ```

   Or on Windows, simply double-click `run_app.bat`

## Usage

1. **Open the app** in your browser (usually at `http://localhost:8501`)

2. **Enter Bangla text** in the text area

3. **Click "Analyze Emotion"** to get predictions

4. **View results**:
   - Predicted emotion and intensity with confidence scores
   - Interactive probability charts
   - Detailed breakdown of all probabilities

5. **Try batch analysis** for multiple texts at once

## Model Architecture

The app follows the exact inference logic from the training notebook:

- **Base Model**: Bangla BERT (sagorsarker/bangla-bert-base)
- **Architecture**: Multi-task classification with two heads:
  - Emotion classifier (6 emotions)
  - Intensity classifier (3 levels)
- **Input**: Text sequences up to 128 tokens
- **Output**: Emotion + Intensity predictions with confidence scores

## File Structure

```
├── app_improved.py           # Main Streamlit application
├── app.py                   # Original Streamlit app
├── inference.py             # Simple command-line inference script
├── inference.ipynb          # Jupyter notebook with inference logic
├── labels_mapping.json      # Label mappings for emotions and intensities
├── requirements.txt         # Python dependencies
├── run_app.bat             # Windows batch script to run the app
└── README.md               # This file
```

## Technical Details

- **Framework**: Streamlit for web interface
- **Model Loading**: Hugging Face Transformers
- **Inference**: PyTorch with CUDA support (if available)
- **Visualization**: Plotly for interactive charts
- **Reproducibility**: Fixed random seeds for consistent results

## Example Texts

Try these sample Bangla texts:

- `আমি খুব খুশি আজকে।` (I am very happy today)
- `এটা খুব দুঃখজনক খবর।` (This is very sad news)  
- `আমি রাগে ফেটে পড়েছি।` (I am bursting with anger)
- `এই দৃশ্যটা দেখে আমি অবাক হয়ে গেছি।` (I was surprised to see this scene)

## Troubleshooting

### Model Loading Issues
- Ensure you have a stable internet connection
- Check if the Hugging Face model `ashrafulparan/Emotion-BERT` is accessible
- If using locally, ensure `labels_mapping.json` is in the same directory

### Performance
- The app works on both CPU and GPU
- GPU will provide faster inference for batch processing
- First-time model loading may take a few minutes

### Dependencies
- If you encounter package conflicts, try creating a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

## License

This project is for educational and research purposes. Please respect the license terms of the underlying BERT model and datasets.

## Acknowledgments

- Hugging Face for the Transformers library
- Streamlit for the web framework
- The creators of the Bangla BERT model
- The dataset contributors for Bangla emotion classification
