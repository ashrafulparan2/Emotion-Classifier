# Bangla Emotion Detection Streamlit App

A web application for detecting emotions and their intensity in Bangla text using a fine-tuned BERT model.

## Features

- 🎭 **Emotion Detection**: Identifies emotions like angry, fear, happy, love, sad, surprise
- 📈 **Intensity Analysis**: Determines the intensity level (low, medium, high)
- 📊 **Interactive Visualizations**: Bar charts showing probability distributions
- 🇧🇩 **Bangla Text Support**: Optimized for Bengali/Bangla language
- 🤗 **Hugging Face Integration**: Uses the model `ashrafulparan/Emotion-BERT`

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or on Windows, you can run:
   ```bash
   setup.bat
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

## Usage

1. Enter your Bangla text in the text area
2. Click the "Analyze Emotion" button
3. View the predicted emotion and intensity with confidence scores
4. Explore the probability distributions in the interactive charts
5. Check detailed results in the expandable section

## Example Texts

The app includes several example texts you can try:
- **Happy**: "আমি আজ খুব খুশি এবং উৎসাহিত বোধ করছি"
- **Sad**: "আমি খুব দুঃখিত এবং হতাশ বোধ করছি"
- **Angry**: "আমি খুব রাগান্বিত এবং বিরক্ত"

## Model Information

- **Model**: `ashrafulparan/Emotion-BERT`
- **Base Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Task**: Multi-task classification (Emotion + Intensity)
- **Language**: Bangla/Bengali

## Supported Emotions

1. **Angry** - রাগ
2. **Fear** - ভয়
3. **Happy** - খুশি
4. **Love** - ভালোবাসা
5. **Sad** - দুঃখ
6. **Surprise** - বিস্ময়

## Intensity Levels

1. **Low** - কম
2. **Medium** - মাঝারি
3. **High** - বেশি

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- Plotly
- Pandas
- NumPy

## File Structure

```
├── app.py              # Main Streamlit application
├── inference.py        # Original inference script
├── requirements.txt    # Python dependencies
├── setup.bat          # Windows setup script
└── README.md          # This file
```

## Troubleshooting

1. **Model loading issues**: Make sure you have a stable internet connection to download the model from Hugging Face
2. **CUDA errors**: The app will automatically use CPU if CUDA is not available
3. **Memory issues**: For large texts, the model uses a maximum length of 128 tokens

## Contributing

Feel free to contribute by:
- Adding more example texts
- Improving the UI/UX
- Adding new features
- Reporting bugs

## License

This project is open source. Please check the individual model license on Hugging Face.
