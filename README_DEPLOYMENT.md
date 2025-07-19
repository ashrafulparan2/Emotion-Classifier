# Bangla Emotion Detection - Cloud Deployment

This repository contains a Streamlit application for detecting emotions and their intensity in Bangla text using a fine-tuned BERT model.

## 🚀 Live Demo

Deploy this app on various cloud platforms:

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Deploy using `app_cloud.py`
5. The app will automatically install dependencies from `requirements_cloud.txt`

### Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Choose "Streamlit" as the SDK
3. Upload the files:
   - `app_cloud.py` (rename to `app.py`)
   - `requirements_cloud.txt` (rename to `requirements.txt`)
4. The space will automatically build and deploy

### Railway
1. Create account on [Railway](https://railway.app/)
2. Deploy from GitHub repository
3. Set start command: `streamlit run app_cloud.py --server.port $PORT`

### Render
1. Create account on [Render](https://render.com/)
2. Connect GitHub repository
3. Set build command: `pip install -r requirements_cloud.txt`
4. Set start command: `streamlit run app_cloud.py --server.port $PORT --server.address 0.0.0.0`

## 📁 Files

- `app_cloud.py` - Main Streamlit application (cloud-optimized)
- `requirements_cloud.txt` - Python dependencies for cloud deployment
- `inference.ipynb` - Original inference notebook
- `README_DEPLOYMENT.md` - This deployment guide

## 🎯 Features

- **Real-time Emotion Detection**: Analyze Bangla text for emotions (angry, disgust, fear, happy, sad, surprise)
- **Intensity Classification**: Determine emotion intensity (Low, Medium, High)
- **Interactive UI**: Beautiful and responsive interface
- **Batch Processing**: Analyze multiple texts at once
- **Probability Visualization**: Interactive charts showing prediction confidence
- **Export Results**: Download analysis results as CSV

## 🔧 Model Information

- **Model**: `ashrafulparan/Emotion-BERT`
- **Base Model**: Bangla BERT
- **Tasks**: Multi-task classification (emotion + intensity)
- **Architecture**: Custom BERT with dual classification heads

## 💻 Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements_cloud.txt

# Run the app
streamlit run app_cloud.py
```

## 🌐 Environment Variables

For production deployment, you may want to set:

```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 📊 Usage

1. Enter Bangla text in the input area
2. Click "Analyze Emotion" 
3. View predicted emotion and intensity
4. Explore probability distributions
5. Use batch analysis for multiple texts
6. Download results if needed

## 🤝 Contributing

Feel free to contribute by:
- Improving the UI/UX
- Adding more visualization options
- Optimizing performance
- Adding more language support

## 📄 License

This project is open source. Please check the model license at [ashrafulparan/Emotion-BERT](https://huggingface.co/ashrafulparan/Emotion-BERT) for usage terms.

## 🙏 Acknowledgments

- Model by ashrafulparan
- Built with Streamlit and Hugging Face Transformers
- Bangla NLP community for datasets and research
