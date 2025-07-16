"""
Streamlit App for Bangla Emotion and Intensity Detection
Uses the model from Hugging Face: ashrafulparan/Emotion-BERT
"""

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel, AutoConfig
from torch.nn.functional import softmax
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class BertForMultiTaskClassification(BertPreTrainedModel):
    """Custom BERT model for multi-task emotion and intensity classification"""
    
    def __init__(self, config, num_emotions, num_intensities):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_emotion = nn.Linear(config.hidden_size, num_emotions)
        self.classifier_intensity = nn.Linear(config.hidden_size, num_intensities)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits_emotion = self.classifier_emotion(pooled_output)
        logits_intensity = self.classifier_intensity(pooled_output)

        return (logits_emotion, logits_intensity)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model, tokenizer, and label mappings"""
    
    # Try to load from Hugging Face first
    model_name = "ashrafulparan/Emotion-BERT"
    
    try:
        # Try loading from Hugging Face
        from huggingface_hub import hf_hub_download
        import json
        
        # Download label mappings
        labels_file = hf_hub_download(repo_id=model_name, filename="labels_mapping.json")
        with open(labels_file, 'r', encoding='utf-8') as f:
            label_mappings = json.load(f)
        
        emotions = label_mappings["emotions"]
        intensities = label_mappings["intensities"]
        id_to_emotion = {int(k): v for k, v in label_mappings["id_to_emotion"].items()}
        id_to_intensity = {int(k): v for k, v in label_mappings["id_to_intensity"].items()}
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load config
        config = AutoConfig.from_pretrained(model_name)
        
        # Create model with correct dimensions
        model = BertForMultiTaskClassification(
            config, 
            num_emotions=len(emotions),
            num_intensities=len(intensities)
        )
        
        # Download and load model weights
        model_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, tokenizer, id_to_emotion, id_to_intensity
        
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        
        # Fallback to local model if available
        return load_local_model()

def load_local_model():
    """Fallback function to load model locally"""
    model_dir = "saved_bangla_emotion_model"
    
    try:
        import json
        import os
        
        # Load label mappings
        labels_path = os.path.join(model_dir, "labels_mapping.json")
        with open(labels_path, 'r', encoding='utf-8') as f:
            label_mappings = json.load(f)
        
        emotions = label_mappings["emotions"]
        intensities = label_mappings["intensities"]
        id_to_emotion = {int(k): v for k, v in label_mappings["id_to_emotion"].items()}
        id_to_intensity = {int(k): v for k, v in label_mappings["id_to_intensity"].items()}
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        config = AutoConfig.from_pretrained(model_dir)
        
        model = BertForMultiTaskClassification(
            config, 
            num_emotions=len(emotions),
            num_intensities=len(intensities)
        )
        
        # Load state dict
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
        
        model.eval()
        
        return model, tokenizer, id_to_emotion, id_to_intensity
        
    except Exception as e:
        st.error(f"Error loading local model: {e}")
        st.error("Please make sure either:")
        st.error("1. The model 'ashrafulparan/Emotion-BERT' is available on Hugging Face, OR")
        st.error("2. The 'saved_bangla_emotion_model' directory exists locally with the trained model")
        return None, None, None, None

def predict_emotion_intensity(text, model, tokenizer, id_to_emotion, id_to_intensity):
    """Predict emotion and intensity for a single text"""
    
    if model is None:
        return None, None, None, None, None, None
    
    device = next(model.parameters()).device
    
    # Tokenize input
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Move to device
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # Make prediction
    with torch.no_grad():
        logits_emotion, logits_intensity = model(**encoding)
        
        # Get probabilities
        probs_emotion = softmax(logits_emotion, dim=-1)
        probs_intensity = softmax(logits_intensity, dim=-1)
        
        # Get predictions
        pred_emotion_id = torch.argmax(logits_emotion, dim=-1).item()
        pred_intensity_id = torch.argmax(logits_intensity, dim=-1).item()
        
        # Map back to labels
        pred_emotion = id_to_emotion[pred_emotion_id]
        pred_intensity = id_to_intensity[pred_intensity_id]
        
        # Get confidence scores
        emotion_confidence = probs_emotion[0][pred_emotion_id].item()
        intensity_confidence = probs_intensity[0][pred_intensity_id].item()
        
        return (pred_emotion, pred_intensity, emotion_confidence, intensity_confidence, 
                probs_emotion[0].cpu().numpy(), probs_intensity[0].cpu().numpy())

def create_emotion_chart(probabilities, labels):
    """Create a bar chart for emotion probabilities"""
    df = pd.DataFrame({
        'Emotion': labels,
        'Probability': probabilities
    })
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(df, x='Probability', y='Emotion', orientation='h',
                 title='Emotion Probabilities',
                 color='Probability',
                 color_continuous_scale='viridis')
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_intensity_chart(probabilities, labels):
    """Create a bar chart for intensity probabilities"""
    df = pd.DataFrame({
        'Intensity': labels,
        'Probability': probabilities
    })
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(df, x='Probability', y='Intensity', orientation='h',
                 title='Intensity Probabilities',
                 color='Probability',
                 color_continuous_scale='plasma')
    fig.update_layout(height=300, showlegend=False)
    return fig

def main():
    """Main Streamlit app"""
    
    st.set_page_config(
        page_title="Bangla Emotion Detection",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("🇧🇩 Bangla Emotion and Intensity Detection")
    st.markdown("### Analyze emotions and their intensity in Bangla text using BERT")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("This app uses a fine-tuned BERT model to detect emotions and their intensity in Bangla text.")
        st.write("**Model:** ashrafulparan/Emotion-BERT")
        st.write("**Emotions:** Angry, Fear, Happy, Love, Sad, Surprise")
        st.write("**Intensity:** Low, Medium, High")
        
        st.header("How to use")
        st.write("1. Enter your Bangla text in the text area")
        st.write("2. Click 'Analyze Emotion' button")
        st.write("3. View the predicted emotion and intensity")
        st.write("4. Explore the probability distributions")
    
    # Load model
    with st.spinner("Loading model from Hugging Face..."):
        model, tokenizer, id_to_emotion, id_to_intensity = load_model_and_tokenizer()
    
    if model is None:
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Input section
    st.header("📝 Input Text")
    
    # Default text
    default_text = "আমি আজ খুব খুশি এবং উৎসাহিত বোধ করছি"
    
    # Text input
    text_input = st.text_area(
        "Enter Bangla text to analyze:",
        value=default_text,
        height=100,
        placeholder="Type your Bangla text here..."
    )
    
    # Example texts
    st.write("**Example texts:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Happy Example"):
            text_input = "আমি আজ খুব খুশি এবং উৎসাহিত বোধ করছি"
            st.rerun()
    
    with col2:
        if st.button("Sad Example"):
            text_input = "আমি খুব দুঃখিত এবং হতাশ বোধ করছি"
            st.rerun()
    
    with col3:
        if st.button("Angry Example"):
            text_input = "আমি খুব রাগান্বিত এবং বিরক্ত"
            st.rerun()
    
    # Analysis button
    if st.button("🔍 Analyze Emotion", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing emotion..."):
                # Make prediction
                result = predict_emotion_intensity(
                    text_input, model, tokenizer, id_to_emotion, id_to_intensity
                )
                
                if result[0] is not None:
                    emotion, intensity, emotion_conf, intensity_conf, emotion_probs, intensity_probs = result
                    
                    # Results section
                    st.header("📊 Analysis Results")
                    
                    # Main results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="🎭 Predicted Emotion",
                            value=emotion.title(),
                            delta=f"Confidence: {emotion_conf:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            label="📈 Predicted Intensity",
                            value=intensity.title(),
                            delta=f"Confidence: {intensity_conf:.1%}"
                        )
                    
                    # Visualizations
                    st.header("📈 Probability Distributions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Emotion probabilities
                        emotion_labels = list(id_to_emotion.values())
                        fig_emotion = create_emotion_chart(emotion_probs, emotion_labels)
                        st.plotly_chart(fig_emotion, use_container_width=True)
                    
                    with col2:
                        # Intensity probabilities
                        intensity_labels = list(id_to_intensity.values())
                        fig_intensity = create_intensity_chart(intensity_probs, intensity_labels)
                        st.plotly_chart(fig_intensity, use_container_width=True)
                    
                    # Detailed results
                    with st.expander("🔍 Detailed Results"):
                        st.write("**Emotion Probabilities:**")
                        for i, (emotion_name, prob) in enumerate(zip(emotion_labels, emotion_probs)):
                            st.write(f"- {emotion_name.title()}: {prob:.3f} ({prob*100:.1f}%)")
                        
                        st.write("**Intensity Probabilities:**")
                        for i, (intensity_name, prob) in enumerate(zip(intensity_labels, intensity_probs)):
                            st.write(f"- {intensity_name.title()}: {prob:.3f} ({prob*100:.1f}%)")
                else:
                    st.error("Failed to analyze the text. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and Hugging Face Transformers")

if __name__ == "__main__":
    main()
