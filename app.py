"""
Enhanced Streamlit App for Bangla Emotion and Intensity Detection
Uses the model from Hugging Face: ashrafulparan/Emotion-BERT
Following the exact inference logic from the notebook
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel, AutoConfig
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.graph_objects as go
import json
import random

# Set page config
st.set_page_config(
    page_title="Bangla Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set seeds for reproducibility (same as notebook)
def set_seed(seed=42):
    """Set seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class BertForMultiTaskClassification(BertPreTrainedModel):
    """Custom BERT model for multi-task emotion and intensity classification (Same as notebook)"""
    
    def __init__(self, config, num_emotions=7, num_intensities=3):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_emotion = nn.Linear(config.hidden_size, num_emotions)
        self.classifier_intensity = nn.Linear(config.hidden_size, num_intensities)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits_emotion = self.classifier_emotion(pooled_output)
        logits_intensity = self.classifier_intensity(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            emotion_labels = labels[:, 0]
            intensity_labels = labels[:, 1]
            loss_emotion = loss_fct(logits_emotion, emotion_labels)
            loss_intensity = loss_fct(logits_intensity, intensity_labels)
            loss = loss_emotion + loss_intensity

        output = (logits_emotion, logits_intensity) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class EmotionsDataset(Dataset):
    """Dataset class for inference (Same as notebook)"""
    def __init__(self, texts, emotion_labels=None, intensity_labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.emotion_labels = emotion_labels
        self.intensity_labels = intensity_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in encoding.items()}
        
        # For inference, we might not have labels
        if self.emotion_labels is not None and self.intensity_labels is not None:
            emotion_label = self.emotion_labels[idx]
            intensity_label = self.intensity_labels[idx]
            item["labels"] = torch.tensor([emotion_label, intensity_label], dtype=torch.long)
        
        return item

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model, tokenizer, and label mappings"""
    
    model_name = "ashrafulparan/Emotion-BERT"
    
    try:
        st.info("Loading model from Hugging Face...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load config
        config = AutoConfig.from_pretrained(model_name)
        
        # Try to download and load label mappings
        try:
            from huggingface_hub import hf_hub_download
            labels_file = hf_hub_download(repo_id=model_name, filename="labels_mapping.json")
            with open(labels_file, 'r', encoding='utf-8') as f:
                label_mappings = json.load(f)
        except:
            # Use local labels_mapping.json if available
            try:
                with open('labels_mapping.json', 'r', encoding='utf-8') as f:
                    label_mappings = json.load(f)
            except:
                # Default mappings based on the notebook
                label_mappings = {
                    "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise"],
                    "intensities": ["0.0", "1.0", "2.0"],
                    "emotion_to_id": {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5},
                    "id_to_emotion": {"0": "angry", "1": "disgust", "2": "fear", "3": "happy", "4": "sad", "5": "surprise"},
                    "intensity_to_id": {"0.0": 0, "1.0": 1, "2.0": 2},
                    "id_to_intensity": {"0": "0.0", "1": "1.0", "2": "2.0"}
                }
        
        emotions = label_mappings["emotions"]
        intensities = label_mappings["intensities"]
        id_to_emotion = {int(k): v for k, v in label_mappings["id_to_emotion"].items()}
        id_to_intensity = {int(k): v for k, v in label_mappings["id_to_intensity"].items()}
        
        # Create model with correct dimensions
        model = BertForMultiTaskClassification.from_pretrained(
            model_name,
            num_emotions=len(emotions),
            num_intensities=len(intensities)
        )
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded successfully from {model_name}")
        st.info(f"📋 Emotions: {emotions}")
        st.info(f"📋 Intensities: {intensities}")
        st.info(f"🖥️ Device: {device}")
        
        return model, tokenizer, id_to_emotion, id_to_intensity, emotions, intensities
        
    except Exception as e:
        st.error(f"❌ Error loading model from Hugging Face: {e}")
        st.error("Please ensure the model 'ashrafulparan/Emotion-BERT' is available and accessible.")
        return None, None, None, None, None, None

def predict_emotions_and_intensities(texts, model, tokenizer, id_to_emotion, id_to_intensity, batch_size=32):
    """
    Perform inference on a list of texts using the trained model (Same logic as notebook).
    Returns predictions in the same format as training evaluation.
    """
    # Create dataset for inference
    dataset = EmotionsDataset(texts, tokenizer=tokenizer)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Store predictions
    all_emotion_logits = []
    all_intensity_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device (CPU/GPU)
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            
            # Get model predictions
            outputs = model(**batch)
            logits_emotion, logits_intensity = outputs[0], outputs[1]
            
            # Store logits
            all_emotion_logits.append(logits_emotion.cpu().numpy())
            all_intensity_logits.append(logits_intensity.cpu().numpy())
    
    # Concatenate all predictions
    emotion_logits = np.concatenate(all_emotion_logits, axis=0)
    intensity_logits = np.concatenate(all_intensity_logits, axis=0)
    
    # Convert logits to predictions (same as training)
    emotion_predictions = np.argmax(emotion_logits, axis=1)
    intensity_predictions = np.argmax(intensity_logits, axis=1)
    
    # Convert predictions to labels
    predicted_emotions = [id_to_emotion[pred] for pred in emotion_predictions]
    predicted_intensities = [id_to_intensity[pred] for pred in intensity_predictions]
    
    # Get probabilities
    emotion_probs = softmax(torch.tensor(emotion_logits), dim=1).numpy()
    intensity_probs = softmax(torch.tensor(intensity_logits), dim=1).numpy()
    
    # Create results dataframe (same format as training)
    results_df = pd.DataFrame({
        'text': texts,
        'predicted_emotion_id': emotion_predictions,
        'predicted_intensity_id': intensity_predictions,
        'predicted_emotion': predicted_emotions,
        'predicted_intensity': predicted_intensities
    })
    
    return results_df, emotion_logits, intensity_logits, emotion_probs, intensity_probs

def create_emotion_chart(probabilities, labels, predicted_emotion):
    """Create a bar chart for emotion probabilities"""
    df = pd.DataFrame({
        'Emotion': [emotion.title() for emotion in labels],
        'Probability': probabilities
    })
    df = df.sort_values('Probability', ascending=True)
    
    # Color bars, highlight the predicted emotion
    colors = ['#FF6B6B' if emotion.lower() == predicted_emotion.lower() else '#4ECDC4' 
              for emotion in df['Emotion']]
    
    fig = px.bar(df, x='Probability', y='Emotion', orientation='h',
                 title='🎭 Emotion Probabilities',
                 color='Probability',
                 color_continuous_scale='viridis')
    
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        height=300, 
        showlegend=False,
        xaxis=dict(title="Probability", range=[0, 1]),
        yaxis=dict(title="Emotion")
    )
    
    return fig

def create_intensity_chart(probabilities, labels, predicted_intensity):
    """Create a bar chart for intensity probabilities"""
    # Map intensity labels to more readable format
    intensity_map = {"0.0": "Low", "1.0": "Medium", "2.0": "High"}
    readable_labels = [intensity_map.get(label, label) for label in labels]
    
    df = pd.DataFrame({
        'Intensity': readable_labels,
        'Probability': probabilities
    })
    df = df.sort_values('Probability', ascending=True)
    
    # Color bars, highlight the predicted intensity
    predicted_readable = intensity_map.get(predicted_intensity, predicted_intensity)
    colors = ['#FF6B6B' if intensity == predicted_readable else '#FFB347' 
              for intensity in df['Intensity']]
    
    fig = px.bar(df, x='Probability', y='Intensity', orientation='h',
                 title='📈 Intensity Probabilities',
                 color='Probability',
                 color_continuous_scale='plasma')
    
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        height=250, 
        showlegend=False,
        xaxis=dict(title="Probability", range=[0, 1]),
        yaxis=dict(title="Intensity")
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    st.title("🇧🇩 Bangla Emotion and Intensity Detection")
    st.markdown("### Analyze emotions and their intensity in Bangla text using fine-tuned BERT")
    st.markdown("**Model:** [ashrafulparan/Emotion-BERT](https://huggingface.co/ashrafulparan/Emotion-BERT)")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📖 About")
        st.write("This app uses a fine-tuned BERT model to detect emotions and their intensity in Bangla text.")
        st.write("**Model:** ashrafulparan/Emotion-BERT")
        st.write("**Emotions:** Angry, Disgust, Fear, Happy, Sad, Surprise")
        st.write("**Intensity Levels:** Low (0.0), Medium (1.0), High (2.0)")
        
        st.header("🔧 How to use")
        st.write("1. Enter your Bangla text in the text area")
        st.write("2. Click 'Analyze Emotion' button")
        st.write("3. View the predicted emotion and intensity")
        st.write("4. Explore the probability distributions")
        
        st.header("📊 Model Info")
        st.write("- Based on Bangla BERT")
        st.write("- Multi-task classification")
        st.write("- Trained on Bangla emotion dataset")
        st.write("- Follows exact notebook inference logic")
        
        # Sample texts for quick testing
        st.header("🎯 Quick Examples")
        sample_texts = [
            "আমি খুব খুশি আজকে।",
            "এটা খুব দুঃখজনক খবর।",
            "ভোটের হার কম হলে দোষ, বেশি হলে দোষ, ভোটের সময় মারামারি না হওয়াটাও দোষের, আসলে সমালোচকরা কী চায় ??",
            "এই দৃশ্যটা দেখে আমি অবাক হয়ে গেছি।",
            "আমি তোমাকে ভালোবাসি।"
        ]
        
        st.write("Click to try these examples:")
        for i, text in enumerate(sample_texts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.text_input = text
    
    # Load model
    with st.spinner("🔄 Loading model from Hugging Face..."):
        model, tokenizer, id_to_emotion, id_to_intensity, emotions, intensities = load_model_and_tokenizer()
    
    if model is None:
        st.error("Failed to load model. Please check your connection and try again.")
        st.stop()
    
    # Input section
    st.header("📝 Input Text")
    
    # Get text from session state if available
    default_text = st.session_state.get('text_input', "আমি আজ খুব খুশি এবং উৎসাহিত বোধ করছি")
    
    # Text input
    text_input = st.text_area(
        "Enter Bangla text to analyze:",
        value=default_text,
        height=120,
        placeholder="Type your Bangla text here...",
        help="Enter any Bangla text to analyze its emotion and intensity"
    )
    
    # Update session state
    st.session_state.text_input = text_input
    
    # Analysis section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    if analyze_button:
        if text_input.strip():
            with st.spinner("🧠 Analyzing emotion and intensity..."):
                # Make prediction using the same logic as notebook
                try:
                    results_df, emotion_logits, intensity_logits, emotion_probs, intensity_probs = predict_emotions_and_intensities(
                        [text_input], model, tokenizer, id_to_emotion, id_to_intensity, batch_size=1
                    )
                    
                    # Get results for the single text
                    result = results_df.iloc[0]
                    emotion = result['predicted_emotion']
                    intensity = result['predicted_intensity']
                    emotion_prob = emotion_probs[0]
                    intensity_prob = intensity_probs[0]
                    
                    # Get confidence scores
                    emotion_confidence = emotion_prob[result['predicted_emotion_id']]
                    intensity_confidence = intensity_prob[result['predicted_intensity_id']]
                    
                    # Results section
                    st.header("📊 Analysis Results")
                    
                    # Main results with better formatting
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Emotion result with emoji
                        emotion_emojis = {
                            'angry': '😠', 'disgust': '🤢', 'fear': '😨', 
                            'happy': '😊', 'sad': '😢', 'surprise': '😮'
                        }
                        emoji = emotion_emojis.get(emotion, '😐')
                        
                        st.metric(
                            label="🎭 Predicted Emotion",
                            value=f"{emoji} {emotion.title()}",
                            delta=f"Confidence: {emotion_confidence:.1%}"
                        )
                    
                    with col2:
                        # Intensity result
                        intensity_map = {"0.0": "Low", "1.0": "Medium", "2.0": "High"}
                        intensity_readable = intensity_map.get(intensity, intensity)
                        intensity_emojis = {"Low": "🔻", "Medium": "🔸", "High": "🔺"}
                        intensity_emoji = intensity_emojis.get(intensity_readable, "📊")
                        
                        st.metric(
                            label="📈 Predicted Intensity",
                            value=f"{intensity_emoji} {intensity_readable}",
                            delta=f"Confidence: {intensity_confidence:.1%}"
                        )
                    
                    # Visualizations
                    st.header("📈 Probability Distributions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Emotion probabilities
                        fig_emotion = create_emotion_chart(emotion_prob, emotions, emotion)
                        st.plotly_chart(fig_emotion, use_container_width=True)
                    
                    with col2:
                        # Intensity probabilities
                        fig_intensity = create_intensity_chart(intensity_prob, intensities, intensity)
                        st.plotly_chart(fig_intensity, use_container_width=True)
                    
                    # Detailed results in expandable section
                    with st.expander("🔍 Detailed Analysis"):
                        st.subheader("📊 All Emotion Probabilities")
                        for i, (emotion_name, prob) in enumerate(zip(emotions, emotion_prob)):
                            emoji = emotion_emojis.get(emotion_name, '😐')
                            is_predicted = (i == result['predicted_emotion_id'])
                            marker = "👉 " if is_predicted else "   "
                            st.write(f"{marker}{emoji} **{emotion_name.title()}**: {prob:.3f} ({prob*100:.1f}%)")
                        
                        st.subheader("📈 All Intensity Probabilities")
                        for i, (intensity_name, prob) in enumerate(zip(intensities, intensity_prob)):
                            readable = intensity_map.get(intensity_name, intensity_name)
                            emoji = intensity_emojis.get(readable, "📊")
                            is_predicted = (i == result['predicted_intensity_id'])
                            marker = "👉 " if is_predicted else "   "
                            st.write(f"{marker}{emoji} **{readable} ({intensity_name})**: {prob:.3f} ({prob*100:.1f}%)")
                        
                        # Technical details
                        st.subheader("⚙️ Technical Details")
                        st.write(f"**Text Length:** {len(text_input)} characters")
                        st.write(f"**Emotion ID:** {result['predicted_emotion_id']}")
                        st.write(f"**Intensity ID:** {result['predicted_intensity_id']}")
                        st.write(f"**Model:** ashrafulparan/Emotion-BERT")
                        st.write(f"**Device:** {next(model.parameters()).device}")
                    
                    # Show raw results in JSON format (for developers)
                    with st.expander("🛠️ Raw Results (JSON)"):
                        raw_results = {
                            "text": text_input,
                            "predicted_emotion": emotion,
                            "predicted_intensity": intensity,
                            "emotion_confidence": float(emotion_confidence),
                            "intensity_confidence": float(intensity_confidence),
                            "emotion_probabilities": {emotions[i]: float(prob) for i, prob in enumerate(emotion_prob)},
                            "intensity_probabilities": {intensities[i]: float(prob) for i, prob in enumerate(intensity_prob)}
                        }
                        st.json(raw_results)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    st.error("Please try again or check if the input text is valid.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Additional features
    st.header("🚀 Batch Analysis")
    st.write("Analyze multiple texts at once:")
    
    batch_texts = st.text_area(
        "Enter multiple texts (one per line):",
        placeholder="ভোটের হার কম হলে দোষ, বেশি হলে দোষ, ভোটের সময় মারামারি না হওয়াটাও দোষের, আসলে সমালোচকরা কী চায় ??",
        height=100
    )
    
    if st.button("🔍 Analyze Batch", type="secondary"):
        if batch_texts.strip():
            texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
            if texts:
                with st.spinner(f"🧠 Analyzing {len(texts)} texts..."):
                    try:
                        results_df, _, _, emotion_probs, intensity_probs = predict_emotions_and_intensities(
                            texts, model, tokenizer, id_to_emotion, id_to_intensity, batch_size=8
                        )
                        
                        st.subheader("📊 Batch Results")
                        
                        # Display results in a nice table
                        display_df = results_df.copy()
                        display_df['predicted_emotion'] = display_df['predicted_emotion'].str.title()
                        intensity_map = {"0.0": "Low", "1.0": "Medium", "2.0": "High"}
                        display_df['predicted_intensity'] = display_df['predicted_intensity'].map(intensity_map)
                        
                        st.dataframe(
                            display_df[['text', 'predicted_emotion', 'predicted_intensity']],
                            use_container_width=True,
                            column_config={
                                "text": "Text",
                                "predicted_emotion": "Emotion",
                                "predicted_intensity": "Intensity"
                            }
                        )
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="emotion_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch analysis: {e}")
            else:
                st.warning("⚠️ Please enter valid texts (one per line).")
        else:
            st.warning("⚠️ Please enter some texts to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ using [Streamlit](https://streamlit.io) and [Hugging Face Transformers](https://huggingface.co/transformers/) | "
        "Model: [ashrafulparan/Emotion-BERT](https://huggingface.co/ashrafulparan/Emotion-BERT)"
    )

if __name__ == "__main__":
    main()
