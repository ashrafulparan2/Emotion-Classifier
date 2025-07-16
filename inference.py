"""
Simple inference script for Bangla Emotion and Intensity Detection
Just predicts emotion and intensity for a single text and prints the result.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertPreTrainedModel, BertModel
from torch.nn.functional import softmax

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

def load_model_and_tokenizer(model_dir="/kaggle/input/emotion-model/saved_bangla_emotion_model"):
    """Load the trained model, tokenizer, and label mappings"""
    
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
    from transformers import AutoConfig
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

def predict_emotion_intensity(text, model, tokenizer, id_to_emotion, id_to_intensity):
    """Predict emotion and intensity for a single text"""
    
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
        
        return pred_emotion, pred_intensity, emotion_confidence, intensity_confidence

def main():
    """Main function"""
    
    # Text to analyze (change this to your text)
    text = "যার জন্য চুরি করি সেই বলে চোর !!"
    
    # print("Loading model...")
    
    try:
        # Load model and tokenizer
        model, tokenizer, id_to_emotion, id_to_intensity = load_model_and_tokenizer()
        
        # print("Model loaded successfully!")
        print(f"Analyzing text: '{text}'")
        print("-" * 50)
        
        # Make prediction
        emotion, intensity, emotion_conf, intensity_conf = predict_emotion_intensity(
            text, model, tokenizer, id_to_emotion, id_to_intensity
        )
        
        # Print results
        print(f"Predicted Emotion: {emotion}")
        print(f"Emotion Confidence: {emotion_conf:.3f}")
        print(f"Predicted Intensity: {intensity}")
        print(f"Intensity Confidence: {intensity_conf:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the 'saved_bangla_emotion_model' directory exists and contains the trained model.")

if __name__ == "__main__":
    main()
