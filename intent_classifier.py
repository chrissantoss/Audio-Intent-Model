from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class IntentClassifier:
    def __init__(self):
        # Load the model and tokenizer from the local directory
        model_path = "intent_model"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Define intent labels
        self.intent_labels = [
            "weather_query",
            "set_reminder",
            "play_music",
            "general_query",
            "time_query"
        ]
        
    def predict(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            
        # Get the predicted class and confidence
        probabilities = torch.softmax(predictions, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get the intent label
        predicted_intent = self.intent_labels[predicted_class]
        
        return predicted_intent, confidence 