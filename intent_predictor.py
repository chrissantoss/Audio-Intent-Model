from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict_intent(text):
    # Load the saved model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./intent_model")
    tokenizer = AutoTokenizer.from_pretrained("./intent_model")
    
    # Prepare the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        predicted_intent = model.config.id2label[predictions.item()]
    
    return predicted_intent

# Test some examples
test_texts = [
    "What's the weather going to be like?",
    "Set a reminder for my meeting tomorrow",
    "Play some rock music",
    "What time is it?",
]

for text in test_texts:
    intent = predict_intent(text)
    print(f"\nText: {text}")
    print(f"Predicted Intent: {intent}") 