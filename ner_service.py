from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained NER model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=device)

# Example inputs with named entities
texts = [
    "John works at Microsoft in New York.",
    "Apple announced their new iPhone at WWDC in California.",
    "Sarah Johnson visited Google's headquarters last Friday."
]

# Process each text
for text in texts:
    print(f"\nText: {text}")
    entities = ner_pipeline(text)
    print("Detected entities:")
    for entity in entities:
        print(f"- {entity['word']}: {entity['entity']} (confidence: {entity['score']:.2f})")
