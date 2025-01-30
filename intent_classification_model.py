# This is the main application file
# It uses the trained model to make actual predictions
# Key functions:
# - Loads your fine-tuned model
# - Processes input text
# - Returns predicted intents/entities
# Example usage would be detecting intents like:
# - Setting reminders
# - Making appointments
# - General queries

import nltk
nltk.download('punkt')

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import json

# Load test data
with open('testing_data.json', 'r') as f:
    data = json.load(f)

# Convert "label" to "intent" to match existing code
for item in data:
    item["intent"] = item["label"]
    del item["label"]

# Load and preprocess data
data = [
    {"text": "What's the weather like today?", "intent": "weather_query"},
    {"text": "Remind me to call John at 3 PM", "intent": "set_reminder"},
    {"text": "Play my workout playlist", "intent": "play_music"},
    {"text": "What about tomorrow?", "intent": "follow_up"},
]

# Prepare dataset
dataset = Dataset.from_list(data)
labels = list(set([item["intent"] for item in data]))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
dataset = dataset.map(lambda x: {"label": label2id[x["intent"]]})

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels)
)

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
