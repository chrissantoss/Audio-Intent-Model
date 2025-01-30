import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from datasets import Dataset
from tqdm import tqdm
import json
import numpy as np
import os
from sklearn.model_selection import KFold
from collections import Counter

def augment_text(text):
    # Simple augmentation techniques
    augmented = []
    
    # Add polite variations
    augmented.append(f"please {text}")
    augmented.append(f"could you {text}")
    augmented.append(f"i would like to {text}")
    
    # Add time-based variations if relevant
    if any(word in text.lower() for word in ['timer', 'alarm', 'reminder', 'schedule']):
        augmented.append(f"{text} right now")
        augmented.append(f"{text} please")
    
    return augmented

# Define validation function
def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Convert input_ids to Long type
            input_ids = batch["input_ids"].to(torch.long).to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(torch.long).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Load training data from JSON file
with open('training_data.json', 'r') as f:
    data = json.load(f)

# Check and standardize data format
standardized_data = []
for item in data:
    if "label" in item:
        standardized_data.append({
            "text": item["text"],
            "intent": item["label"],
            "language": item.get("language", "en")  # Default to English if not specified
        })
    elif "intent" in item:
        standardized_data.append({
            "text": item["text"],
            "intent": item["intent"],
            "language": item.get("language", "en")
        })
    else:
        print(f"Skipping malformed data item: {item}")
        continue

data = standardized_data

# Add more high-quality training examples
additional_data = [
    # Play music intent variations
    {"text": "play my favorite songs", "intent": "play_music"},
    {"text": "start my workout playlist", "intent": "play_music"},
    {"text": "play some rock music", "intent": "play_music"},
    {"text": "put on some jazz", "intent": "play_music"},
    {"text": "play the latest album by Taylor Swift", "intent": "play_music"},
    {"text": "start my study playlist", "intent": "play_music"},
    {"text": "play something relaxing", "intent": "play_music"},
    {"text": "play top hits playlist", "intent": "play_music"},

    # Calendar event variations
    {"text": "schedule a meeting for tomorrow at 2pm", "intent": "calendar_event"},
    {"text": "add dentist appointment to my calendar", "intent": "calendar_event"},
    {"text": "create a new event for team sync", "intent": "calendar_event"},
    {"text": "schedule lunch meeting with John", "intent": "calendar_event"},
    {"text": "add birthday party to calendar", "intent": "calendar_event"},
    {"text": "create event for project review", "intent": "calendar_event"},
    {"text": "schedule weekly standup", "intent": "calendar_event"},
    {"text": "add doctor's appointment to calendar", "intent": "calendar_event"},

    # Weather variations
    {"text": "what's the temperature outside", "intent": "get_weather"},
    {"text": "will it rain today", "intent": "get_weather"},
    {"text": "what's the forecast for tomorrow", "intent": "get_weather"},
    {"text": "is it going to be sunny", "intent": "get_weather"},
    {"text": "check weather for this weekend", "intent": "get_weather"},
    {"text": "what's the humidity today", "intent": "get_weather"},

    # Timer variations
    {"text": "set a timer for 10 minutes", "intent": "set_timer"},
    {"text": "start countdown for 5 minutes", "intent": "set_timer"},
    {"text": "timer 30 minutes", "intent": "set_timer"},
    {"text": "countdown 15 minutes", "intent": "set_timer"},

    # Alarm variations
    {"text": "wake me up at 7am", "intent": "set_alarm"},
    {"text": "set an alarm for 6:30 tomorrow", "intent": "set_alarm"},
    {"text": "alarm for 8am", "intent": "set_alarm"},
    {"text": "set morning alarm", "intent": "set_alarm"},

    # Reminder variations
    {"text": "remind me to take medicine at 2pm", "intent": "set_reminder"},
    {"text": "set a reminder for grocery shopping", "intent": "set_reminder"},
    {"text": "remind me about the meeting tomorrow", "intent": "set_reminder"},
    {"text": "set reminder for dad's birthday", "intent": "set_reminder"},

    # Web search variations
    {"text": "search for nearby restaurants", "intent": "web_search"},
    {"text": "look up recipe for pasta", "intent": "web_search"},
    {"text": "find information about mars", "intent": "web_search"},
    {"text": "search best movies 2023", "intent": "web_search"},

    # Date variations
    {"text": "what day is it today", "intent": "get_date"},
    {"text": "what's today's date", "intent": "get_date"},
    {"text": "tell me the date", "intent": "get_date"},
    {"text": "what day is tomorrow", "intent": "get_date"},

    # Directions variations
    {"text": "how do I get to the airport", "intent": "get_directions"},
    {"text": "navigate to nearest gas station", "intent": "get_directions"},
    {"text": "directions to downtown", "intent": "get_directions"},
    {"text": "find route to shopping mall", "intent": "get_directions"},

    # Time variations
    {"text": "what time is it", "intent": "get_time"},
    {"text": "tell me the current time", "intent": "get_time"},
    {"text": "what's the time in Tokyo", "intent": "get_time"},
    {"text": "check time in London", "intent": "get_time"},

    # More complex music queries with context
    {"text": "play my workout playlist while I exercise", "intent": "play_music"},
    {"text": "play some relaxing jazz for studying", "intent": "play_music"},
    {"text": "play the latest hits from spotify", "intent": "play_music"},
    {"text": "can you play my bedtime playlist", "intent": "play_music"},
    {"text": "play some background music for dinner", "intent": "play_music"},

    # Calendar events with more context
    {"text": "schedule a one hour meeting with the team", "intent": "calendar_event"},
    {"text": "add my doctor's appointment for next tuesday at 3pm", "intent": "calendar_event"},
    {"text": "create a recurring meeting every monday at 9am", "intent": "calendar_event"},
    {"text": "block my calendar for the presentation tomorrow", "intent": "calendar_event"},
    {"text": "schedule a video call with clients at 2pm", "intent": "calendar_event"},

    # More natural weather queries
    {"text": "do i need an umbrella today", "intent": "get_weather"},
    {"text": "how cold is it going to be tomorrow", "intent": "get_weather"},
    {"text": "check if it's going to snow this weekend", "intent": "get_weather"},
    {"text": "what should i wear today weather wise", "intent": "get_weather"},

    # Complex timer requests
    {"text": "set a timer for the cookies in the oven", "intent": "set_timer"},
    {"text": "start a 45 minute workout timer", "intent": "set_timer"},
    {"text": "set a quick 3 minute timer", "intent": "set_timer"},

    # Natural alarm requests
    {"text": "wake me up before sunrise", "intent": "set_alarm"},
    {"text": "set my morning alarm for work", "intent": "set_alarm"},
    {"text": "i need to wake up at 5am tomorrow", "intent": "set_alarm"},

    # Contextual reminders
    {"text": "remind me to call mom when i get home", "intent": "set_reminder"},
    {"text": "set a reminder for my medication every 8 hours", "intent": "set_reminder"},
    {"text": "remind me about the meeting in 30 minutes", "intent": "set_reminder"},

    # Specific web searches
    {"text": "find reviews for the new iphone", "intent": "web_search"},
    {"text": "search for vegetarian restaurants nearby", "intent": "web_search"},
    {"text": "look up how to fix a leaky faucet", "intent": "web_search"},
]

# Extend the original dataset
data.extend(additional_data)

# Augment training data
print("Augmenting training data...")
augmented_data = []
for item in data:
    augmented_data.append(item)  # Keep original
    for aug_text in augment_text(item["text"]):
        augmented_data.append({"text": aug_text, "intent": item["intent"]})
data = augmented_data
print(f"Dataset size after augmentation: {len(data)} examples")

# Create label mappings
unique_labels = sorted(set(item["intent"] for item in data))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(unique_labels)

print(f"Labels: {label2id}")

# Prepare dataset with mapped labels
def map_labels(example):
    example["label"] = label2id[example["intent"]]
    return example

dataset = Dataset.from_list(data)
dataset = dataset.map(map_labels)

# Load model with correct number of labels
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "./intent_model"

if os.path.exists(model_path):
    print("Loading existing model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        dropout=0.1
    )

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set number of epochs
epochs = 12

# Initialize K-fold cross validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Track metrics across folds
fold_accuracies = []
best_accuracy = 0.0
best_fold = 0

# Convert dataset to numpy for splitting
dataset_array = np.array(data)

# Set gradient clipping
max_grad_norm = 1.0

# Add class weights to handle imbalance
intent_counts = Counter(item["intent"] for item in data)
total_samples = len(data)
class_weights = {label: total_samples / (len(unique_labels) * count) 
                for label, count in intent_counts.items()}
class_weights_tensor = torch.tensor([class_weights[id2label[i]] 
                                   for i in range(len(unique_labels))]).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

# Perform k-fold cross validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_array)):
    print(f"\nTraining Fold {fold + 1}/{n_splits}")
    
    # Split data for this fold
    train_data = dataset_array[train_idx].tolist()
    val_data = dataset_array[val_idx].tolist()
    
    # Create datasets for this fold
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Map labels
    train_dataset = train_dataset.map(map_labels)
    val_dataset = val_dataset.map(map_labels)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            text_pair=[x["language"]] * len(x["text"])
        ),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            text_pair=[x["language"]] * len(x["text"])
        ),
        batched=True
    )
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        dropout=0.1
    ).to(device)
    
    # Training setup for this fold
    optimizer = AdamW(model.parameters(), lr=5e-6)
    num_training_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 4,
        num_training_steps=num_training_steps
    )
    
    # Training loop for this fold
    fold_best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(torch.long).to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(torch.long).to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            loop.set_description(f"Epoch {epoch + 1}/{epochs}")
            loop.set_postfix(loss=loss.item())
        
        # Validation
        val_accuracy = validate_model(model, val_loader, device)
        print(f"Fold {fold + 1}, Epoch {epoch + 1} - Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save best model for this fold
        if val_accuracy > fold_best_accuracy:
            fold_best_accuracy = val_accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_fold = fold
                print(f"New best model! Saving... (Accuracy: {val_accuracy:.2f}%)")
                model.save_pretrained(f"./intent_model_fold{fold}")
                tokenizer.save_pretrained(f"./intent_model_fold{fold}")
        
    fold_accuracies.append(fold_best_accuracy)

# Print cross-validation results
print("\nCross-validation Results:")
for fold, accuracy in enumerate(fold_accuracies):
    print(f"Fold {fold + 1}: {accuracy:.2f}%")
print(f"\nAverage Accuracy: {np.mean(fold_accuracies):.2f}%")
print(f"Best Accuracy: {best_accuracy:.2f}% (Fold {best_fold + 1})")
