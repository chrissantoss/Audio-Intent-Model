from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from language_detection import detect_language
from typing import Dict, Optional

# Initialize FastAPI app
app = FastAPI(title="Intent Recognition API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n" + "="*50)
print("Intent Recognition API!")
print("="*50 + "\n")

# Load models for different languages
models: Dict[str, AutoModelForSequenceClassification] = {
    'en': AutoModelForSequenceClassification.from_pretrained('./intent_model_en'),
    'es': AutoModelForSequenceClassification.from_pretrained('./intent_model_es'),
    'fr': AutoModelForSequenceClassification.from_pretrained('./intent_model_fr'),
    # Add more languages as needed
}

tokenizers: Dict[str, AutoTokenizer] = {
    'en': AutoTokenizer.from_pretrained('./intent_model_en'),
    'es': AutoTokenizer.from_pretrained('./intent_model_es'),
    'fr': AutoTokenizer.from_pretrained('./intent_model_fr'),
    # Add more languages as needed
}

# Move models to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
for model in models.values():
    model.to(device)

# Define request/response models
class IntentRequest(BaseModel):
    text: str
    language: Optional[str] = None

class IntentResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    language: str

@app.post("/classify", response_model=IntentResponse)
async def classify_intent(request: IntentRequest) -> IntentResponse:
    try:
        # Detect language if not provided
        language = request.language or detect_language(request.text)
        
        # Fall back to English if language not supported
        if language not in models:
            language = 'en'
        
        # Get appropriate model and tokenizer
        model = models[language]
        tokenizer = tokenizers[language]
        
        # Tokenize input
        inputs = tokenizer(
            request.text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1)
            confidence = probs[0][prediction].item() * 100
        
        # Get intent label
        intent = model.config.id2label[prediction.item()]
        
        return IntentResponse(
            text=request.text,
            intent=intent,
            confidence=confidence,
            language=language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 