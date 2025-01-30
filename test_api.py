import requests
import json

def test_intent_classifier():
    url = "http://localhost:5001/classify"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": "http://localhost:3000"
    }
    
    test_texts = [
        "what's the weather like today",
        "set an alarm for 7am",
        "play some music",
        "remind me to call mom"
    ]
    
    print("Connecting to:", url)
    for text in test_texts:
        payload = {"text": text}
        try:
            print(f"\nTesting: {text}")
            print("Sending request...")
            response = requests.post(url, headers=headers, json=payload)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Intent: {result['intent']}")
                print(f"Confidence: {result['confidence']:.2f}%")
            else:
                print(f"Error Response: {response.text}")
                print(f"Headers: {response.headers}")
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")

if __name__ == "__main__":
    test_intent_classifier() 