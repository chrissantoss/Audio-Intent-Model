import requests
from requests.exceptions import ConnectionError, JSONDecodeError

def test_api():
    url = "http://localhost:5001/predict"
    
    test_cases = [
        "What's the weather like today?",
        "Set a reminder for my meeting",
        "Play some rock music",
        "What about tomorrow?",
        "Remind me to call mom",
        "Is it going to rain?"
    ]
    
    try:
        # First test if server is running
        requests.get("http://localhost:5001/")
        
        # If server is running, proceed with test cases
        for text in test_cases:
            try:
                response = requests.post(url, json={"text": text})
                response.raise_for_status()  # Raise an error for bad status codes
                result = response.json()
                print(f"\nInput: {result['text']}")
                print(f"Predicted Intent: {result['intent']}")
            except JSONDecodeError:
                print(f"Error: Could not parse response for text: {text}")
            except Exception as e:
                print(f"Error processing text: {text}")
                print(f"Error details: {str(e)}")
                
    except ConnectionError:
        print("\nError: Could not connect to the server.")
        print("Please make sure to run 'python app.py' first in a separate terminal.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    test_api() 