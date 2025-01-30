import tkinter as tk
from tkinter import ttk, scrolledtext
import requests
import json

class IntentClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intent Classifier")
        self.root.geometry("600x400")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input field
        self.input_label = ttk.Label(main_frame, text="Enter your request:")
        self.input_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.input_field = ttk.Entry(main_frame, width=50)
        self.input_field.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Submit button
        self.submit_button = ttk.Button(main_frame, text="Classify Intent", command=self.classify_intent)
        self.submit_button.grid(row=1, column=1, padx=5)
        
        # Output area
        self.output_area = scrolledtext.ScrolledText(main_frame, width=60, height=15, wrap=tk.WORD)
        self.output_area.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Bind Enter key to submit
        self.input_field.bind('<Return>', lambda e: self.classify_intent())
        
        # Configure grid
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
    def classify_intent(self):
        text = self.input_field.get()
        if not text:
            return
            
        try:
            # Make API request to your backend
            response = requests.post('http://localhost:5000/classify', 
                                  json={'text': text})
            
            if response.status_code == 200:
                result = response.json()
                # Format and display result
                self.display_result(text, result)
            else:
                self.output_area.insert(tk.END, f"Error: {response.status_code}\n")
                
        except requests.exceptions.RequestException as e:
            self.output_area.insert(tk.END, f"Error connecting to server: {str(e)}\n")
            
        # Clear input field
        self.input_field.delete(0, tk.END)
        
    def display_result(self, text, result):
        # Clear previous output
        self.output_area.delete(1.0, tk.END)
        
        # Display input and predicted intent
        self.output_area.insert(tk.END, f"Input: {text}\n\n")
        self.output_area.insert(tk.END, f"Predicted Intent: {result['intent']}\n")
        self.output_area.insert(tk.END, f"Confidence: {result['confidence']:.2f}%\n\n")
        
        # Make the new text visible
        self.output_area.see(tk.END)

def main():
    root = tk.Tk()
    app = IntentClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 