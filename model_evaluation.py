import json
from intent_predictor import predict_intent
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

def evaluate_model():
    # Load test cases from testing_data.json
    with open('testing_data.json', 'r') as f:
        test_cases = [(item["text"], item["label"]) for item in json.load(f)]

    results = {
        "total": len(test_cases),
        "correct": 0,
        "total_confidence": 0.0
    }
    
    results_by_intent = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "avg_confidence": 0.0
    })

    print("\nModel Evaluation Results:")
    print("-" * 50)

    for text, expected_intent in test_cases:
        # Get prediction and calculate confidence
        predicted_intent = predict_intent(text)
        
        is_correct = predicted_intent == expected_intent
        
        # Update statistics
        if is_correct:
            results["correct"] += 1

        # Update intent-specific results
        intent_stats = results_by_intent[expected_intent]
        intent_stats["total"] += 1
        intent_stats["correct"] += int(is_correct)

        print(f"\nInput: {text}")
        print(f"Expected Intent: {expected_intent}")
        print(f"Predicted Intent: {predicted_intent}")
        print(f"Correct: {'✓' if is_correct else '✗'}")

    # Calculate and print overall metrics
    print("\n" + "=" * 50)
    print("Overall Results:")
    accuracy = (results["correct"] / results["total"]) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct Predictions: {results['correct']}/{results['total']}")

    # Print per-intent results
    print("\nPer-Intent Results:")
    print("-" * 50)
    for intent, stats in results_by_intent.items():
        intent_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"{intent}: {intent_accuracy:.2f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    evaluate_model()