import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.predict import predict

label, confidence = predict("data/test/sample.jpg")

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}")