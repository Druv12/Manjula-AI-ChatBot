import json
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# ----------------------
# Training data examples
# ----------------------
train_data = [
    {"instruction": "Translate this sentence to French:", "input": "I love programming.", "output": "J'adore programmer."},
    {"instruction": "Summarize the paragraph:", "input": "Python is a popular language used for AI.", "output": "Python is widely used for AI."},
    {"instruction": "Answer the question:", "input": "Who developed the theory of relativity?", "output": "Albert Einstein."},
    {"instruction": "Write a Python function to add two numbers", "input": "", "output": "def add(a, b):\n    return a + b"},
    {"instruction": "Explain the concept of recursion in simple terms.", "input": "", "output": "Recursion is when a function calls itself to solve a smaller part of a problem until it reaches a base case."}
]

# --------------------------
# Validation data examples
# --------------------------
validation_data = [
    {"instruction": "Translate to Hindi:", "input": "Good evening", "output": "शुभ संध्या"},
    {"instruction": "Summarize:", "input": "Machine learning is part of AI that learns patterns from data.", "output": "ML learns patterns from data."},
    {"instruction": "Answer the question:", "input": "What is the capital of France?", "output": "Paris."}
]

# --------------------------
# Write train.jsonl
# --------------------------
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# --------------------------
# Write validation.jsonl
# --------------------------
with open("data/validation.jsonl", "w", encoding="utf-8") as f:
    for item in validation_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Created train.jsonl and validation.jsonl in the data/ folder")
