import torch
from transformers import (
    pipeline
)
model_path = "D:/VSCODE/namasoft/best_model"

pipe = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0 if torch.cuda.is_available() else -1)

queries = [
    "Develop a multi-step plan to reduce carbon emissions in a mid-sized city.",
    "The model should understand moderately difficult content.",
    "Quantum physics requires advanced understanding.",
    "Develop a multi-step plan to reduce carbon emissions in a mid-sized city, considering economic, social, and political factors."
]

for q in queries:
    pred = pipe(q)[0]
    print(f"🔹 Text: {q}")
    print(f"   → Predicted: {pred['label']} (Score: {pred['score']:.4f})\n")

def classify_text(text):
    pred = pipe(text)[0]
    return pred['label']
