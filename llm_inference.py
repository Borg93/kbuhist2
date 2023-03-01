from transformers import pipeline

pred_model = pipeline("fill-mask", model = "./bert-base-cased-swedish-1800-accelerate",
                       tokenizer = "./bert-base-cased-swedish-1800-accelerate")

text = "En sådan svag mor, som tillät att man drog vagnen för att [MASK] ett friskt barn."

preds = pred_model(text)

for pred in preds:
    print(f">>> {pred['sequence']}")