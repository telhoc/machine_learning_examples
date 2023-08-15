# Reuse existing model for text classification 
from transformers import DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer

# Load the saved model
model_path = './models/'
classifier = DistilBertForSequenceClassification.from_pretrained(model_path)

# Load SBERT for sentence embeddings
sbert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def classify_text(text):
    # Convert text to embedding
    embedding = sbert_model.encode([text], convert_to_tensor=True)

    # Use the model for prediction
    with torch.no_grad():
        logits = classifier(embedding)[0]
    predicted_class_idx = torch.argmax(logits, dim=1).item()

    # Convert class index back to original class name
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_class

# Example usage
sample_text = "I am testing the classifier"
result = classify_text(sample_text)
print(f"The text belongs to the class: {result}")

