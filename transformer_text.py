# Text classification using transformers. Data files should in the format of [sentence or text \t label]
# You need pip install transformers sentence-transformers
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load data
train = pd.read_csv('train.csv', delimiter='\t', header=None, names=['text', 'label'])
test = pd.read_csv('test.csv', delimiter='\t', header=None, names=['text', 'label'])

# Convert labels to integer indices
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train['label'])
y_test_encoded = label_encoder.transform(test['label'])

# Get sentence embeddings using SBERT
model_name = 'distilbert-base-nli-mean-tokens'
sbert_model = SentenceTransformer(model_name)
X_train_embeddings = sbert_model.encode(train['text'].to_list(), convert_to_tensor=True)
X_test_embeddings = sbert_model.encode(test['text'].to_list(), convert_to_tensor=True)

# Define classification model and training arguments
num_labels = len(label_encoder.classes_)
classifier = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    do_train=True,
    do_eval=True
)

# Train the model
trainer = Trainer(
    model=classifier,
    args=training_args,
    train_dataset=list(zip(X_train_embeddings, y_train_encoded)),
    eval_dataset=list(zip(X_test_embeddings, y_test_encoded))
)
trainer.train()

# Evaluate the model
predictions = trainer.predict(X_test_embeddings)[0].argmax(axis=-1)
accuracy = accuracy_score(y_test_encoded, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model and configuration
model_save_path = './models/'
trainer.save_model(model_save_path)
classifier.config.save_pretrained(model_save_path)


