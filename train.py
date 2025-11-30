import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import os

# Load and prepare dataset
df = pd.read_csv("fraud_messages.csv")  # Dataset with columns: message, label
df['label'] = df['label'].map({'not fraud': 0, 'fraud': 1})  # Convert to numeric labels

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['message'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Dataset class
class FraudDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx])}
    
    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = FraudDataset(train_encodings, train_labels)
val_dataset = FraudDataset(val_encodings, val_labels)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Evaluate model
preds_output = trainer.predict(val_dataset)
predictions = preds_output.predictions.argmax(axis=1)
print("\nClassification Report:\n")
print(classification_report(val_labels, predictions))

# ✅ Save the model and tokenizer
save_path = "./fraud_detection_model"
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\nModel and tokenizer saved in directory: {save_path}")
