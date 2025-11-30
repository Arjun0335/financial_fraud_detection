import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from transformers import BertTokenizer, BertForSequenceClassification

# ✅ Load dataset
df = pd.read_csv("fraud_messages.csv")  # Dataset with columns: message, label
df['label'] = df['label'].map({'not fraud': 0, 'fraud': 1})  # Convert to numeric labels

# Split data (same as training)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['message'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# ✅ Load saved model and tokenizer
model_path = "./fraud_detection_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Put model in evaluation mode
model.eval()

# Tokenize validation data
val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

# Run predictions
with torch.no_grad():
    outputs = model(**val_encodings)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # Probability of fraud
    predictions = torch.argmax(logits, dim=1).numpy()

y_true = val_labels

# ✅ Classification report & accuracy
print("\nClassification Report:\n")
print(classification_report(y_true, predictions))
print("Accuracy:", accuracy_score(y_true, predictions))

# ✅ ROC-AUC score
roc_auc = roc_auc_score(y_true, probs)
print("ROC-AUC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, probs)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
