import pandas as pd
from datasets import Dataset
from utils import clean_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Step 1: Load the Test Dataset
test_dataset_path = './data/price_queries.csv'
test_df = pd.read_csv(test_dataset_path)

# Map string labels to numerical values and clean text
label_mapping = {"contact agent": 0, "other": 1}
test_df['label'] = test_df['label'].map(label_mapping)
test_df['query'] = test_df['query'].map(clean_text)

# Convert to Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)

# Step 2: Load the Trained Model and Tokenizer
model_path = './results/model-6'  
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Step 3: Preprocess the Test Dataset
def preprocess_function(examples):
    return tokenizer(examples['query'], padding='max_length', truncation=True, max_length=50)

test_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 4: Make Predictions and Evaluate
trainer = Trainer(model=model)

# Make predictions
predictions = trainer.predict(test_dataset)

# Extract the logits and apply softmax to get probabilities
logits = predictions.predictions
predicted_labels = logits.argmax(axis=-1)

# Get the true labels
true_labels = test_dataset['label']

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

# Find indices where predictions do not match true labels
misclassified_indices = [i for i in range(len(true_labels)) if true_labels[i] != predicted_labels[i]]

# Step 6: Extract Mislabeled Data Points
misclassified_df = test_df.loc[misclassified_indices, :]
misclassified_df['predicted_label'] = predicted_labels[misclassified_indices]

# Step 7: Display or Save the Mislabeled Data Points
print(misclassified_df[['query', 'label', 'predicted_label']])

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
