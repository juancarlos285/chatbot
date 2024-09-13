import pandas as pd
from utils import clean_text
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch
import json

torch.mps.empty_cache()

# Load your dataset
dataset = pd.read_csv("data/contact_agent_dataset.csv")

# Turn labels to numerical values and clean text
label_mapping = {"contact agent": 0, "other": 1}
dataset["label"] = dataset["label"].map(label_mapping)
dataset["query"] = dataset["query"].map(clean_text)

# Split the dataset into train and test sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


# Preprocess the dataset
def preprocess_function(examples):
    # Tokenize the text
    inputs = tokenizer(
        examples["query"], padding="max_length", truncation=True, max_length=50
    )
    # Add the labels
    inputs["labels"] = examples["label"]
    return inputs


# Apply the preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set the format for PyTorch
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load the pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2, ignore_mismatched_sizes=True
)

# Add dropout layers
model.classifier.dropout = torch.nn.Dropout(p=0.3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    learning_rate=5e-6,
    lr_scheduler_type="linear",
    warmup_steps=50,
    num_train_epochs=5,
    weight_decay=0.05,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save model
model_folder = "./results/model-6"
trainer.save_model(model_folder)

# Save the training history to a JSON file
log_history = trainer.state.log_history
log_history_file = model_folder + "/log_history.json"

with open(log_history_file, "w") as f:
    json.dump(log_history, f, indent=4)
