import pandas as pd
from utils import clean_text
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch

torch.mps.empty_cache()

# Load your dataset
dataset = pd.read_csv('data/contact_agent_dataset.csv')

#Turn labels to numerical values and clean text
label_mapping = {' contact agent': 0, ' other': 1}
dataset[' label'] = dataset[' label'].map(label_mapping)
dataset['query'] = dataset['query'].map(clean_text)

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
    inputs = tokenizer(examples['query'], padding='max_length', truncation=True, max_length=50)
    # Add the labels
    inputs['labels'] = examples[' label']
    return inputs

# Apply the preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Set the format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load the pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base",
                                                            num_labels=2,
                                                            ignore_mismatched_sizes=True)
model.classifier.dropout = torch.nn.Dropout(p=0.3)  # Increase dropout probability

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='steps',
    eval_steps=10,
    save_strategy='steps',
    learning_rate=2e-5,
    lr_scheduler_type='linear',
    num_train_epochs=3,
    weight_decay=0.05,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return 'contact agent' if predicted_class_id == 0 else 'other'