import platform
import warnings

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datasets import Dataset
import numpy as np
import evaluate
from transformers import logging

warnings.filterwarnings('ignore')


# Load the preprocessed dataset
df = pd.read_csv('preprocessed_goemotions.csv')

# Ask user for output directory for the model
output_dir = input('\nPlease specify an output directory for the model: ')

# Setup training arguments
training_args = TrainingArguments(output_dir=output_dir)

# Specify the metric for evaluation
metric = evaluate.load("accuracy")

# Define compute metrics function to evaluate predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Encode labels using LabelEncoder
le = LabelEncoder()
df.labels = le.fit_transform(df.labels)

# Ensure text column is of type unicode
df.text = df.text.astype('U')

# Convert dataframe to a Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text', 'labels']].reset_index(drop=True))

# Encode labels in the dataset
dataset = dataset.class_encode_column("labels")

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# Define tokenisation function for padding and truncation
print("Applying tokenisation function to the dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets
split = tokenized_datasets.train_test_split(test_size=0.2, stratify_by_column='labels')
train_dataset = split['train']
val_dataset = split['test']

# Initialize model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=14)

# Check for CUDA availability and set the device accordingly
if platform.system() == 'Darwin':
    # Check if the current system is a MacBook
    device = torch.device('mps')
elif torch.cuda.is_available():
    # Use CUDA if not on a MacBook and CUDA is available
    device = torch.device('cuda')
else:
    # Fallback to CPU if CUDA (and therefore MPS) is not available
    device = torch.device('cpu')

print("Device set to:", device)

# Move model to the device
model = model.to(device)

# Initialize the trainer with the model, training arguments, training and validation datasets, and evaluation function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
print("")
trainer.train()

# Save the trained model
trainer.save_model()