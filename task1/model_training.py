##### DATA PREPARATION #####

import re
import json

# Load the raw dataset from JSON file

f = open('raw_dataset.json', 'r', encoding='utf-8')
data = json.load(f)

def convert_data_to_bert_format(data):
    """
    Convert raw NER data into BERT-compatible BIO format.
    
    BIO Format:
    - B-TAG: Beginning of an entity
    - I-TAG: Inside/continuation of an entity
    - O: Outside any entity
    
    Args:
        data (dict): Raw data containing sentences and their entity labels
        
    Returns:
        list: List of dictionaries containing tokenized sentences and their BIO labels
    """
    converted_data = []
    
    # Process each sentence in the dataset
    for sentence_data in data['sentences']:
        # Tokenize the sentence into words and punctuation
        words = re.findall(r'\w+|[^\w\s]', sentence_data['text'])
        
        # Initialize all words with 'O' (Outside) label
        word_labels = ["O"] * len(words)
        
        # Process each entity and its corresponding label in the sentence
        for entity, entity_name in sentence_data['labels'].items():
            # Handle both string and list entity names
            # Some datasets might have entity names as strings, others as lists
            if isinstance(entity_name, str):
                entity_words = entity_name.split()
            else:
                entity_words = entity_name
                
            # Find and label all occurrences of the entity in the sentence
            for i in range(len(words) - len(entity_words) + 1):
                # Check if we found the entity at current position
                if words[i:i + len(entity_words)] == entity_words:
                    # Label the first word of entity with B- (Beginning)
                    word_labels[i] = f"B-{entity.upper()}"
                    
                    # Label subsequent words with I- (Inside)
                    for j in range(1, len(entity_words)):
                        word_labels[i + j] = f"I-{entity.upper()}"
        
        # Store the processed sentence and its labels
        converted_data.append({
            "sentence": words,          # Tokenized words
            "labels": word_labels       # Corresponding BIO labels
        })
    
    return converted_data

# Convert the data to BERT format and save it
converted_data = convert_data_to_bert_format(data)

# Save the processed data to a new JSON file
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)



##### TRAINING #####
import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from datasets import Dataset

# 1. Data Loading and Preparation
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# JSON file with training data
file_path = "data.json"
data = load_data(file_path)

def convert_to_dataset(data):
    """
    Convert raw JSON data into format suitable for training.
    
    Args:
        data (list): List of dictionaries containing sentences and their NER labels
        
    Returns:
        tuple: (tokenized_data, labels) where:
            - tokenized_data: List of sentences
            - labels: List of corresponding NER tags
    """
    tokenized_data = []
    labels = []
    for item in data:
        tokenized_data.append(item["sentence"])
        labels.append(item["labels"])
    return tokenized_data, labels

# Convert data into required format
sentences, ner_tags = convert_to_dataset(data)

# Create label mapping for converting string labels to integers
label_list = sorted(set(tag for tags in ner_tags for tag in tags))
label_map = {label: i for i, label in enumerate(label_list)}

def align_labels_with_tokens(tokenizer, sentence, labels):
    """
    Align NER labels with tokenized input, handling subword tokenization.
    
    Args:
        tokenizer: Hugging Face tokenizer
        sentence (list): List of input sentences
        labels (list): List of NER labels for each sentence
        
    Returns:
        dict: Tokenized inputs with aligned labels
    """
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, 
                               is_split_into_words=True, return_tensors="pt")
    labels_enc = []
    
    # Process each sentence and its labels
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        # Handle subword tokens and special tokens
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get label -100 (ignored in loss calculation)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of word gets the NER label
                label_ids.append(label_map[label[word_idx]])
            else:
                # Subsequent subword tokens: keep I- labels, ignore B- labels
                label_ids.append(label_map[label[word_idx]] 
                               if label[word_idx].startswith("I-") else -100)
            previous_word_idx = word_idx
            
        labels_enc.append(label_ids)
    
    tokenized_inputs["labels"] = torch.tensor(labels_enc)
    return tokenized_inputs

# 2. Model and Tokenizer Initialization
# Load pre-trained DistilBERT tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-cased", 
                                                        num_labels=len(label_map))

# Prepare training data
train_data = align_labels_with_tokens(tokenizer, sentences, ner_tags)

# Convert to Hugging Face Dataset format and split into train/test
dataset = Dataset.from_dict(train_data)
train_test_split = dataset.train_test_split(test_size=0.2)

# 3. Training Configuration
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory for storing training outputs
    eval_strategy="epoch",           # Evaluation strategy
    per_device_train_batch_size=32,  # Training batch size
    per_device_eval_batch_size=32,   # Evaluation batch size
    num_train_epochs=10,             # Number of training epochs
    weight_decay=0.01,               # Weight decay for regularization
)

# Initialize data collator for handling variable length sequences
data_collator = DataCollatorForTokenClassification(tokenizer)

# 4. Trainer Setup
# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,                           # The model to train
    args=training_args,                    # Training arguments
    train_dataset=train_test_split["train"], # Training data
    eval_dataset=train_test_split["test"],   # Evaluation data
    data_collator=data_collator,           # Data collator
    processing_class=tokenizer,            # Tokenizer for processing inputs
)

# 5. Model Training and Saving
# Start the training process
trainer.train()