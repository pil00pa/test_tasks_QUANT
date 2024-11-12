import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

# Example sentences for testing NER model
bad_sentence = "Mount Fuji is Japan's iconic and highest mountain, known for its symmetrical beauty and cultural significance."
good_sentence = "Softly, softly crawl, snail on the slope of Mount Fuji, up to the heights."
sentences = [bad_sentence.split(), good_sentence.split()]

# Load the trained model and tokenizer from saved path
model_path = "./trained_distilbert_ner"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForTokenClassification.from_pretrained(model_path)

# Tokenize input sentences
# is_split_into_words=True because we're passing pre-tokenized sentences
inputs = tokenizer(sentences, 
                  padding=True,        # Add padding to make all sequences same length
                  truncation=True,     # Truncate sequences that are too long
                  is_split_into_words=True, 
                  return_tensors="pt")  # Return PyTorch tensors

# Get model predictions
# torch.no_grad() disables gradient calculation for inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the most likely label for each token
predictions = torch.argmax(outputs.logits, dim=2)

# Get the label mapping dictionary from model config
# This maps numerical label IDs back to string labels
label_map = model.config.id2label

# Convert predicted label indices back to actual labels
predicted_labels = []
for i, sentence in enumerate(sentences):
    # Get word IDs to handle subword tokenization
    word_ids = inputs.word_ids(batch_index=i)
    predicted_labels_sentence = []
    
    # Process predictions for each token
    for word_id, label_id in zip(word_ids, predictions[i]):
        if word_id is not None:  # Skip special tokens (CLS, SEP, PAD)
            label = label_map[label_id.item()]
            # Only keep the first subword token's prediction for each word
            if len(predicted_labels_sentence) <= word_id:
                predicted_labels_sentence.append(label)
    
    predicted_labels.append(predicted_labels_sentence)

# Define mapping from model labels to entity types
label_to_entity = {
    'LABEL_0': "Mountain",  # First label type
    'LABEL_1': "Mountain",  # Second label type
    'LABEL_2': ""           # No entity
}

# Display results
for i, (sentence, pred_labels) in enumerate(zip(sentences, predicted_labels)):
    print(f"Sentence {i+1}:")
    for word, pred_label in zip(sentence, pred_labels):
        print(f"{word} -- {label_to_entity[pred_label]}")
    print("\n")