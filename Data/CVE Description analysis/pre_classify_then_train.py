import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset

# Load the data
df = pd.read_csv("/workspace/Cyber-sec-v1/Data/CSV Files/cve_data.csv")

# Handle missing values
df.ffill(inplace=True)

# Create a labeled dataset (example labels, you need to provide actual labels)
df['CIA_Class'] = df['Description'].apply(lambda x: 'Confidentiality' if 'confidentiality' in x.lower() else ('Integrity' if 'integrity' in x.lower() else ('Availability' if 'availability' in x.lower() else 'Other')))

# Encode labels
label_map = {'Confidentiality': 0, 'Integrity': 1, 'Availability': 2, 'Other': 3}
df['label'] = df['CIA_Class'].map(label_map)

# Split data into training and testing sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Description'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

# Define a custom dataset class
class CveDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Create datasets
train_dataset = CveDataset(train_texts, train_labels, tokenizer)
val_dataset = CveDataset(val_texts, val_labels, tokenizer)

# Use DataCollatorWithPadding to handle padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Use the fine-tuned model for predictions
fine_tuned_model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')
fine_tuned_tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model')

def classify_cia(description):
    inputs = fine_tuned_tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = fine_tuned_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return list(label_map.keys())[list(label_map.values()).index(predicted_class)]

# Apply the classification function to the Description column
df['CIA_Class'] = df['Description'].apply(classify_cia)

# Save the updated DataFrame to CSV file
df.to_csv("/workspace/Cyber-sec-v1/Data/CSV Files/cve_data.csv/CVSS_data_pre-trained.csv", mode='w', header=True, index=False)
print("Data updated and saved to cve_data_pre-trained_updated.csv")