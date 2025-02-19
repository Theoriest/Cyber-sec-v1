import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset
import joblib

# Load the data
df = pd.read_csv("/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/cleaned_cve_data.csv")

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

# Separate rows with missing CVSS_Score
missing_cvss = df[df['CVSS_Score'].isna()]
available_cvss = df[df['CVSS_Score'].notna()]

# Split data into features and target for available CVSS_Score
features = available_cvss.drop(columns=['CVSS_Score'])
target = available_cvss['CVSS_Score']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Textual feature extraction
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=1000))
])

# Categorical feature encoding
categorical_features = ['CIA_Class']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'Description'),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline with preprocessor and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# Predict missing CVSS_Score values
missing_features = missing_cvss.drop(columns=['CVSS_Score'])
predicted_cvss = model.predict(missing_features)

# Fill the missing CVSS_Score values with the predictions
missing_cvss['CVSS_Score'] = predicted_cvss

# Combine the dataframes
df = pd.concat([available_cvss, missing_cvss])

# Save the updated data to a new CSV file
save_path = "/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/updated_cve_data_pre_classify_then_train.csv"
df.to_csv(save_path, index=False)
print(f"Updated data saved to {save_path}")

# Train the final model
# Split data into features and target
features = df.drop(columns=['CVSS_Score'])
target = df['CVSS_Score']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fit the final model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model score: {score}")

# Save the final trained model to a file
joblib.dump(model, 'sage.pkl')
print("Final model saved to Pre_Classify_Then_Train.pkl")

# Save evaluation results to a file
with open("model_evaluation_results.txt", "a") as f:
    f.write("Model: Pre_Classify_Then_Train\n")
    f.write(f"Score: {score}\n\n")