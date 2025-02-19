import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

# Load the data
df = pd.read_csv("/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/cleaned_cve_data.csv")

# Handle missing values
df.ffill(inplace=True)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define a function to classify descriptions
def classify_cia(description):
    inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    if predicted_class == 0:
        return 'Confidentiality'
    elif predicted_class == 1:
        return 'Integrity'
    elif predicted_class == 2:
        return 'Availability'
    else:
        return 'Other'

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
save_path = "/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/updated_cve_data_with_tf.csv"
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
joblib.dump(model, 'Susano_1.pkl')
print("Final model saved to Susano_1.pkl")

# Save evaluation results to a file
with open("model_evaluation_results.txt", "a") as f:
    f.write("Model: Susano\n")
    f.write(f"Score: {score}\n\n")
