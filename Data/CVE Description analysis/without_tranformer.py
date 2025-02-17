import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv("/workspace/Cyber-sec-v1/Data/CSV Files/cve_data.csv")

# Handle missing values
df.ffill(inplace=True)

# Classify vulnerabilities according to the CIA triad
def classify_cia(description):
    if 'confidentiality' in description.lower():
        return 'Confidentiality'
    elif 'integrity' in description.lower():
        return 'Integrity'
    elif 'availability' in description.lower():
        return 'Availability'
    else:
        return 'Other'

df['CIA_Class'] = df['Description'].apply(classify_cia)

# Split data into features and target
features = df.drop(columns=['CVSS_Score'])
target = df['CVSS_Score']

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

# Train the model
model.fit(X_train, y_train)

# Predict missing values
df['Predicted_CVSS_Score'] = model.predict(features)

# Save the updated DataFrame to CSV file
df.to_csv("/workspace/Cyber-sec-v1/Data/CSV Files/CVSS_without_tf.csv", mode='w', header=True, index=False)
print("Data updated and saved to CVSS_without_tf.csv")