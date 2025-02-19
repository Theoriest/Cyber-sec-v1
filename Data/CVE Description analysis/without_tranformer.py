import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the data
df = pd.read_csv("/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/cleaned_cve_data.csv")

# Handle missing values in other columns if necessary
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

# save the updated data to a new CSV file
save_path = "/home/lee/Documents/GitHub/Cyber-sec-v1/Data/CSV Files/updated_cve_data_!t.csv"
df.to_csv(save_path, index=False)
print(f"Updated data saved to {save_path}")

#Train the final model
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
joblib.dump(model, 'sakura.pkl')
print("Final model saved to sakura.pkl")

# Save evaluation results to a file
with open("model_evaluation_results.txt", "a") as f:
    f.write("Model: sakura\n")
    f.write(f"Score: {score}\n\n")