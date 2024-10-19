import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np

# Load the second dataset (with URLs and labels in the last column)
df2 = pd.read_csv('/Users/jason/dev/Hack49/dataset_phishing.csv')

# Extract the labels (assuming 'status' is the column name for phishing/legitimate)
df2['label'] = df2['status'].map({'phishing': 1, 'legitimate': 0})

# Function to extract features from a URL
def extract_features(url):
    features = []
    features.append(len(url))  # URL length
    features.append(url.count('.'))  # Number of dots in the URL
    features.append(len(url.split('.')) - 2)  # Number of subdomains
    features.append(url.count('/') - 2)  # Count slashes, subtracting for the protocol
    features.append(1 if 'https' in url else 0)  # Presence of HTTPS
    features.append(1 if '@' in url else 0)  # Presence of '@'
    features.append(1 if '-' in url else 0)  # Presence of '-'
    features.append(url.count('?'))  # Count of query components
    features.append(len(re.findall(r'[0-9]', url)))  # Count of numeric characters
    features.append(1 if "login" in url else 0)  # Example of a suspicious keyword
    features.append(len(re.findall(r'[A-Z]', url)))  # Count of uppercase letters
    features.append(len(re.findall(r'[a-z]', url)))  # Count of lowercase letters
    features.append(url.count('%'))  # Count of percent signs
    features.append(len(re.findall(r'[_]', url)))  # Count of underscores
    features.append(len(re.findall(r'~', url)))  # Count of tilde
    features.append(url.count('-'))  # Count of hyphens
    features.append(len(url.split('&')) - 1)  # Count of query parameters

    # Ensure the feature length is 98
    while len(features) < 98:  # Make sure we have 98 features
        features.append(0)  # Fill missing features with zero
    
    return features

# Apply feature extraction to the URLs in the dataset
features_list = [extract_features(url) for url in df2['url']]
features_secondary_df = pd.DataFrame(features_list, columns=[f'feature_{i}' for i in range(98)])  # Ensure 98 features

# Use the 'label' column as the target variable (phishing: 1, legitimate: 0)
labels_secondary_df = df2['label']

# Prepare Features and Labels
X = features_secondary_df  # Features
y = labels_secondary_df  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Save the trained model using pickle
with open('phishing_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Function to adjust new URL features to match the model's expected feature count
def adjust_new_url_features(new_url_features, expected_num_features):
    # Pad the URL features with zeros if necessary to match the model's expected input size
    while len(new_url_features) < expected_num_features:
        new_url_features.append(0)
    return new_url_features

# Example usage: classifying a new URL
new_url = "https://www.youtube.com/"  # Replace with the URL you want to test
new_url_features = extract_features(new_url)

# Ensure the new_url_features has the correct shape (98 features)
new_url_features_adjusted = adjust_new_url_features(new_url_features, 98)
new_url_features_adjusted = np.array(new_url_features_adjusted).reshape(1, -1)

# Make a prediction with the model
try:
    prediction = model.predict(new_url_features_adjusted)  # Use the reshaped array for prediction
    if prediction[0] == 1:
        print(f"Prediction for '{new_url}': Phishing")
    else:
        print(f"Prediction for '{new_url}': Legitimate")
except ValueError as e:
    print(f"Error during prediction: {e}")

# Evaluate model performance on the test set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# # Get feature importances and visualize
# importances = model.feature_importances_
# indices = np.argsort(importances)[::-1]

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices], align="center")
# plt.xticks(range(X.shape[1]), [f'feature_{i}' for i in indices], rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.show()
