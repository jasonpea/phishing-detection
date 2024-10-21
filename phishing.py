import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from collections import Counter
import math

# Load dataset
df = pd.read_csv('/Users/jason/dev/Hack49/dataset_phishing.csv')

# Extract labels
df['label'] = df['status'].map({'phishing': 1, 'legitimate': 0})

# Function to check if URL contains an IP address
def has_ip_address(url):
    return 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0

# Function to count special characters in the URL
def count_special_chars(url):
    return len(re.findall(r'[@_!#$%^&*()<>?/\|}{~:]', url))

# Function to extract the top-level domain (TLD)
def get_tld(url):
    return urlparse(url).netloc.split('.')[-1]

def calculate_entropy(string):
    probabilities = [n_x/len(string) for x, n_x in Counter(string).items()]
    entropy = -sum([p * math.log2(p) for p in probabilities])
    return entropy

# Updated function to extract features from a URL
def extract_features(url):
    features = []
    features.append(len(url))  
    features.append(url.count('.'))  
    features.append(len(url.split('.')) - 2)  
    features.append(url.count('/') - 2)  
    features.append(1 if 'https' in url else 0)  
    features.append(1 if '@' in url else 0)  
    features.append(1 if '-' in url else 0)  
    features.append(url.count('?'))  
    features.append(len(re.findall(r'[0-9]', url))) 
    features.append(1 if "login" in url else 0)  
    features.append(len(re.findall(r'[A-Z]', url)))  
    features.append(len(re.findall(r'[a-z]', url)))  
    features.append(url.count('%'))  
    features.append(len(re.findall(r'[_]', url)))  
    features.append(len(re.findall(r'~', url)))  
    features.append(url.count('-'))  
    features.append(len(url.split('&')) - 1)  
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)  
    features.append(len(re.findall(r'[!#$%^&*]', url)))  
    features.append(1 if "secure" in url else 0)  
    features.append(1 if "confirm" in url else 0) 
    features.append(len(urlparse(url).netloc))  
    features.append(has_ip_address(url))  
    features.append(count_special_chars(url))  
    features.append(len(urlparse(url).path))  
    features.append(len(re.findall(r'[a-zA-Z]', urlparse(url).path)))  
    
    # One-hot encode the TLD
    tld = get_tld(url)

    return features

# Apply feature extraction to the URLs in the dataset
features_list = [extract_features(url) for url in df['url']]
features_secondary_df = pd.DataFrame(features_list, columns=[f'feature_{i}' for i in range(len(features_list[0]))])


X_manual = features_secondary_df
y = df['label']

#Extract TF-IDF features from URLs
tfidf_vectorizer = TfidfVectorizer(analyzer = 'char', ngram_range = (2, 4), max_features = 400)
X_tfidf = tfidf_vectorizer.fit_transform(df['url']).toarray()

# Combine TF-IDF features with manually extracted features
X_combined = np.hstack((X_manual, X_tfidf))  # Stack them horizontally to create the combined feature set

# Split the combined feature set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size = 0.3, random_state = 42)

# HP tuning with RandomSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [10, 20, None],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'bootstrap': [True, False]  
}

rf = RandomForestClassifier(random_state = 42)
randomized_search = RandomizedSearchCV(estimator = rf, param_distributions = param_dist, cv = 3, n_jobs =- 1, verbose = 0, n_iter = 10)  # n_iter=10 limits to 10 random searches
randomized_search.fit(X_train, y_train)

# Get the best model
best_model = randomized_search.best_estimator_

# Make predictions and evaluate the model
y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nModel Accuracy: {accuracy:.2f}")

# Save the trained model using pickle
with open('phishing_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# testing
new_url = "	https://compound.finance.qomshora.ir"  
new_url_features = extract_features(new_url)

# Apply the same TF-IDF transformation to the new URL
new_url_tfidf = tfidf_vectorizer.transform([new_url]).toarray()

# Combine the manually extracted features with the TF-IDF features for the new URL
new_url_combined = np.hstack((np.array(new_url_features).reshape(1, -1), new_url_tfidf))

# Make a prediction with the model
try:
    prediction = best_model.predict(new_url_combined)
    if prediction[0] == 1:
        print(f"Prediction for '{new_url}': Phishing")
    else:
        print(f"Prediction for '{new_url}': Legitimate")
except ValueError as e:
    print(f"Error during prediction: {e}")

# Evaluate model performance on the test set
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))



#Feature importance and visualization
# importances = best_model.feature_importances_
# indices = np.argsort(importances)[::-1]

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices], align="center")
# plt.xticks(range(X.shape[1]), [f'feature_{i}' for i in indices], rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.show()
