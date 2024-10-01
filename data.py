import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data (replace with your own dataset)
texts = ["I love this movie", "This movie is terrible", "The acting was amazing", "The plot was confusing"]
labels = [1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_vec, y_train)

# Predict on the test set
predictions = classifier.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
