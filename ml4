# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the CART decision tree model without pruning (to observe overfitting)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions and calculate accuracy on training and testing sets
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("Training accuracy (unpruned):", train_accuracy)
print("Testing accuracy (unpruned):", test_accuracy)

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree without Pruning")
plt.show()

# Implement pruning by setting a maximum depth to prevent overfitting
pruned_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
pruned_clf.fit(X_train, y_train)

# Make predictions and calculate accuracy on training and testing sets for pruned tree
pruned_train_pred = pruned_clf.predict(X_train)
pruned_test_pred = pruned_clf.predict(X_test)
pruned_train_accuracy = accuracy_score(y_train, pruned_train_pred)
pruned_test_accuracy = accuracy_score(y_test, pruned_test_pred)

print("Training accuracy (pruned):", pruned_train_accuracy)
print("Testing accuracy (pruned):", pruned_test_accuracy)

# Visualize the pruned decision tree
plt.figure(figsize=(15, 10))
plot_tree(pruned_clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree with Pruning (max depth=3)")
plt.show()

# Classify a new sample
new_sample = [[5.0, 3.5, 1.5, 0.2]]
prediction = pruned_clf.predict(new_sample)
print("Prediction for new sample:", data.target_names[prediction[0]])
