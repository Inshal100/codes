##LAB  3####
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]


# Plot class distribution
plt.figure(figsize=(8,6))
sns.countplot(x='species', data=df, palette='Set2')
plt.title('Class Distribution of Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Pairplot to visualize all features together by species
sns.pairplot(df, hue='species', palette='Set2', height=2.5)
plt.suptitle("Pairplot of Iris Dataset Features by Species", y=1.02)
plt.show()

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]


print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")


# Train the Decision Tree Classifier with Gini index
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)

# Train the Decision Tree Classifier with Information Gain (Entropy)
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)

# Make predictions
y_pred_gini = dt_gini.predict(X_test)
y_pred_entropy = dt_entropy.predict(X_test)

# Evaluate the models
accuracy_gini = accuracy_score(y_test, y_pred_gini)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

print(f"Accuracy of Decision Tree with Gini Index: {accuracy_gini * 100:.2f}%")
print(f"Accuracy of Decision Tree with Information Gain (Entropy): {accuracy_entropy * 100:.2f}%")

# Prepare data for visualization
methods = ['Gini Index', 'Information Gain (Entropy)']
accuracies = [accuracy_gini, accuracy_entropy]

# Create a bar chart to visualize the comparison
plt.figure(figsize=(8, 6))
plt.bar(methods, accuracies, color=['blue', 'green'])
plt.title('Comparison of Decision Tree Classifiers')
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


from sklearn.tree import export_graphviz
import graphviz


# Export the decision tree trained with Gini index to a .png image
dot_data_gini = export_graphviz(dt_gini, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
graph_gini = graphviz.Source(dot_data_gini)
graph_gini.render("decision_tree_gini", format="png")

# Export the decision tree trained with Information Gain (Entropy) to a .png image
dot_data_entropy = export_graphviz(dt_entropy, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
graph_entropy = graphviz.Source(dot_data_entropy)
graph_entropy.render("decision_tree_entropy", format="png")



from sklearn.model_selection import GridSearchCV

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV with the Decision Tree Classifier using Gini index
grid_search_gini = GridSearchCV(DecisionTreeClassifier(criterion='gini', random_state=42), param_grid, cv=5)
grid_search_gini.fit(X_train, y_train)

# Get the best parameters and score
best_params_gini = grid_search_gini.best_params_
best_score_gini = grid_search_gini.best_score_

# Perform GridSearchCV with the Decision Tree Classifier using Information Gain (Entropy)
grid_search_entropy = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=42), param_grid, cv=5)
grid_search_entropy.fit(X_train, y_train)

# Get the best parameters and score
best_params_entropy = grid_search_entropy.best_params_
best_score_entropy = grid_search_entropy.best_score_

print(f"Best parameters for Gini Index: {best_params_gini}")
print(f"Best cross-validation score for Gini Index: {best_score_gini * 100:.2f}%")
print(f"Best parameters for Information Gain (Entropy): {best_params_entropy}")
print(f"Best cross-validation score for Information Gain (Entropy): {best_score_entropy * 100:.2f}%")

import matplotlib.pyplot as plt

# Prepare the data for plotting
methods = ['Gini Index', 'Information Gain (Entropy)']
accuracies = [accuracy_gini, accuracy_entropy]

# Plot the comparison
plt.figure(figsize=(8, 6))
plt.bar(methods, accuracies, color=['blue', 'green'])
plt.title('Comparison of Decision Tree Classifiers')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


# Plot histograms for each feature

# Sepal Length Histogram
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='sepal length (cm)', hue='species', kde=True, bins=20)
plt.title('Comparison of Sepal Length by Species')
plt.show()

# Sepal Width Histogram
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='sepal width (cm)', hue='species', kde=True, bins=20)
plt.title('Comparison of Sepal Width by Species')
plt.show()

# Petal Length Histogram
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='petal length (cm)', hue='species', kde=True, bins=20)
plt.title('Comparison of Petal Length by Species')
plt.show()

# Petal Width Histogram
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='petal width (cm)', hue='species', kde=True, bins=20)
plt.title('Comparison of Petal Width by Species')
plt.show()
