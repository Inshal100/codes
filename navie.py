####LAB 4 ### NAVIE BASE:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y  # Add target labels
df.head()


# 2.1 Preprocessing Check for null values
print(df.isnull().sum())

# 3 Visualize the dataset
#3.1Using pairplot
sns.pairplot(df, hue="species", diag_kind="hist", palette="Set2")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()


# 3.2 Visualize class distribution
sns.countplot(x=df['species'])
plt.title("Class Distribution in the Iris Dataset")
plt.show()


# 4 Splitting the dataset using Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
all_conf_matrices = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 5 Train Naive Bayes Classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 6 Make Predictions
    y_pred = model.predict(X_test)

    # 7 Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    conf_matrix = confusion_matrix(y_test, y_pred)
    all_conf_matrices.append(conf_matrix)

    print("Fold Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))


# 8 Compute Overall Accuracy
print("\nOverall Accuracy (Mean of all folds):", np.mean(accuracies))


# 9. Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# 11. Visualizing the Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# 12. Create a Table Showing Correct & Incorrect Predictions
comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
comparison_df["Correct"] = comparison_df["Actual"] == comparison_df["Predicted"]
print(comparison_df)


# Count correct & incorrect predictions per class
correct_counts = comparison_df.groupby("Actual")["Correct"].sum()
incorrect_counts = comparison_df.groupby("Actual")["Correct"].count() - correct_counts


# 13. Create a DataFrame for Visualization
summary_df = pd.DataFrame({"Correct": correct_counts, "Incorrect": incorrect_counts})
summary_df.index = iris.target_names  # Label index with class names
print("\nPrediction Summary Table:")
print(summary_df)


# 14.  Plot the Bar Graph of Correct vs Incorrect Predictions
summary_df.plot(kind="bar", stacked=True, color=["green", "red"], figsize=(8, 5))
plt.title("Correct vs. Incorrect Predictions by Class")
plt.xlabel("Iris Species")
plt.ylabel("Count")
plt.legend(["Correct", "Incorrect"])
plt.xticks(rotation=0)
plt.show()
