#LAB 5 ######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Dataset shape:", df.shape)

#  Check for null values
print("\nChecking for null values in dataset:")
null_counts = df.isnull().sum()
for col, count in null_counts.items():
    print(f"{col}: {count} null values")

# Class distribution
print("\nClass distribution:\n", df['target'].value_counts())

# Class distribution plot
sns.countplot(x=df["target"], palette="Set2")
plt.xticks(ticks=[0, 1], labels=["Malignant", "Benign"])
plt.title("Class Distribution in Breast Cancer Dataset")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# Pairplot of selected features and target
selected_features = df.columns[:5].tolist() + ['target']  # Choose a few for readability
sns.pairplot(df[selected_features], hue='target', palette='Set2')
plt.suptitle("Pairplot of Selected Features (with Target)", y=1.02)
plt.show()


# Split data
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalize features (for chi2)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Variance Threshold
var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X_scaled)

# Chi-Square Feature Selection
k = 10  # Select top 10 features
chi2_selector = SelectKBest(score_func=chi2, k=k)
X_selected = chi2_selector.fit_transform(X_var, y)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies, cv_precisions, cv_recalls, cv_f1s = [], [], [], []

print("\nPerforming Stratified K-Fold Cross Validation:")
for i, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
    X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_fold, y_train_fold)
    y_val_pred = knn.predict(X_val_fold)

    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
    print(f"Fold {i} Accuracy: {fold_accuracy:.2f}")

    cv_accuracies.append(fold_accuracy)
    cv_precisions.append(precision_score(y_val_fold, y_val_pred))
    cv_recalls.append(recall_score(y_val_fold, y_val_pred))
    cv_f1s.append(f1_score(y_val_fold, y_val_pred))

# Cross-validation Results
print("\nStratified K-Fold Cross Validation Results (Chi-Square + VarianceThreshold):")
print(f"Avg Accuracy: {np.mean(cv_accuracies):.2f}")
print(f"Avg Precision: {np.mean(cv_precisions):.2f}")
print(f"Avg Recall: {np.mean(cv_recalls):.2f}")
print(f"Avg F1 Score: {np.mean(cv_f1s):.2f}")

# Test set evaluation with same feature selection pipeline
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_var = var_thresh.transform(X_train_scaled)
X_test_var = var_thresh.transform(X_test_scaled)

X_train_selected = chi2_selector.transform(X_train_var)
X_test_selected = chi2_selector.transform(X_test_var)

knn_final = KNeighborsClassifier(n_neighbors=5)
knn_final.fit(X_train_selected, y_train)
y_pred = knn_final.predict(X_test_selected)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, knn_final.predict_proba(X_test_selected)[:, 1])
cm = confusion_matrix(y_test, y_pred)

print("\nFinal Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, knn_final.predict_proba(X_test_selected)[:, 1])
plt.plot(fpr, tpr, label="kNN (AUC = {:.2f})".format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Table of predictions
comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
comparison_df["Correct"] = comparison_df["Actual"] == comparison_df["Predicted"]
print("\nPrediction Comparison Table:")
print(comparison_df)

# Summary Plot: Correct vs Incorrect
correct_counts = comparison_df["Correct"].sum()
incorrect_counts = len(comparison_df) - correct_counts

summary_df = pd.DataFrame({"Correct": [correct_counts], "Incorrect": [incorrect_counts]})
summary_df.index = ["Total Predictions"]
summary_df.plot(kind="bar", stacked=True, color=["green", "red"], figsize=(6, 4))
plt.title("Correct vs. Incorrect Predictions")
plt.ylabel("Count")
plt.legend(["Correct", "Incorrect"])
plt.xticks(rotation=0)
plt.show()
