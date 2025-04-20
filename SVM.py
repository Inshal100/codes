import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # Add target variable


# Check for missing values
print("Missing Values in Dataset:\n", df.isnull().sum())


# Display basic dataset information
print("\nDataset Information:")
print(df.info())


# Visualize dataset before training (Class Distribution)
plt.figure(figsize=(6, 4))
sns.countplot(x=df['target'], palette=['red', 'green'])
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title("Class Distribution Before Training")
plt.show()


# Visualize dataset before training (Pair Plot)
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'target']
df_selected = df[selected_features].copy()
df_selected['target'] = df_selected['target'].map({0: 'Malignant', 1: 'Benign'})
sns.pairplot(df_selected, hue='target', palette='coolwarm', diag_kind='kde')
plt.suptitle("Pair Plot Before Training", y=1.02)
plt.show()

# Split dataset using Stratified Sampling
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Data Pre-processing: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Apply PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("\nPCA: Reduced dimensions to", X_train_pca.shape[1])

# 📊 PCA Visualization
plt.figure(figsize=(8,6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Visualization of Training Data")
plt.colorbar(label='Class')
plt.show()


# Train SVM with cross-validation to find the best accuracy
best_score = 0
best_C = None
for C in [0.1, 1, 10, 100]:
    model = SVC(C=C, kernel='rbf', probability=True, random_state=42)
    scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    print(f"C={C}: Cross-Validation Accuracy = {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_C = C

print(f"\nBest SVM Model Chosen: C={best_C} with Accuracy = {best_score:.4f}")

# Train the final SVM model
svm = SVC(C=best_C, kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_pca, y_train)

# Make predictions
y_pred = svm.predict(X_test_pca)
y_prob = svm.predict_proba(X_test_pca)[:, 1] #because we need the probability of the positive class fOR roc curve


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print(f"ROC-AUC Score: {roc_auc:.4f}")


# Visualization: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix After Training")
plt.show()


# 📊 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# 📊 Visualizing PCA after Training
plt.figure(figsize=(8,6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='coolwarm', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - SVM Classification Results")
plt.colorbar(label='Predicted Class')
plt.show()

# Select top 20 features based on absolute correlation with the target variable
top_20_features = df.corr().abs()['target'].sort_values(ascending=False)[1:21].index  # Exclude 'target' itself

# Plot correlation heatmap for selected 20 features
plt.figure(figsize=(12,8))
sns.heatmap(df[top_20_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Top 20 Feature Correlation Heatmap")
plt.show()


# Box Plot for Selected Features
plt.figure(figsize=(12,6))
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
df_melted = df.melt(id_vars=['target'], value_vars=selected_features)

sns.boxplot(x='variable', y='value', hue='target', data=df_melted, palette='coolwarm')
plt.title("Box Plot of Selected Features")
plt.xticks(rotation=15)
plt.show()
