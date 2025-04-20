####lab 6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

#2 Loading the dataset
iris=load_iris()
X=iris.data
y=iris.target
df=pd.DataFrame(X, columns=iris.feature_names)
df['species']=y #Adding target labels
df.head()

# 3 Visulize the dataset
# 3.1 Visulize the class distribution
target_names = ['setosa', 'versicolor', 'virginica']

sns.countplot(x=df['species'], palette='Set2')
plt.title('Class Distribution', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Visulize the dataset
sns.pairplot(df, hue="species", diag_kind="hist",palette="Set2")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()


#5 Split the datset using Stratified Sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#6 Data Preprocesing -Standdardilaztion
scaler = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


#6 PCA for visulizationg class noundries
pca = PCA(n_components=2) #Reduces 4D feature space to 2D for visualizing decision boundaries.
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled )


# 7 Train random forest(Bagging)
rf=RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred=rf.predict(X_test)


#8  Train Adaboost(Boosting)
ada=AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
ada_pred=ada.predict(X_test)



#9 Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Adaboost Accuracy:", accuracy_score(y_test, ada_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("\nAdaboost Classification Report:\n", classification_report(y_test, ada_pred))


#9 Confusion Matrix heatmap
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, ada_pred), annot=True, fmt='d', cmap="Oranges")
plt.title("AdaBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#10  Function to plot decision boundary
def plot_boundary(model, X, y, title):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict for all points in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')

    # Plot actual points
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=[iris.target_names[i] for i in y],
                    palette='Set1')

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.show()

# Train models on PCA data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_pca, y_train)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X_train_pca, y_train)

# Plot boundaries
plot_boundary(rf_model, X_train_pca, y_train, "Random Forest Decision Boundary")
plot_boundary(ada_model, X_train_pca, y_train, "AdaBoost Decision Boundary")


# Count correct and incorrect predictions for each model
rf_correct = (rf_pred == y_test).sum()
rf_incorrect = (rf_pred != y_test).sum()

ada_correct = (ada_pred == y_test).sum()
ada_incorrect = (ada_pred != y_test).sum()

# Prepare data for bar plot
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'AdaBoost', 'AdaBoost'],
    'Prediction': ['Correct', 'Incorrect', 'Correct', 'Incorrect'],
    'Count': [rf_correct, rf_incorrect, ada_correct, ada_incorrect]
})

# Plot the bar graph
plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x='Model', y='Count', hue='Prediction', palette='pastel')
plt.title("Correct vs Incorrect Predictions")
plt.ylabel("Number of Predictions")
plt.show()
