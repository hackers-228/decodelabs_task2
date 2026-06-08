# ==========================================================
# DECODELABS PROJECT 2
# DATA CLASSIFICATION USING AI
# Dataset: Iris Dataset
# Algorithm: K-Nearest Neighbors (KNN)
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)

# ==========================================================
# STEP 1: LOAD DATASET
# ==========================================================

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)

print("="*60)
print("IRIS DATASET OVERVIEW")
print("="*60)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Labels:")
print(iris.target_names)

# ==========================================================
# STEP 2: TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================================
# STEP 3: FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# STEP 4: BUILD KNN MODEL
# ==========================================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

# ==========================================================
# STEP 5: PREDICTIONS
# ==========================================================

y_pred = knn.predict(X_test)

# ==========================================================
# STEP 6: MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"\nAccuracy Score : {accuracy:.4f}")
print(f"F1 Score       : {f1:.4f}")

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# ==========================================================
# STEP 7: CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("Confusion Matrix - Iris Classification")
plt.show()

# ==========================================================
# STEP 8: SAMPLE PREDICTION
# ==========================================================

sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

prediction = knn.predict(sample_scaled)

print("\nSample Flower Prediction:")
print("Predicted Class:",
      iris.target_names[prediction[0]])

print("\nProject Completed Successfully.")