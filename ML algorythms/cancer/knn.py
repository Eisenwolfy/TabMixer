import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


# Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# We are separating diabetes by median test: 0 - weak progress, 1 - strong progress
median_y = np.median(y)
y_class = np.where(y < median_y, 0, 1)

# train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["Weak Progression", "Strong Progression"]))

# Plotting heatmap
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Weak", "Strong"], yticklabels=["Weak", "Strong"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Diabetes Dataset")
plt.show()
