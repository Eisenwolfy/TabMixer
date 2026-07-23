import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================================================
# Dataset
# =========================================================================
data = fetch_california_housing()
X = data.data
y = (data.target > np.median(data.target)).astype(int)   # 0 = cheap, 1 = expensive
class_names = np.array(["cheap", "expensive"])

# train/test split (stratified on the real, balanced binary target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================================
# KNN
# =========================================================================
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_preds)
conf_matrix = confusion_matrix(y_test, knn_preds)

# =========================================================================
# Random Forest
# =========================================================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)

print(f"KNN Accuracy: {knn_acc * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")