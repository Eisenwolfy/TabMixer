from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.preprocessing import StandardScaler

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



rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
print()
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
