import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =========================================================================
# Data Preparation & Split
# =========================================================================
data = fetch_california_housing()
X = data.data
y = (data.target > np.median(data.target)).astype(int)   # 0 = cheap, 1 = expensive
class_names = np.array(["cheap", "expensive"])

# Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train / Validation (for early stopping)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# NOTE: no StandardScaler here on purpose - tree-based models are invariant
# to monotonic feature scaling, scaling would be a no-op for them.

# =========================================================================
# XGBoost
# =========================================================================
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    early_stopping_rounds=20,
    random_state=42
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_preds = xgb.predict(X_test)

# =========================================================================
# CatBoost
# =========================================================================
cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=False
)

cat.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=20
)

cat_preds = cat.predict(X_test)

# =========================================================================
# Results
# =========================================================================
print("\n" + "=" * 40)
print("FINAL RESULTS - California Housing")
print("=" * 40)

print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"CatBoost Accuracy: {accuracy_score(y_test, cat_preds):.4f}")

print("\nXGBoost Report")
print(classification_report(y_test, xgb_preds, target_names=class_names))

print("\nCatBoost Report")
print(classification_report(y_test, cat_preds, target_names=class_names))
