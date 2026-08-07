import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =========================================================================
# Data Preparation & Split
# =========================================================================
data = fetch_openml(name="adult", version=2, as_frame=True)

X = data.data
y = (data.target == ">50K").astype(int)

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Train / Test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train / Validation
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.15, random_state=42, stratify=y_train
)


# -------------------------------------------------------------------------
# Preprocessing: impute only, NO one-hot / ordinal encoding.
#
# Unlike the MLP/TabMixer tests, both XGBoost and CatBoost can consume
# categorical columns natively and do it better than an arbitrary ordinal
# code would - so we keep categories as categories and let each library
# handle them the way it's designed to:
#   - CatBoost: pass raw string columns + cat_features=[...]
#   - XGBoost:  pass pandas 'category' dtype columns + enable_categorical=True
# -------------------------------------------------------------------------
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")


def impute(X_raw, fit=False):
    num_vals = (num_imputer.fit_transform if fit else num_imputer.transform)(X_raw[num_cols])
    cat_vals = (cat_imputer.fit_transform if fit else cat_imputer.transform)(X_raw[cat_cols])

    num_df = pd.DataFrame(num_vals, columns=num_cols, index=X_raw.index)
    cat_df = pd.DataFrame(cat_vals, columns=cat_cols, index=X_raw.index).astype(str)

    return pd.concat([num_df, cat_df], axis=1)


X_train = impute(X_train_raw, fit=True)
X_val = impute(X_val_raw)
X_test = impute(X_test_raw)

# =========================================================================
# XGBoost (native categorical support via pandas 'category' dtype)
# =========================================================================
X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()
X_test_xgb = X_test.copy()
for df in (X_train_xgb, X_val_xgb, X_test_xgb):
    df[cat_cols] = df[cat_cols].astype("category")

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=20,
    random_state=42
)

xgb.fit(
    X_train_xgb, y_train,
    eval_set=[(X_val_xgb, y_val)],
    verbose=False
)

xgb_preds = xgb.predict(X_test_xgb)

# =========================================================================
# CatBoost (native categorical support via cat_features)
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
    cat_features=list(cat_cols),
    use_best_model=True,
    early_stopping_rounds=20
)

cat_preds = cat.predict(X_test)

# =========================================================================
# Results
# =========================================================================
target_names = ["<=50K", ">50K"]

print("\n" + "=" * 40)
print("FINAL RESULTS - Adult Income")
print("=" * 40)

print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"CatBoost Accuracy: {accuracy_score(y_test, cat_preds):.4f}")

print("\nXGBoost Report")
print(classification_report(y_test, xgb_preds, target_names=target_names))

print("\nCatBoost Report")
print(classification_report(y_test, cat_preds, target_names=target_names))
