from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================================================================
# Data Preparation & Split
# =========================================================================
data = fetch_openml(name="adult", version=2, as_frame=True)
X, y = data.data, data.target
y_class = (y == '>50K').astype(int)

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)

# =========================================================================
#KNN
# =========================================================================
knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
print(f"KNN Accuracy: {accuracy_score(y_test, knn.predict(X_test)) * 100:.2f}%")


# =========================================================================
#RF
# =========================================================================
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")
