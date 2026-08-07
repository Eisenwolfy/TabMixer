import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from TabMixer import TabMixer
from numpy_MLP import MLP


# =========================================================================
# Data Preparation & Split
# =========================================================================
data = fetch_openml(name="adult", version=2, as_frame=True)
X, y = data.data, data.target
y_class = (y == '>50K').astype(int).to_numpy()

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Train-Test Split (raw, before fitting any preprocessing -> no leakage)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Train-Val Split
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.15, random_state=42, stratify=y_train
)


preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
])

X_train = preprocess.fit_transform(X_train_raw).astype(np.float32)
X_val = preprocess.transform(X_val_raw).astype(np.float32)
X_test = preprocess.transform(X_test_raw).astype(np.float32)

F = X_train.shape[1]
print(f"Number of features after preprocessing: {F}")

# NumPy MLP expects (features, samples)
X_train_np = X_train.T
X_val_np = X_val.T
X_test_np = X_test.T


# =========================================================================
# 1) Training Custom NumPy MLP
# =========================================================================
print("=" * 30)
print("TRAINING MODEL 1: Scratch NumPy MLP")
print("=" * 30)

mlp = MLP(
    input_size=F,
    hidden_sizes=[64, 32],
    output_size=2,
    epochs=100,
    learning_rate=0.001,
    batch_size=128,
    dropout_rate=0.2,
    patience=10
)

mlp.train(X_train_np, y_train, X_val_np, y_val)
mlp_preds = mlp.predict(X_test_np)


# =========================================================================
# PyTorch Common Data Setup (Expects shapes: [samples, features])
# =========================================================================
X_train_pt = torch.tensor(X_train, dtype=torch.float32)
y_train_pt = torch.tensor(y_train, dtype=torch.long)
X_val_pt = torch.tensor(X_val, dtype=torch.float32)
y_val_pt = torch.tensor(y_val, dtype=torch.long)
X_test_pt = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_pt, y_train_pt)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# =========================================================================
# 2) Built-in PyTorch MLP Architecture
# =========================================================================
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super().__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


print("=" * 30)
print("Training Built-in PyTorch MLP")
print("=" * 30)

pt_mlp = PyTorchMLP(input_size=F, hidden_sizes=[64, 32], output_size=2, dropout_rate=0.2)
optimizer_pt_mlp = optim.AdamW(pt_mlp.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_pt_mlp_loss = float('inf')
best_pt_mlp_state = None
patience_counter = 0

pt_mlp.train()
for epoch in range(100):
    epoch_loss = 0.0
    num_batches = 0
    for batch_x, batch_y in train_loader:
        optimizer_pt_mlp.zero_grad()
        out = pt_mlp(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer_pt_mlp.step()
        epoch_loss += loss.item()
        num_batches += 1

    pt_mlp.eval()
    with torch.no_grad():
        val_loss = criterion(pt_mlp(X_val_pt), y_val_pt).item()
    pt_mlp.train()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {epoch_loss/num_batches:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_pt_mlp_loss:
        best_pt_mlp_loss = val_loss
        patience_counter = 0
        best_pt_mlp_state = {k: v.clone() for k, v in pt_mlp.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f"\nEarly stopping PyTorch MLP on epoch {epoch}")
            break

if best_pt_mlp_state is not None:
    pt_mlp.load_state_dict(best_pt_mlp_state)

pt_mlp.eval()
with torch.no_grad():
    pt_mlp_preds = torch.argmax(pt_mlp(X_test_pt), dim=1).numpy()


# =========================================================================
# 3) Training PyTorch TabMixer
# =========================================================================
print("=" * 30)
print("Training PyTorch TabMixer")
print("=" * 30)

tm_model = TabMixer(F=F, D=32, hidden_dim=64, n_blocks=3, n_classes=2, pooling="mean")
optimizer_tm = optim.AdamW(tm_model.parameters(), lr=0.001, weight_decay=1e-4)

best_tm_loss = float('inf')
best_tm_state = None
patience_counter = 0

tm_model.train()
for epoch in range(100):
    epoch_loss = 0.0
    num_batches = 0
    for batch_x, batch_y in train_loader:
        optimizer_tm.zero_grad()
        out = tm_model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer_tm.step()
        epoch_loss += loss.item()
        num_batches += 1

    tm_model.eval()
    with torch.no_grad():
        val_loss = criterion(tm_model(X_val_pt), y_val_pt).item()
    tm_model.train()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {epoch_loss/num_batches:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_tm_loss:
        best_tm_loss = val_loss
        patience_counter = 0
        best_tm_state = {k: v.clone() for k, v in tm_model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print(f"\nEarly stopping TabMixer on epoch {epoch}")
            break

if best_tm_state is not None:
    tm_model.load_state_dict(best_tm_state)

tm_model.eval()
with torch.no_grad():
    tm_preds = torch.argmax(tm_model(X_test_pt), dim=1).numpy()


# =========================================================================
# 4) FINAL COMPARISON REPORT
# =========================================================================
target_names = ["<=50K", ">50K"]

print("\n" + "#" * 40)
print("Final test & result comparison")
print("#" * 40)

print(f"NumPy MLP Accuracy: {accuracy_score(y_test, mlp_preds):.4f}")
print(f"Built-in PyTorch MLP: {accuracy_score(y_test, pt_mlp_preds):.4f}")
print(f"TabMixer Accuracy: {accuracy_score(y_test, tm_preds):.4f}\n")

print("Built-in PyTorch MLP Classification Report")
print(classification_report(y_test, pt_mlp_preds, target_names=target_names))

print("TabMixer Classification Report")
print(classification_report(y_test, tm_preds, target_names=target_names))

# Plot losses from the NumPy implementation
mlp.visualizing_loss()
