import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mm import TabMixer
from standartMLP00 import MLP

# =========================================================================
# Data Preparation & Split
# =========================================================================
data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X).T

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X.T, y, test_size=0.2, random_state=42
)
X_train, X_test = X_train.T, X_test.T

# Train-Val Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train.T, y_train, test_size=0.15, random_state=42
)
X_train, X_val = X_train.T, X_val.T


# =========================================================================
# 1) Training Custom NumPy MLP
# =========================================================================
print("=" * 30)
print("TRAINING MODEL 1: Scratch NumPy MLP")
print("=" * 30)

mlp = MLP(
    input_size=30,
    hidden_sizes=[64, 32],
    output_size=2,
    epochs=500,
    learning_rate=0.001,
    batch_size=32,
    dropout_rate=0.2,
    patience=20
)

mlp.train(X_train, y_train, X_val, y_val)
mlp_preds = mlp.predict(X_test)


# =========================================================================
# PyTorch Common Data Setup (Expects shapes: [samples, features])
# =========================================================================
X_train_pt = torch.tensor(X_train.T, dtype=torch.float32)
y_train_pt = torch.tensor(y_train, dtype=torch.long)
X_val_pt = torch.tensor(X_val.T, dtype=torch.float32)
y_val_pt = torch.tensor(y_val, dtype=torch.long)
X_test_pt = torch.tensor(X_test.T, dtype=torch.float32)

train_dataset = TensorDataset(X_train_pt, y_train_pt)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


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

pt_mlp = PyTorchMLP(input_size=30, hidden_sizes=[64, 32], output_size=2, dropout_rate=0.2)
optimizer_pt_mlp = optim.AdamW(pt_mlp.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_pt_mlp_loss = float('inf')
best_pt_mlp_state = None
patience_counter = 0

pt_mlp.train()
for epoch in range(500):
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

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {epoch_loss/num_batches:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_pt_mlp_loss:
        best_pt_mlp_loss = val_loss
        patience_counter = 0
        best_pt_mlp_state = {k: v.clone() for k, v in pt_mlp.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 20:
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

tm_model = TabMixer(F=30, D=32, hidden_dim=64, n_blocks=3, n_classes=2, pooling="mean")
optimizer_tm = optim.AdamW(tm_model.parameters(), lr=0.001, weight_decay=1e-4)

best_tm_loss = float('inf')
best_tm_state = None
patience_counter = 0

tm_model.train()
for epoch in range(500):
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

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {epoch_loss/num_batches:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_tm_loss:
        best_tm_loss = val_loss
        patience_counter = 0
        best_tm_state = {k: v.clone() for k, v in tm_model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 20:
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
print("\n" + "#" * 40)
print("Final test & result comparison")
print("#" * 40)

print(f"NumPy MLP Accuracy: {accuracy_score(y_test, mlp_preds):.4f}")
print(f"Built-in PyTorch MLP: {accuracy_score(y_test, pt_mlp_preds):.4f}")
print(f"TabMixer Accuracy: {accuracy_score(y_test, tm_preds):.4f}\n")

print("Built-in PyTorch MLP Classification Report")
print(classification_report(y_test, pt_mlp_preds, target_names=data.target_names))

print("TabMixer Classification Report")
print(classification_report(y_test, tm_preds, target_names=data.target_names))


# =========================================================================
# 5) RUN AN INFERENCE EXAMPLE ON ROW 0
# =========================================================================
print("\n" + "=" * 30)
print("Example of Prediction on Row 0")
print("=" * 30)
raw_sample = data.data[0]
real_label = data.target_names[data.target[0]]
print(f"Real diagnosis: {real_label}\n")

print("-> NumPy MLP Predict:")
mlp.predict_single(raw_sample, scaler, data.target_names)

print("\n-> Built-in PyTorch MLP Predict:")
with torch.no_grad():
    sample_scaled = scaler.transform(raw_sample.reshape(1, -1))
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
    pt_mlp_probs = torch.softmax(pt_mlp(sample_tensor), dim=1).squeeze(0)
    print(f"Diagnosis: {data.target_names[torch.argmax(pt_mlp_probs).item()]}")
    print(f"Confidence: {pt_mlp_probs[torch.argmax(pt_mlp_probs).item()].item()*100:.1f}%")

print("\n-> TabMixer Predict:")
with torch.no_grad():
    tm_probs = torch.softmax(tm_model(sample_tensor), dim=1).squeeze(0)
    print(f"Diagnosis: {data.target_names[torch.argmax(tm_probs).item()]}")
    print(f"Confidence: {tm_probs[torch.argmax(tm_probs).item()].item()*100:.1f}%")

# Plot losses from your NumPy implementation
mlp.visualizing_loss()
