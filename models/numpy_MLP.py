import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, epochs,
                 learning_rate, batch_size=32, dropout_rate=0.2, patience=20):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size # number of samples in one batch
        self.dropout_rate = dropout_rate # part of neurons which we turn off randomly
        self.patience = patience # number of epochs
        self.losses = [] # loss through epochs
        self.val_losses = [] # val through epochs
        self.training = True # True = train, False = inference

        self.weights = []
        self.biases = []

        # Adam optimizer
        self.m_W = []
        self.v_W = []
        self.m_b = []
        self.v_b = []
        self.t = 0 # steps counter for bias correction in Adam

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(W)
            self.biases.append(b)

            self.m_W.append(np.zeros_like(W))
            self.v_W.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    # -------------------------------------------------------------------------
    # Activation
    # -------------------------------------------------------------------------

    def softmax(self, Z):
        # subtract max for numerical stability
        shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(shifted)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


    def tanh(self, Z):
        # tanh(z) = (e^z - e^-z) / (e^z + e^-z)
        # used for hidden layers
        return np.tanh(Z)

    def gradient_tanh(self, Z):
        # derivative of tanh: d/dz tanh(z) = 1 - tanh(z)^2
        # used in backprop
        return 1 - np.tanh(Z) ** 2

    def sigmoid(self, x):
        # sigmoid(x) = 1 / (1 + e^-x)
        # used in output layers to get binary classification
        return 1 / (1 + np.exp(-x))

    def conversion(self, x, num_classes):
        # one-hot encoding: converting labels (integer) into matrices
        oh = np.zeros((num_classes, x.shape[0]))
        oh[x, np.arange(x.shape[0])] = 1
        return oh

    # -------------------------------------------------------------------------
    # DROPOUT
    # -------------------------------------------------------------------------

    def dropout(self, A, rate):
        if not self.training:
            return A, None
        mask = (np.random.rand(*A.shape) > rate) / (1 - rate)
        return A * mask, mask

    # -------------------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------------------

    def forward(self, X):
        self.Z_cache = [] # save for backwards
        self.A_cache = [] # analogously
        self.dropout_masks = []

        A = X
        self.A_cache.append(A)  # activation of null layer

        # hidden layers
        for i in range(len(self.weights) - 1):
            Z = self.weights[i] @ A + self.biases[i]
            A = self.tanh(Z)
            A, mask = self.dropout(A, self.dropout_rate)
            self.Z_cache.append(Z)
            self.A_cache.append(A)
            self.dropout_masks.append(mask)

        # output layer
        Z = self.weights[-1] @ A + self.biases[-1]
        A = self.softmax(Z)
        self.Z_cache.append(Z)
        self.A_cache.append(A)

        return A

    # -------------------------------------------------------------------------
    # BACKWARD PASS
    # -------------------------------------------------------------------------

    def backward(self, X, y):

        N = X.shape[1]  # number of samples in batch
        y_one_hot = self.conversion(y, self.output_size)

        gradients_W = []
        gradients_b = []

        dZ = self.A_cache[-1] - y_one_hot  # shape: (output_size, N)

        for i in reversed(range(len(self.weights))):
            A_prev = self.A_cache[i]

            dW = (dZ @ A_prev.T) / N
            db = np.sum(dZ, axis=1, keepdims=True) / N

            gradients_W.insert(0, dW)
            gradients_b.insert(0, db)

            if i > 0:
                dA_prev = self.weights[i].T @ dZ
                if self.dropout_masks[i-1] is not None:
                    dA_prev *= self.dropout_masks[i-1]
                dZ = dA_prev * self.gradient_tanh(self.Z_cache[i-1])

        return gradients_W, gradients_b

    # -------------------------------------------------------------------------
    # ADAM OPTIMIZER
    # -------------------------------------------------------------------------

    def update_parameters(self, gradients_W, gradients_b):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.t += 1

        for i in range(len(self.weights)):
            self.m_W[i] = beta1 * self.m_W[i] + (1 - beta1) * gradients_W[i]
            self.v_W[i] = beta2 * self.v_W[i] + (1 - beta2) * gradients_W[i] ** 2
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * gradients_b[i] ** 2

            m_W_corr = self.m_W[i] / (1 - beta1 ** self.t)
            v_W_corr = self.v_W[i] / (1 - beta2 ** self.t)
            m_b_corr = self.m_b[i] / (1 - beta1 ** self.t)
            v_b_corr = self.v_b[i] / (1 - beta2 ** self.t)

            self.weights[i] -= self.learning_rate * m_W_corr / (np.sqrt(v_W_corr) + eps)
            self.biases[i] -= self.learning_rate * m_b_corr / (np.sqrt(v_b_corr) + eps)

    # -------------------------------------------------------------------------
    # LOSS
    # -------------------------------------------------------------------------

    def cross_entropy_loss(self, outputs, original):
        N = outputs.shape[1]
        eps = 1e-8
        return -np.sum(original * np.log(outputs + eps)) / N


    def mse_loss(self, outputs, original):
        N = outputs.shape[1]
        return np.sum((outputs - original) ** 2) / N

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    def train(self, X_train, y_train, X_val, y_val):
        N = X_train.shape[1]
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_biases  = None

        for epoch in range(self.epochs):
            self.training = True

            indices = np.random.permutation(N)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            num_batches = 0

            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[start:end]

                # forward → loss → backward → update
                output = self.forward(X_batch)
                y_one_hot = self.conversion(y_batch, self.output_size)
                loss = self.cross_entropy_loss(output, y_one_hot)
                epoch_loss += loss
                num_batches += 1

                gradients_W, gradients_b = self.backward(X_batch, y_batch)
                self.update_parameters(gradients_W, gradients_b)

            self.losses.append(epoch_loss / num_batches)

            self.training = False
            val_output = self.forward(X_val)
            y_val_one_hot = self.conversion(y_val, self.output_size)
            val_loss = self.cross_entropy_loss(val_output, y_val_one_hot)
            self.val_losses.append(val_loss)

            if epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | Loss: {epoch_loss/num_batches:.4f} | Val Loss: {val_loss:.4f}")

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping on epoch{epoch}")
                    break
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        print("\n" + "=" * 30)
        print("End of Training")
        print("=" * 30)
        self.training = False
        train_preds = self.predict(X_train)
        val_preds   = self.predict(X_val)
        print(f"Train Accuracy: {accuracy_score(y_train, train_preds):.4f}")
        print(f"Val Accuracy:   {accuracy_score(y_val,   val_preds):.4f}")
        print("=" * 30 + "\n")

    # -------------------------------------------------------------------------
    # PREDICT
    # -------------------------------------------------------------------------

    def predict(self, X):
        self.training = False
        output = self.forward(X)
        return np.argmax(output, axis=0)  # shape: (N,)

    def predict_single(self, x_raw, scaler, class_names):

        x_scaled = scaler.transform(x_raw.reshape(1, -1)).T  # → (30, 1)

        self.training = False
        output = self.forward(x_scaled)  # → (2, 1)

        predicted_class = np.argmax(output, axis=0)[0]
        confidence = output[predicted_class, 0] * 100

        print(f"Diagnosis: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Probability: benign={output[1,0]*100:.1f}%  malignant={output[0,0]*100:.1f}%")

        return class_names[predicted_class]

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def visualizing_loss(self):
        plt.plot(self.losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.show()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X).T

X_train, X_test, y_train, y_test = train_test_split(
    X.T, y, test_size=0.2, random_state=42
)
X_train, X_test = X_train.T, X_test.T

X_train, X_val, y_train, y_val = train_test_split(
    X_train.T, y_train, test_size=0.15, random_state=42
)
X_train, X_val = X_train.T, X_val.T

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

preds = mlp.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds, target_names=data.target_names))
mlp.visualizing_loss()

print("Example of Prediction")
raw_sample = data.data[0]
real_label = data.target_names[data.target[0]]
print(f"Real diagnosis: {real_label}")
mlp.predict_single(raw_sample, scaler, data.target_names)
