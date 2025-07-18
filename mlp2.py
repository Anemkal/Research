
import numpy as np
from typing import List, Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x: np.ndarray) -> np.ndarray:
    # TODO: implement element-wise derivative of sigmoid
    return x * ( 1 - x )


def softmax(x: np.ndarray) -> np.ndarray:
    # TODO
    ex = np.exp(x - np.max(x, axis=1, keepdims = True))
    return ex / np.sum(ex, axis=1, keepdims=True)



def cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    # TODO: implement numerically stable cross-entropy
    y_hat = np.clip(y_hat, 1e-7, 1-1e-7)
    return -np.sum(y_true * np.log(y_hat)) / y_true.shape[0]



def d_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # TODO: implement gradient of loss w.r.t. network output
    return (y_hat - y_true) / len(y_true)



class Dense:

    def __init__(self, in_dim: int, out_dim: int, weight_scale: float = 1e-2):
        self.W = weight_scale * np.random.randn(out_dim, in_dim)
        self.b = np.zeros(out_dim)
        self.x: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W.T + self.b  # (batch, out_dim)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # TODO: implement gradient calculations
        self.dW = grad_out.T @ self.x
        self.db = np.sum(grad_out, axis=0)
        grad_input = grad_out @ self.dW

        return grad_input

    def step(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db


class MLP:

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        *,
        weight_scale: float = 1e-2,
        learning_rate: float = 1e-3,
    ):
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers: List[Dense] = [
            Dense(dims[i], dims[i + 1], weight_scale=weight_scale)
            for i in range(len(dims) - 1)
        ]
        self.learning_rate = learning_rate

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = sigmoid(x)
        return x

    def backward(self, logits: np.ndarray, y_true: np.ndarray) -> None:
        # TODO
        grad = d_cross_entropy(softmax(logits), y_true)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)


    def step(self) -> None:
        # TODO: loop over layers and update W, b with learning rate
        for layer in self.layers:
            layer.step(self.learning_rate)


    def train_step(
        self, x: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        logits = self.forward(x)
        loss = cross_entropy(softmax(logits), y_true)
        self.backward(logits, y_true)
        self.step()
        return loss, logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(softmax(logits), axis=1)


# testing
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
y_onehot = one_hot(y, 2)

model = MLP(input_dim=2, hidden_dims=[4], output_dim=2, learning_rate=0.1)

for epoch in range(1000):
    loss, _ = model.train_step(X, y_onehot)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

preds = model.predict(X)
print("Predictions:", preds)
print("True labels:", y)