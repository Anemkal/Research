import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # self.h1 = nnLinear(input_dim, hidden1)
        # self.h2 = nn.Linear(hidden1, hidden2)
        # self.out = nn.Linear(hidden2, output_dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.sigmoid(x)
        return x
        # x = torch.sigmoid(self.h1(x))
        # x = torch.sigmoid(self.h2(x))
        # return self.out(x)


# training
def train(model, x, y_true, epochs=10000, lr=0.5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y_true)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return model


X_np = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_np = [0, 1, 1, 0]

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)
model = MLP(input_dim=2, hidden_dims=[4, 4], output_dim=2)
model = train(model, X, y)

# testing
with torch.no_grad():
    logits = model(X)
    predicted_classes = logits.argmax(dim=1)
    print("Predicted labels:", predicted_classes.numpy())
    print("True labels:", y.numpy())
