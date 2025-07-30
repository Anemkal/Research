import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import tarfile
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# with tarfile.open('cifar-10-python.tar') as tar:
#     tar.extractall(path='./data')

# trainset = datasets.CIFAR10(root='./data', train=True, download=False)

dataset = torchvision.datasets.CIFAR10(
    root='.', train=True, download=True, transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)


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
# def train(model, x, y_true, epochs=10000, lr=0.5):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         logits = model(x)
#         loss = criterion(logits, y_true)
#         loss.backward()
#         optimizer.step()
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
#     return model

def train(model, dataloader, epochs=5, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.view(images.size(0), -1)  # Flatten [B, 3, 32, 32] → [B, 3072]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# a = [[0, 0], [0, 1], [1, 0], [1, 1]]
# b = [0, 1, 1, 0]
#
# X = torch.tensor(a, dtype=torch.float32)
# y = torch.tensor(b, dtype=torch.long)
# model = MLP(input_dim=2, hidden_dims=[4, 4], output_dim=2)
# model = train(model, X, y)
#
# # testing
# with torch.no_grad():
#     logits = model(X)
#     predicted_classes = logits.argmax(dim=1)
#     print("Predicted labels:", predicted_classes.numpy())
#     print("True labels:", y.numpy())

input_dim = 3 * 32 * 32
output_dim = 10

model = MLP(input_dim=input_dim, hidden_dims=[256, 128], output_dim=output_dim)
train(model, loader)

with torch.no_grad():
    images, labels = next(iter(loader))
    images_flat = images.view(images.size(0), -1)
    outputs = model(images_flat)
    predicted = outputs.argmax(dim=1)

    print("Predicted:", predicted[:10].tolist())
    print("True Labels: ", labels[:10].tolist())