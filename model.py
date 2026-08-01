import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("1. Loading the MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train = train_dataset.data.float() / 255.0
y_train = train_dataset.targets
x_test = test_dataset.data.float() / 255.0
y_test = test_dataset.targets

print("\n2. Exploring the data:")
print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

print("\n3. Preprocessing the data...")
x_train_flat = x_train.reshape((x_train.shape[0], 28*28))
x_test_flat = x_test.reshape((x_test.shape[0], 28*28))
print(f"Training samples used for model.fit: {x_train_flat.shape[0]}")

print("\n4. Building the model architecture...")
model = nn.Sequential(
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

print("\n5. Compiling the model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nModel Architecture:")
print(model)

train_loader = DataLoader(
    list(zip(x_train_flat, y_train)),
    batch_size=128,
    shuffle=True
)
val_loader = DataLoader(
    list(zip(x_train_flat[:6000], y_train[:6000])),
    batch_size=128,
    shuffle=False
)
print(f"Training loader contains all {len(train_loader.dataset)} samples")

print("\n6. Training the model...")
epochs = 20
best_val_acc = 0.0
best_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), 'digit_model.pth')

print("\n7. Evaluating on the test set...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in DataLoader(list(zip(x_test_flat, y_test)), batch_size=32):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"\n🎉 Final Test Accuracy: {test_accuracy*100:.2f}%")

print("\n8. Making a prediction on a test example...")
test_image = x_test_flat[0].unsqueeze(0).to(device)
with torch.no_grad():
    prediction = model(test_image)
    predicted_digit = torch.argmax(prediction).item()

true_digit = y_test[0].item()

print(f"Model's Prediction: {predicted_digit}")
print(f"Actual Digit: {true_digit}")


plt.imshow(x_test[0].cpu().numpy(), cmap='gray')
plt.title(f"Predicted: {predicted_digit}, Actual: {true_digit}")
plt.axis('off')
plt.show()