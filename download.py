from torchvision import datasets, transforms
import os
from PIL import Image

# Define preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download MNIST
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create folders for each digit (0-9)
base_path = './data/mnist_images'
for digit in range(10):
    os.makedirs(f'{base_path}/{digit}', exist_ok=True)

# Save training images sorted by digit
for idx, (image, label) in enumerate(train_dataset):
    # Convert tensor back to PIL Image
    image = transforms.ToPILImage()(image)
    # Save to appropriate folder
    image.save(f'{base_path}/{label}/train_{idx}.png')

# Save test images sorted by digit
for idx, (image, label) in enumerate(test_dataset):
    image = transforms.ToPILImage()(image)
    image.save(f'{base_path}/{label}/test_{idx}.png')

print("Done! Images organized in ./data/mnist_images/0/ through ./data/mnist_images/9/")