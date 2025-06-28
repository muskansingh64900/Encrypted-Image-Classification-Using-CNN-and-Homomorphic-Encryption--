import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tenseal as ts
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                    download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                   download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_loss = []
train_accuracy = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct_train = 0
    total_train = 0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(trainloader)
    epoch_accuracy = 100 * correct_train / total_train
    train_loss.append(avg_loss)
    train_accuracy.append(epoch_accuracy)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

fc_layer = model.net[-1]
fc_weights = fc_layer.weight.detach().cpu().numpy()
fc_bias = fc_layer.bias.detach().cpu().numpy()

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.net.children())[:-1])
    
    def forward(self, x):
        return self.features(x)

feature_extractor = FeatureExtractor(model).to(device)
feature_extractor.eval()

def encrypted_predict(image_tensor):
    with torch.no_grad():
        features = feature_extractor(image_tensor.unsqueeze(0).to(device))
        features = features.cpu().numpy().flatten()
    
    encrypted_vec = ts.ckks_vector(context, features)
    
    logits = []
    for i in range(fc_weights.shape[0]):
        weights = fc_weights[i]
        bias = fc_bias[i]
        
        enc_logit = encrypted_vec.dot(weights)
        enc_logit = enc_logit + ts.ckks_vector(context, [bias])
        decrypted_logit = enc_logit.decrypt()[0]
        logits.append(decrypted_logit)
    
    return np.argmax(logits)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

plaintext_accuracy = 100 * correct / total
print(f"Plaintext Test Accuracy: {plaintext_accuracy:.2f}%")

correct_enc = 0
total_enc = 0

for images, labels in testloader:
    for i in range(images.size(0)):
        img = images[i]
        label = labels[i].item()
        
        pred = encrypted_predict(img)
        
        if pred == label:
            correct_enc += 1
        total_enc += 1

encrypted_accuracy = 100 * correct_enc / total_enc
print(f"Encrypted Test Accuracy: {encrypted_accuracy:.2f}%")

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Training Loss
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), train_loss, 'b-', marker='o', linewidth=2, markersize=6)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, num_epochs + 1))

# Plot 2: Training Accuracy
plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy, 'g-', marker='s', linewidth=2, markersize=6)
plt.title('Training Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, num_epochs + 1))
plt.ylim(0, 100)

# Plot 3: Comparison of Plaintext vs Encrypted Accuracy
plt.subplot(1, 3, 3)
methods = ['Plaintext', 'Encrypted']
accuracies = [plaintext_accuracy, encrypted_accuracy]
colors = ['skyblue', 'lightcoral']
bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
plt.title('Plaintext vs Encrypted Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Additional plot: Sample predictions visualization
plt.figure(figsize=(12, 8))
sample_images = []
sample_labels = []
sample_predictions = []

# Get a few sample images for visualization
for images, labels in testloader:
    for i in range(min(8, images.size(0))):
        img = images[i]
        label = labels[i].item()
        pred = encrypted_predict(img)
        
        sample_images.append(img.squeeze().numpy())
        sample_labels.append(label)
        sample_predictions.append(pred)
        
        if len(sample_images) >= 8:
            break
    if len(sample_images) >= 8:
        break

# Plot sample predictions
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_images[i], cmap='gray')
    color = 'green' if sample_labels[i] == sample_predictions[i] else 'red'
    plt.title(f'True: {sample_labels[i]}, Pred: {sample_predictions[i]}', 
              color=color, fontweight='bold')
    plt.axis('off')

plt.suptitle('Sample Encrypted Predictions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
