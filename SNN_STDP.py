import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import snntorch as snn 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])

# Training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 2. Define a Rate-Encoding Function
# -----------------------------
def rate_encode(img, num_steps=100):
    """
    Converts an image tensor into a binary spike train based on pixel intensities.
    Each pixel's intensity (0-1) is used as the probability of emitting a spike at each time step.
    """
    img_flat = img.view(-1)
    # Create spike train: shape [num_steps, num_pixels]
    spikes = torch.bernoulli(img_flat.repeat(num_steps, 1))
    return spikes

# -----------------------------
# 3. Define the SNN Model with Lateral Inhibition
# -----------------------------
class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_steps=100):
        super(SNN, self).__init__()
        self.num_steps = num_steps
        self.fc = nn.Linear(input_size, hidden_size, bias=False)
        self.beta = 0.9  # decay factor for membrane potential

        # Partition neurons into excitatory (80%) and inhibitory (20%).
        num_excitatory = int(0.8 * hidden_size)
        self.excitatory_mask = torch.zeros(hidden_size, device=device)
        self.inhibitory_mask = torch.zeros(hidden_size, device=device)
        self.excitatory_mask[:num_excitatory] = 1.0  # excitatory neurons
        self.inhibitory_mask[num_excitatory:] = 1.0  # inhibitory neurons

    def forward(self, x):
        """
        x: Input tensor of shape [batch, input_size] for one time step.
        Returns:
            spk: Spike output of shape [batch, hidden_size] for this time step.
        """
        batch_size = x.shape[0]
        # Note: For simplicity, we will update the membrane potential per time step here.
        # In an actual simulation, you'd handle stateful dynamics over time; here we assume each call is a single time step.
        mem = torch.zeros(batch_size, self.fc.out_features, device=x.device)
        cur = self.fc(x)
        mem = self.beta * mem + cur
        spk = (mem >= 1.0).float()  # generate spikes with threshold 1.0
        mem = mem * (1 - spk)       # reset membrane potential for neurons that spiked

        # Apply a simple lateral inhibition using the inhibitory mask
        inhibition = spk * self.inhibitory_mask
        spk = spk - inhibition
        spk = (spk > 0).float()  # ensure spikes are binary

        return spk

# -----------------------------
# 4. Define a Simple STDP Update Rule
# -----------------------------
def stdp_update(pre_spikes, post_spikes, weights, lr=1e-3):
    """
    Applies a simplified STDP rule.
    Args:
      pre_spikes: [batch, input_size] spike train at current time step.
      post_spikes: [batch, hidden_size] spike output at current time step.
      weights: current weight tensor.
      lr: learning rate.
    Returns:
      Updated weights.
    """
    batch_size = pre_spikes.shape[0]
    dw = torch.zeros_like(weights)
    for b in range(batch_size):
        # Outer product for each sample and accumulate the update.
        dw += torch.ger(post_spikes[b], pre_spikes[b])
    dw /= batch_size
    weights += lr * dw
    return weights

# -----------------------------
# 5. Model Hyperparameters and Instantiation
# -----------------------------
input_size = 28 * 28      # MNIST images are 28x28
hidden_size = 100         # Number of neurons in the SNN layer
num_steps = 100           # Simulation time steps per image
num_epochs = 100          # Number of training epochs

model = SNN(input_size, hidden_size, num_steps=num_steps).to(device)

# -----------------------------
# 6. Training Loop with STDP
# -----------------------------
print("Starting training ...")
spike_record_all = []  # Optional: To save spike patterns over batches

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, input_size).to(device)
        batch_size = images.shape[0]

        # Generate a spike train for each image: [num_steps, batch, input_size]
        spike_train = torch.stack([rate_encode(img, num_steps=num_steps) for img in images], dim=1).to(device)

        # Optional: clamp weights to be non-negative
        model.fc.weight.data.clamp_(0)

        total_post_spikes = torch.zeros(batch_size, hidden_size, device=device)
        # For each time step, forward pass and apply STDP
        for t in range(num_steps):
            x_t = spike_train[t]
            spk_t = model(x_t)  # shape [batch, hidden_size]
            total_post_spikes += spk_t
            # Update weights using the current time step's spikes
            new_weight = stdp_update(x_t, spk_t, model.fc.weight.data, lr=1e-3)
            model.fc.weight.data = new_weight

        # Optionally store spike patterns
        spike_record_all.append(total_post_spikes.cpu().detach().numpy())

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}]")
            
print("Training complete.")

# -----------------------------
# 7. Mapping Clusters to Digits (Neuron Label Assignment)
# -----------------------------
# For each neuron, count how many spikes are fired when an image of a given digit is shown.
neuron_digit_counts = np.zeros((hidden_size, 10))
for images, labels in train_loader:
    images = images.view(-1, input_size).to(device)
    batch_size = images.shape[0]
    spike_train = torch.stack([rate_encode(img, num_steps=num_steps) for img in images], dim=1).to(device)
    total_spikes = torch.zeros(batch_size, hidden_size, device=device)
    for t in range(num_steps):
        x_t = spike_train[t]
        spk_t = model(x_t)
        total_spikes += spk_t
    total_spikes = total_spikes.cpu().detach().numpy()
    labels = labels.numpy()
    for b in range(batch_size):
        neuron_digit_counts[:, labels[b]] += total_spikes[b]

# Assign to each neuron the digit for which it fired most frequently.
neuron_labels = np.argmax(neuron_digit_counts, axis=1)
print("Assigned labels to neurons:")
print(neuron_labels)

# -----------------------------
# 8. Testing Loop: Evaluate Accuracy on the Test Set
# -----------------------------
correct = 0
total = 0
all_true = []
all_pred = []

for images, labels in test_loader:
    images = images.view(-1, input_size).to(device)
    batch_size = images.shape[0]
    spike_train = torch.stack([rate_encode(img, num_steps=num_steps) for img in images], dim=1).to(device)
    total_spikes = torch.zeros(batch_size, hidden_size, device=device)
    
    # Accumulate spikes over time steps for test data.
    for t in range(num_steps):
        x_t = spike_train[t]
        spk_t = model(x_t)
        total_spikes += spk_t

    # Use the neuron mapping to form votes for each digit.
    digit_votes = torch.zeros(batch_size, 10, device=device)
    for neuron in range(hidden_size):
        digit = neuron_labels[neuron]
        digit_votes[:, digit] += total_spikes[:, neuron]
    
    predictions = torch.argmax(digit_votes, dim=1)
    labels = labels.to(device)
    correct += (predictions == labels).sum().item()
    total += batch_size

    all_true.extend(labels.cpu().numpy())
    all_pred.extend(predictions.cpu().numpy())

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# -----------------------------
# 9. Plotting for Visualization
# -----------------------------
# Plot the learned weight matrix.
weights = model.fc.weight.data.cpu().detach().numpy()
plt.figure(figsize=(8, 6))
plt.imshow(weights, aspect='auto', cmap='hot')
plt.colorbar()
plt.title("Learned Weight Matrix")
plt.xlabel("Input Neuron (Pixel)")
plt.ylabel("SNN Neuron")
plt.savefig("weights.png")
plt.show()
print("Saved weight matrix visualization as 'weights.png'.")

# Plot a Confusion Matrix for test predictions
cm = confusion_matrix(all_true, all_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(10)])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Set")
plt.savefig("confusion_matrix.png")
plt.show()
print("Saved confusion matrix as 'confusion_matrix.png'.")
