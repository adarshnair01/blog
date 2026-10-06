---
layout: post
title: "You're Training MNIST All Wrong: The Brutal Math and Code Behind Deep Learning Mechanics"
date: 2026-08-19 08:55:23 +0530
excerpt: "Think MNIST is just a toy dataset? Think again. Dive deep into the unvarnished tensor mechanics, backpropagation calculus, and forward-pass engineering that separates script kiddies from real AI engineers."
author: "Adarsh Nair"
categories: ai
tags: ["Deep Learning", "MNIST", "Python", "PyTorch", "Neural Networks"]
---

# You're Training MNIST All Wrong: The Brutal Math and Code Behind Deep Learning Mechanics

Every machine learning practitioner has trained a model on the Modified National Institute of Standards and Technology (MNIST) database. It is the "Hello, World!" of computer vision—a rite of passage featuring 60,000 training images and 10,000 testing images of handwritten digits, each sized at a humble $28 \times 28$ pixels. 

Yet, when most beginners approach MNIST, they treat neural networks as black-box magic. They import `torch` or `tensorflow`, slap together a couple of `nn.Linear` layers, call `.backward()`, and marvel at 98% accuracy. But do you actually know what is happening under the hood? Do you understand how gradients flow through high-dimensional weight spaces, or how matrix multiplication orchestrates feature extraction at the micro-level?

If you want to move from an API consumer to a foundational AI engineer, you need to understand the raw mechanics. Let's peel back the abstractions and look at the mathematical and computational reality of deep learning on MNIST.

---

## 1. The Anatomy of MNIST: Tensors and Shapes

Before a single weight is updated, we must confront the data structure. A single MNIST image is not an image to a computer; it is a rank-2 tensor of shape $(28, 28)$ containing floating-point values typically normalized between $0.0$ and $1.0$.

When we process data in mini-batches (say, a batch size of $64$), our input tensor $X$ escalates to a rank-3 tensor of shape:

$$\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$$

Where:
* $B = 64$ (Batch size)
* $C = 1$ (Grayscale channel)
* $H = 28$ (Height)
* $W = 28$ (Width)

To feed this into a standard Multi-Layer Perceptron (MLP), we must *flatten* the spatial dimensions. We reshape $\mathbf{X}$ from $(64, 1, 28, 28)$ into a matrix of shape $(64, 784)$. 

### The Flattening Operation in PyTorch

```python
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
```

Each of the 784 input features corresponds to a specific pixel location across all images in the batch. Every single one of these pixels will be multiplied by a corresponding weight in our first hidden layer.

---

## 2. The Forward Pass: Matrix Multiplications and Non-Linearities

At its core, a neural network layer is nothing more than an affine transformation followed by an element-wise non-linear activation function. 

For our first hidden layer with $H$ hidden units, the forward pass computes:

$$\mathbf{Z}^{(1)} = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}$$

$$\mathbf{A}^{(1)} = \sigma(\mathbf{Z}^{(1)})$$

Where:
* $\mathbf{X}$ is our input matrix of shape $(64, 784)$.
* $\mathbf{W}^{(1)}$ is our weight matrix of shape $(784, H)$. If $H = 128$, the shape is $(784, 128)$.
* $\mathbf{b}^{(1)}$ is our bias vector of shape $(128)$, broadcasted across the batch.
* $\mathbf{Z}^{(1)}$ is the pre-activation output of shape $(64, 128)$.
* $\sigma(\cdot)$ is our activation function (e.g., ReLU).

Let's implement a clean, low-level NumPy-style forward pass to visualize how data transforms through these dimensions.

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def forward_pass(X, W1, b1, W2, b2):
    # Step 1: Input to Hidden Layer
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    
    # Step 2: Hidden Layer to Output Layer (Logits)
    z2 = np.dot(a1, W2) + b2
    
    return z1, a1, z2
```

Notice that $\mathbf{Z}^{(2)}$ (the output of the final layer) produces raw, unnormalized scores known as **logits**, spanning from $-\infty$ to $+\infty$, with a shape of $(64, 10)$—one logit for each digit class from $0$ to $9$.

---

## 3. The Loss Function: Cross-Entropy and Softmax Mechanics

How do we evaluate how "wrong" our network is? We pass our logits $\mathbf{Z}^{(2)}$ through the **Softmax** function to convert them into a probability distribution $\hat{\mathbf{Y}}$, where all values sum to $1.0$ and range between $0$ and $1$:

$$\hat{y}_{i, c} = \frac{e^{z_{i,c}}}{\sum_{j=0}^{9} e^{z_{i,j}}}$$

Once we have our predicted probabilities, we compute the **Categorical Cross-Entropy Loss** across our batch:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=0}^{9} y_{i,c} \log(\hat{y}_{i,c})$$

Where $y_{i,c}$ is a binary indicator ($0$ or $1$) if class label $c$ is the correct classification for observation $i$.

### Numerical Stability Warning

If you write Softmax and Cross-Entropy from scratch without subtracting the maximum logit value from your vector ($\vec{z} - \max(\vec{z})$), you will quickly encounter `NaN` values due to floating-point overflow when exponentiating large numbers. Frameworks like PyTorch combine Softmax and CrossEntropy into a single numerically stable operation: `nn.CrossEntropyLoss()`.

---

## 4. Backpropagation: The Chain Rule in Action

This is where the magic happens. Backpropagation is simply systematic application of the multivariate chain rule of calculus. We want to find the partial derivative of the loss $\mathcal{L}$ with respect to every weight and bias in our network: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}}$, etc.

Starting from the output layer, the derivative of Cross-Entropy combined with Softmax simplifies remarkably to:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(2)}} = \hat{\mathbf{Y}} - \mathbf{Y}$$

This is simply our predicted probabilities minus the one-hot encoded true labels! 

To compute the gradients for our second-layer weights $\mathbf{W}^{(2)}$, we use the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = (\mathbf{A}^{(1)})^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(2)}} = (\mathbf{A}^{(1)})^T (\hat{\mathbf{Y}} - \mathbf{Y})$$

To propagate the error backward to the first layer, we dot-product the error gradient back through the weight matrix $\mathbf{W}^{(2)}$ and scale it by the derivative of the ReLU activation function:

$$\frac{\partial \mathcal{L}}{\brathb{Z}^{(1)}} = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{(2)}} (\mathbf{W}^{(2)})^T \right) \odot \mathbb{I}(\mathbf{Z}^{(1)} > 0)$$

Where $\odot$ represents the Hadamard (element-wise) product, and $\mathbb{I}$ is the indicator function acting as our ReLU derivative.

---

## 5. Complete PyTorch Implementation

Let's tie everything together into a clean, modular PyTorch implementation that respects the deep learning mechanics we just broke down.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 5

# 2. Data Loading & Normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. Architecture Definition
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# 4. Training Loop
model.train()
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()          # Clear previous gradients
        output = model(data)           # Forward pass
        loss = criterion(output, target) # Compute loss
        loss.backward()                # Backpropagation (compute gradients)
        optimizer.step()               # Parameter update (Gradient Descent step)

    print(f"Epoch {epoch+1}/{EPOCHS} completed. Loss: {loss.item():.4f}")
```

---

## 6. Optimization and Weight Updates

Once gradients are computed and stored in `.grad` attributes of our tensors, the optimizer updates each weight using Stochastic Gradient Descent (SGD):

$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}}$$

Where $\eta$ is our learning rate. If $\eta$ is too high, your loss will oscillate or diverge to infinity. If $\eta$ is too low, your model will take an eternity to crawl out of saddle points and local minima.

---

## Conclusion

MNIST is far more than a simple benchmark; it is a laboratory for understanding how machines learn to perceive structure out of chaos. By mastering the tensor shapes, matrix operations, loss landscapes, and calculus driving backpropagation, you strip away the abstraction layer. 

Next time you train a billion-parameter Large Language Model or a complex Vision Transformer, remember: the underlying mechanics remain fundamentally rooted in the same tensor multiplications and chain rules you just explored with 28x28 handwritten digits. Happy coding!