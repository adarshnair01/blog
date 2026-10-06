---
layout: post
title: "You're Training MNIST All Wrong: The Brutal Mathematics of Why Your Neural Network Actually Works"
date: 2026-07-22 20:51:22 +0530
excerpt: "Stop treating neural networks like black boxes. Let's peel back the layers of backpropagation and linear algebra to see what your code is *actually* doing to handwritten digits."
author: "Adarsh Nair"
categories: ai
tags: ["Deep Learning", "MNIST", "Mathematics", "Python", "Neural Networks"]
---

## The "Hello World" of AI That Everyone Gets Wrong

Every machine learning engineer, data science student, and curious Python scripter has touched the MNIST dataset. It’s the ritual initiation. Seventy thousand grayscale images of handwritten digits, meticulously collected, centered, and normalized. You import `torch` or `tensorflow`, slap together a couple of linear layers, hook up a Cross-Entropy loss function, and watch your validation accuracy rocket to 98% in under three epochs.

You feel like a god. You have built intelligence from scratch. 

Except... do you actually know what just happened inside that matrix multiplication?

Most tutorials treat the mechanics of deep learning like a magic trick. They hand you a pre-packaged autograd engine and tell you to wave the `optimizer.step()` wand. But beneath the shiny APIs lies a relentless, beautiful cascade of linear algebra, chain-rule calculus, and geometric deformations. Today, we are going to tear down the abstraction wall. We are going to look under the hood of MNIST and examine the brutal, elegant mechanics of deep learning.

---

## Anatomy of the Data: What is a "Digit" to a Computer?

Before a neural network can recognize a loop in an eight or the sharp angle of a seven, it has to ingest raw bytes. The MNIST dataset consists of $28 \times 28$ pixel images. To a computer, an image is not a picture; it is a tensor $\mathbf{X} \in \mathbb{R}^{28 \times 28}$, where each element represents a scalar intensity value mapped between $0$ (pure black) and $1$ (pure white).

When we feed this into a standard Multi-Layer Perceptron (MLP), we commit our first act of geometric violence: we flatten the tensor.

$$\mathbf{x} = \text{flatten}(\mathbf{X}) \in \mathbb{R}^{784}$$

Suddenly, our spatial data—where pixels have local neighbors, edges, and corners—is converted into a 784-dimensional vector. The network has no inherent concept of "up," "down," "left," or "right." It only knows 784 independent axes of variation. Every single weight in our first hidden layer must learn to make sense of this unspooled ball of yarn.

---

## Forward Propagation: Transforming Spaces

Let’s build a minimal, NumPy-only deep learning pipeline to understand the mechanics without PyTorch hiding the plumbing. We need a simple two-layer network: an input layer of 784 nodes, a hidden layer of 128 nodes with a ReLU activation, and an output layer of 10 nodes (representing digits 0 through 9).

```python
import numpy as np

def init_parameters(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    # He initialization to prevent exploding/vanishing gradients
    W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((hidden_dim, 1))
    W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros((output_dim, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
```

When an image vector $\mathbf{x}$ enters the first layer, the network performs an affine transformation:

$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$

Here, $\mathbf{W}_1$ is a matrix of size $128 \times 784$. Think of each row of $\mathbf{W}_1$ as a template or a linear filter. When we take the dot product of a row in $\mathbf{W}_1$ with our input vector $\mathbf{x}$, we are calculating a weighted similarity score. If a row in $\mathbf{W}_1$ matches the pattern of a loop in the upper quadrant of the image, the dot product spikes.

Next, we apply a non-linear activation function, ReLU (Rectified Linear Unit):

$$\mathbf{a}_1 = \max(0, \mathbf{z}_1)$$

Why do we need this? Without non-linearity, stacking ten linear layers is mathematically equivalent to stacking one linear layer. The network would just collapse into a single matrix multiplication. Non-linearity allows the network to carve complex, non-linear decision boundaries through our 784-dimensional space, separating the twisted loops of a '6' from the jagged lines of a '4'.

---

## The Loss Landscape and Cross-Entropy

Once our forward pass reaches the output layer, we get raw, unnormalized scores called logits:

$$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2 \quad (\mathbf{z}_2 \in \mathbb{R}^{10})$$

To turn these logits into probabilities that sum to 1, we use the Softmax function:

$$\hat{y}_i = \frac{e^{z_{2,i}}}{\sum_{j=0}^{9} e^{z_{2,j}}}$$

Now we measure how wrong our network is using Categorical Cross-Entropy Loss. For a single true label $y$ (represented as a one-hot vector) and predicted probabilities $\hat{\mathbf{y}}$:

$$\mathcal{L} = -\sum_{i=0}^{9} y_i \log(\hat{y}_i)$$

If the true label is '3', $y_3 = 1$ and all other $y_i = 0$. The loss simplifies to $-\log(\hat{y}_3)$. If our network is confident that the image is a '3' ($\hat{y}_3 \to 1$), the loss approaches $0$. If it thinks it's an '8' ($\hat{y}_3 \to 0$), the loss shoots toward infinity.

---

## Backpropagation: The Chain Rule in Action

This is where the magic—and the math—happens. Backpropagation is nothing more than systematic application of the chain rule from calculus. We want to find how much a tiny change in weight $W_{ij}$ affects the final loss $\mathcal{L}$.

We work backward from the output:

1. **Error at the output layer:**
   $$\delta_2 = \hat{\mathbf{y}} - \mathbf{y}$$
2. **Gradients for the second layer weights and biases:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \delta_2 \mathbf{a}_1^T$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \delta_2$$
3. **Propagate error back to the hidden layer:**
   $$\delta_1 = (\mathbf{W}_2^T \delta_2) \odot \mathbb{I}(\mathbf{z}_1 > 0)$$
   *(where $\mathbb{I}$ is the indicator function representing the derivative of ReLU)*
4. **Gradients for the first layer weights and biases:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \delta_1 \mathbf{x}^T$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \delta_1$$

With these gradients computed, we take a step in the opposite direction of the gradient to minimize the loss (Gradient Descent):

$$\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}_1}$$

where $\eta$ is our learning rate.

---

## Putting It All Together: Complete NumPy Implementation

Let's write out a functional training loop in pure NumPy so you can see every gear turning.

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_propagation(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "X": X}
    return A2, cache

def backward_propagation(cache, params, Y):
    m = Y.shape[1]
    X, A1, A2 = cache["X"], cache["A1"], cache["A2"]
    W2 = params["W2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dZ1 = np.dot(W2.T, dZ2) * (A1 > 0)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def update_parameters(params, grads, learning_rate):
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params
```

When you feed MNIST batches through these functions iteratively, you are watching geometry reshape itself. High-dimensional vectors are being rotated, stretched, and folded until all ten digit classes form cleanly separated clusters.

---

## Why MNIST Still Matters

Critics love to dismiss MNIST as a solved problem. "Why are you talking about handwriting digits in the era of multimodal transformers?" 

Because if you cannot understand why an MLP fails to capture translational invariance—and why a Convolutional Neural Network (CNN) with weight sharing fixes it—you don't understand computer vision. If you can't debug a exploding gradient in a simple two-layer network, you won't survive debugging a 100-billion-parameter LLM.

MNIST is our laboratory. It is where calculus meets pixels, and where abstract theory turns into working code. Stop treating your models like magic. Open the hood, look at the gradients, and master the mechanics.