---
layout: post
title: "Under the Hood of MNIST: The Brutal Mathematical Machinery They Don't Teach in Bootcamp"
date: 2026-06-15 15:31:07 +0530
excerpt: "Think you understand deep learning because you can run model.fit()? Let's tear apart the raw mathematical machinery of MNIST to see what is actually happening under the hood."
author: "Adarsh Nair"
categories: ai
tags: ["AI", "Tech", "Deep Learning", "Mathematics"]
---

Every software engineer entering the field of artificial intelligence goes through the exact same rite of passage: classifying handwritten digits from the MNIST dataset. It is the "Hello World" of machine learning. You import PyTorch or TensorFlow, stack three lines of code, call `.fit()`, and watch the validation accuracy climb to 98%. 

It feels like magic. But that magic is a dangerous abstraction.

When you rely solely on high-level APIs, you miss the breathtaking, brutal, and elegant mathematical machinery that actually makes deep learning work. Underneath those clean method calls lies a chaotic landscape of high-dimensional coordinate spaces, non-linear folding operations, gradient vectors, and optimization algorithms navigating complex topologies.

Today, we are going to tear down the abstractions. We will build a Multi-Layer Perceptron (MLP) from absolute scratch using nothing but raw NumPy. No PyTorch Autograd. No Keras. Just pure, unadulterated linear algebra and calculus. 

By the end of this guide, you will understand the deep learning mechanics of MNIST at a molecular level.

---

## 1. The Coordinate Space: Mapping Pixels to Dimensions

Before we write a single line of math, we must understand what our data actually represents.

An MNIST image is a $28 \times 28$ grid of grayscale pixels. Each pixel has an intensity value ranging from 0 (black) to 255 (white). To feed this into a neural network, we flatten this grid into a single 1D vector $X$ of size $784$.

$$\mathbf{x} \in \mathbb{R}^{784}$$

From a geometric perspective, our input is not a picture of a digit; it is a single point situated inside a **784-dimensional space**. The goal of our neural network is to draw highly complex, non-linear boundary lines in this 784-dimensional space to isolate the points representing "0s" from the "1s", "2s", and so on.

---

## 2. The Anatomy of the Forward Pass

Let's design a simple two-layer network to classify these points:
1. **Input Layer**: $784$ features.
2. **Hidden Layer**: $128$ neurons with a ReLU activation function.
3. **Output Layer**: $10$ neurons (one for each digit class) with a Softmax activation function.

```
Input (784) ---> [ W1, b1 ] ---> ReLU ---> Hidden (128) ---> [ W2, b2 ] ---> Softmax ---> Output (10)
```

### Layer 1: The Affine Transformation and Non-Linear Folding

The first step is an affine transformation. We project our 784-dimensional input into a 128-dimensional hidden space using our first weight matrix $W^{[1]}$ and bias vector $b^{[1]}$:

$$\mathbf{z}^{[1]} = \mathbf{X} \mathbf{W}^{[1]} + \mathbf{b}^{[1]}$$

Let's track the matrix dimensions for a batch of $m$ images:
*   $\mathbf{X}$ has dimensions $(m, 784)$
*   $\mathbf{W}^{[1]}$ has dimensions $(784, 128)$
*   $\mathbf{b}^{[1]}$ has dimensions $(1, 128)$
*   $\mathbf{z}^{[1]}$ has dimensions $(m, 128)$

If we only used linear transformations, our network would be nothing more than a glorified linear regression model, incapable of solving non-linear classification boundaries (like distinguishing a curvy '8' from a straight '1'). To break this linearity, we pass $\mathbf{z}^{[1]}$ through the **Rectified Linear Unit (ReLU)** activation function:

$$\mathbf{a}^{[1]} = \max(0, \mathbf{z}^{[1]})$$

Geometrically, ReLU acts as a spatial folding mechanism. It takes any coordinate value below zero and collapses it to zero, effectively bending the high-dimensional space along the coordinate axes.

### Layer 2: Projecting to the Probability Simplex

Next, we project this 128-dimensional representation into a 10-dimensional output space:

$$\mathbf{z}^{[2]} = \mathbf{a}^{[1]} \mathbf{W}^{[2]} + \mathbf{b}^{[2]}$$

*   $\mathbf{W}^{[2]}$ has dimensions $(128, 10)$
*   $\mathbf{b}^{[2]}$ has dimensions $(1, 10)$
*   $\mathbf{z}^{[2]}$ has dimensions $(m, 10)$

These 10 output values are called **logits**. They range from negative infinity to positive infinity. To transform them into a valid probability distribution where all values sum to 1, we apply the **Softmax** function:

$$\mathbf{a}^{[2]}_i = \text{Softmax}(\mathbf{z}^{[2]}_i) = \frac{e^{\mathbf{z}^{[2]}_i}}{\sum_{j=1}^{10} e^{\mathbf{z}^{[2]}_j}}$$

Now, $\mathbf{a}^{[2]}$ represents our model's predicted probability distribution over the 10 digit classes.

---

## 3. The Loss Landscape: Cross-Entropy

How do we measure how wrong our network is? We use **Categorical Cross-Entropy Loss**. 

For a single training sample, if the true label is represented as a one-hot encoded vector $\mathbf{y}$ (where the correct index is 1 and all others are 0), the loss $L$ is:

$$L = -\sum_{k=1}^{10} \mathbf{y}_k \log(\mathbf{a}^{[2]}_k)$$

Because $\mathbf{y}$ is one-hot encoded, this simplifies to the negative log of the predicted probability for the *correct* class. If the network predicts a $99\%$ probability for the correct class, the loss is near $0$. If it predicts a $1\%$ probability, the loss explodes toward infinity.

---

## 4. Backpropagation: The Engine of Gradient Descent

This is where the real magic happens. To minimize the loss, we must calculate how our weights and biases affect it. We do this by working backward through the network using the calculus **Chain Rule**.

Let's calculate the gradients step-by-step.

### Step 1: Loss with respect to Output Logits ($\frac{\partial L}{\partial \mathbf{z}^{[2]}}$)

Combining Cross-Entropy and Softmax yields an incredibly clean derivative. When you compute the derivative of the combined loss with respect to the pre-activation output logits $\mathbf{z}^{[2]}$, the complex math simplifies down to a simple difference vector:

$$\mathbf{dZ}^{[2]} = \frac{\partial L}{\partial \mathbf{z}^{[2]}} = \mathbf{a}^{[2]} - \mathbf{y}$$

This represents the error of our prediction. If the model predicted $0.7$ for the correct class, the error is $0.7 - 1 = -0.3$.

### Step 2: Gradients of the Output Layer ($\mathbf{W}^{[2]}$, $\mathbf{b}^{[2]}$)

Using the chain rule, we find the gradient of our loss with respect to $\mathbf{W}^{[2]}$ and $\mathbf{b}^{[2]}$:

$$\mathbf{dW}^{[2]} = \frac{\partial L}{\partial \mathbf{W}^{[2]}} = \frac{1}{m} (\mathbf{a}^{[1]})^T \mathbf{dZ}^{[2]}$$

$$\mathbf{db}^{[2]} = \frac{\partial L}{\partial \mathbf{b}^{[2]}} = \frac{1}{m} \sum_{\text{rows}} \mathbf{dZ}^{[2]}$$

### Step 3: Gradients of the Hidden Layer ($\mathbf{W}^{[1]}$, $\mathbf{b}^{[1]}$)

Now we propagate the error further backward, through the non-linear ReLU activation:

$$\mathbf{da}^{[1]} = \mathbf{dZ}^{[2]} (\mathbf{W}^{[2]})^T$$

$$\mathbf{dZ}^{[1]} = \mathbf{da}^{[1]} \odot g'(\mathbf{z}^{[1]})$$

Where $\odot$ is the element-wise (Hadamard) product, and $g'(\mathbf{z}^{[1]})$ is the derivative of the ReLU function, which is $1$ for positive values and $0$ otherwise:

$$g'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

Finally, we calculate the gradients for the first layer's parameters:

$$\mathbf{dW}^{[1]} = \frac{1}{m} \mathbf{X}^T \mathbf{dZ}^{[1]}$$

$$\mathbf{db}^{[1]} = \frac{1}{m} \sum_{\text{rows}} \mathbf{dZ}^{[1]}$$

---

## 5. Building the Engine: Raw NumPy Implementation

Let's translate this mathematical framework into clean, optimized Python code.

```python
import numpy as np

class MNISTNeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # He (Kaiming) Initialization for weights, zeros for biases
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        # Subtraction of max prevents numerical overflow (exploding exponentials)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, A2, Y_one_hot):
        m = Y_one_hot.shape[0]
        # Added epsilon to avoid log(0) undefined errors
        epsilon = 1e-15
        A2 = np.clip(A2, epsilon, 1.0 - epsilon)
        loss = -np.sum(Y_one_hot * np.log(A2)) / m
        return loss

    def backward(self, X, Y_one_hot, learning_rate):
        m = X.shape[0]
        
        # Layer 2 gradients
        dZ2 = self.A2 - Y_one_hot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Layer 1 gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Parameter updates (Stochastic Gradient Descent)
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Utility functions to generate dummy/simulated MNIST data for mechanics validation
def get_one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# Mock validation run
if __name__ == "__main__":
    np.random.seed(42)
    # Simulate a mini-batch of 64 MNIST images
    X_dummy = np.random.rand(64, 784)
    y_dummy = np.random.randint(0, 10, size=64)
    Y_dummy_one_hot = get_one_hot(y_dummy, 10)
    
    nn = MNISTNeuralNetwork()
    
    # Run a single optimization step
    for epoch in range(5):
        predictions = nn.forward(X_dummy)
        loss = nn.compute_loss(predictions, Y_dummy_one_hot)
        nn.backward(X_dummy, Y_dummy_one_hot, learning_rate=0.1)
        print(f"Epoch {epoch+1} - Mathematical Loss: {loss:.6f}")
```

---

## 6. Optimization Mechanics: Navigating the Loss Landscape

The simple Parameter Update line in the code above is the physical execution of **Stochastic Gradient Descent (SGD)**:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

Here, $\eta$ is the learning rate, and $\nabla_\theta L$ is the multi-dimensional gradient vector pointing in the direction of the steepest ascent in our loss landscape. By subtracting this vector, we take a step down the hill.

In a production setting, raw SGD is often too slow or gets trapped in local minima and saddle points. Modern deep learning architectures utilize advanced optimizers like **Adam (Adaptive Moment Estimation)**. Adam tracks both the first moment (the mean) and the second moment (the uncentered variance) of the gradients to dynamically scale learning rates for every single weight in the network.

---

## 7. Why Understanding the Mechanics Matters

It is easy to think, *"Why should I care about this math when PyTorch does it automatically?"*

Here is why:
1. **Vanishing and Exploding Gradients**: If you initialize your weights incorrectly, the gradient calculations in backpropagation will recursively multiply values that are either too small (causing gradients to vanish to zero) or too large (causing them to explode to `NaN`). Knowing the math helps you understand why **He Initialization** is critical for ReLU networks.
2. **Dead ReLUs**: Because the derivative of ReLU is 0 for negative inputs, a large gradient update can knock a neuron into a state where it never activates on any training sample again. The neuron "dies," and its weights stop updating forever. Knowing this allows you to diagnose capacity issues and swap in **Leaky ReLU** or **GELU** activations.
3. **Architectural Design**: When you build custom transformers, diffusion models, or neural radiance fields, you aren't just stacking pre-made layers. You are designing custom forward loops that require mathematically sound gradient flows.

Abstractions are tools, not crutches. By understanding the low-level linear algebra and calculus under the hood of simple datasets like MNIST, you build the mental frameworks required to master the cutting-edge architectures of tomorrow.