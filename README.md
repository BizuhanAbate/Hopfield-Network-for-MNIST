# Hopfield Network for Digit Pattern Recall

This project implements a Hopfield Network to recall digit patterns from the MNIST dataset. Using Hebbian learning, 
the network is trained to store a set of representative digit patterns (0-9). 
It then attempts to reconstruct these patterns from noisy input versions by iteratively updating neuron states.

Hopfield Networks have several applications, particularly in the following areas:

   - `Associative Memory` : They can store and retrieve patterns, making them useful for associative memory tasks. They can recall a stored pattern even when presented with a noisy or incomplete version of it.

   - `Optimization Problems` : They can be used to solve optimization problems, such as the traveling salesman problem. They can find solutions by searching for low-energy states that represent optimal configurations.

   - `Pattern Recognition` : They can recognize patterns in data, making them applicable in image and speech recognition tasks where patterns need to be identified despite noise or distortion.

   - `Content Addressable Memory` : They provide content addressable memory, allowing data retrieval based on partial inputs rather than specific addresses, useful in database searching and retrieval systems.

   - `Robotics` : In robotics, they can be used for path planning and decision-making, helping robots learn from past experiences and navigate environments efficiently.

   - `Signal Processing` : They can be applied in signal processing tasks for filtering and denoising, as they can retrieve clean signals from corrupted versions.

While Hopfield Networks are useful in associative memory and optimization, have several limitations:
 - `Limited Capacity` : They can store only a small number of patterns reliably. With too many patterns, the network risks spurious states and performance drops.

 - `Convergence Issues` : The network may struggle to reach a stable state, especially with noisy or incomplete patterns, sometimes converging to incorrect patterns.

 - `Pattern Interference` : Similar patterns can interfere with each other, leading to incorrect or distorted recalls, making it unreliable for closely related patterns.

 - `Slow Convergence` : For larger networks or complex patterns, convergence can be slow, requiring many iterations to stabilize.

 - `Symmetric Weights Requirement` : Hopfield Networks require symmetric weights, limiting their usefulness in tasks that benefit from asymmetric connections.

 - `No Hidden Layers` : With only a single layer, they lack the depth needed to learn complex patterns, limiting them to simpler tasks.

## Introduction

A Hopfield Network is a recurrent artificial neural network that can store patterns and retrieve them based on partial or noisy inputs. 
This project aims to store 10 distinct digit patterns (from the MNIST dataset) and test if the network can recall the original patterns even when given distorted versions of those patterns.

## How It Works

1. Data Preparation: 
   - The MNIST dataset is loaded, and 10 unique digit patterns (0 to 9) are extracted and binarized for use.
   
2. Network Training:
   - The network’s weight matrix is initialized based on the Hebbian learning rule:

      - $$
   w[i, j] = \sum_{k=1}^{P} \left( \frac{P[i, k] \times P[j, k]}{P.shape[1]} \right)
   $$

   - Each neuron's self-weight (diagonal of the weight matrix) is set to zero to prevent self-feedback.
   
3. Pattern Recall:
   - To test recall, noise is introduced to a pattern by flipping a subset of its pixels.
   - The noisy pattern is fed into the Hopfield network, which iteratively updates its state using the weighted sum of connected neurons until it converges to a stable state.

## Dataset

This implementation uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 70,000 images of handwritten digits (0-9) each of size 28x28 pixels.

## Installation

To run this code, you'll need Python 3 and several packages. You can install the required libraries with:

```bash
pip install numpy matplotlib scikit-image scikit-learn
```

## Usage

### Running the Network

1. Load and preprocess data:
   - Run the code to load the MNIST dataset and select one sample of each digit (0 to 9).
   
2. Train the Hopfield Network:
   - The code initializes the weight matrix based on Hebbian learning using the original patterns.

3. Test Recall with Noise:
   - The network tests its recall ability by taking a noisy pattern and trying to retrieve the closest stored pattern.
   - Results display the original pattern, the noisy input, and the network’s recalled pattern.

### Code Example

```python
# Initialize weights using Hebbian rule
w = np.zeros((784, 784))
for i in range(784):
    for j in range(i + 1, 784):
        for k in range(10):
            w[i, j] += 1 / 10 * (P[i, k] * P[j, k])
            w[j, i] = w[i, j]
np.fill_diagonal(w, 0)

# Recall a pattern
original_pattern = P[:, test_index]
noisy_pattern, recalled_pattern = test_network(w, original_pattern, noise_level=0.1)
```

## Results

After training, the network can recall the original patterns from noisy versions, with varying degrees of success depending on the noise level and number of training iterations.

- Original Pattern: The actual digit pattern stored in the network.
- Noisy Pattern: The pattern with added noise (randomly flipped pixels).
- Recalled Pattern: The pattern that the network recalls, ideally matching the original pattern.
