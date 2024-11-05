import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import Binarizer
from skimage.transform import resize

# Load and preprocess MNIST data to get 10 representative patterns
def preprocess_mnist(X, y, num_pixels=784):
    selected_patterns = []
    for digit in range(10):
        index = np.where(y == str(digit))[0][0]
        img = X[index].reshape(28, 28)
        
        # Resize image 
        if num_pixels != 784:
            img = resize(img, (int(num_pixels**0.5), int(num_pixels**0.5)), mode='reflect', anti_aliasing=True)
        
        # Flatten and binarize image (-1, 1)
        binary_img = Binarizer(threshold=0.5).fit_transform(img.flatten().reshape(1, -1))[0]
        binary_img = np.where(binary_img > 0, 1, -1)  # Convert to bipolar
        
        selected_patterns.append(binary_img)
    
    return np.array(selected_patterns).T  # Transpose for 784 x 10

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Prepare the 10 patterns (10 representative digits from 0 to 9)
P = preprocess_mnist(X, y)  # Each column is a pattern

# Initialize weights with Hebbian rule
num_neurons = P.shape[0]
w = np.zeros((num_neurons, num_neurons))

# Hebbian learning rule: w[i, j] = Σ (P[i, k] * P[j, k]) / P.shape[1]
for i in range(num_neurons):
    for j in range(i + 1, num_neurons):
        w[i, j] = np.sum(P[i, :] * P[j, :]) / P.shape[1]
        w[j, i] = w[i, j]  # Ensure symmetry
np.fill_diagonal(w, 0)  # Set diagonal elements to 0

# Recall function using energy minimization
def recall(weights, pattern, steps=5):
    for _ in range(steps):
        pattern = np.sign(np.dot(weights, pattern))
    return pattern

# Test the network by using a noisy version of a pattern
def test_network(weights, original_pattern, noise_level=0.1):
    # Add noise by flipping a percentage of the elements
    noisy_pattern = original_pattern.copy()
    flip_indices = np.random.choice(num_neurons, int(noise_level * num_neurons), replace=False)
    noisy_pattern[flip_indices] *= -1  # Flip selected bits
    
    # Recall the pattern
    recalled_pattern = recall(weights, noisy_pattern)
    return noisy_pattern, recalled_pattern

# Select a pattern to test (e.g., '9')
test_index = 9
original_pattern = P[:, test_index]

# Generate noisy and recalled patterns
noisy_pattern, recalled_pattern = test_network(w, original_pattern)

# Display original, noisy, and recalled patterns
def show_pattern(pattern, title, size=(28, 28)):
    plt.imshow(pattern.reshape(size), cmap='binary')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
show_pattern(original_pattern, "Original Pattern")
plt.subplot(1, 3, 2)
show_pattern(noisy_pattern, "Noisy Pattern")
plt.subplot(1, 3, 3)
show_pattern(recalled_pattern, "Recalled Pattern")
plt.show()
