"""
EXPERIMENT 7: Associative Memory Network - Hopfield Network
Stores three 5x5 patterns and recalls a pattern from noisy input.
"""

import numpy as np

patterns = np.array([
    [[1,1,1,1,1], [1,0,0,0,1], [1,0,1,0,1], [1,0,0,0,1], [1,1,1,1,1]],
    [[1,0,0,0,1], [1,1,0,0,1], [1,0,1,0,1], [1,0,0,1,1], [1,0,0,0,1]],
    [[1,1,1,1,1], [1,0,0,0,0], [1,0,1,1,1], [1,0,0,0,0], [1,1,1,1,1]]
])

# Convert 0 -> -1 and 1 -> +1, then flatten to 25 neurons
bipolar = np.where(patterns == 0, -1, 1).reshape(3, 25)

# Hebbian weight matrix
W = np.zeros((25, 25))
for p in bipolar:
    W += np.outer(p, p)
np.fill_diagonal(W, 0)
W = W / len(bipolar)

# Add noise by flipping selected pixels in Pattern A
original = bipolar[0].copy()
noisy = original.copy()
noisy[[2, 8, 17]] *= -1

def recall(state, max_steps=100):
    state = state.copy()
    for _ in range(max_steps):
        old = state.copy()
        # Asynchronous update: update one neuron at a time
        for i in range(len(state)):
            net = np.dot(W[i], state)
            if net != 0:
                state[i] = 1 if net > 0 else -1
        if np.array_equal(state, old):
            break
    return state

recalled = recall(noisy)
accuracy = np.mean(recalled == original) * 100

print("Original pattern:\n", patterns[0])
print("Noisy pattern:\n", np.where(noisy.reshape(5,5) == -1, 0, 1))
print("Recalled pattern:\n", np.where(recalled.reshape(5,5) == -1, 0, 1))
print("Recall accuracy:", round(accuracy, 2), "%")
print("Weight matrix shape:", W.shape)
