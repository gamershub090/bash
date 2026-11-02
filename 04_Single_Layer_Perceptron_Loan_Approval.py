"""
EXPERIMENT 4: Single Layer Perceptron - Intelligent Loan Approval
Inputs: income, credit score, employment years, debt, requested loan.
Target: 1 = Approved, 0 = Rejected.
"""

import numpy as np

# Training data from the supplied problem statement (sample rows)
X = np.array([
    [8.5,760,5,1.2,3.0], [12,810,8,2.0,5.0], [6.5,720,4,1.0,2.5],
    [15,850,10,3.0,7.0], [9,690,6,2.5,4.0], [5.5,650,2,2.8,4.0],
    [7,610,3,4.0,5.0], [4.5,580,1,2.5,3.5], [10,620,7,6.0,8.0],
    [6,670,2,3.5,5.5], [11.5,780,7,2.0,4.0], [8,640,3,4.5,6.0]
], dtype=float)
y = np.array([1,1,1,1,1,0,0,0,0,0,1,0])

# Standardize inputs so large-valued features do not dominate
mean = X.mean(axis=0)
std = X.std(axis=0)
X_scaled = (X - mean) / (std + 1e-8)

weights = np.zeros(X.shape[1])
bias = 0.0
learning_rate = 0.1

# Perceptron learning rule
for epoch in range(100):
    for xi, target in zip(X_scaled, y):
        net = np.dot(xi, weights) + bias
        prediction = 1 if net >= 0 else 0
        error = target - prediction
        weights += learning_rate * error * xi
        bias += learning_rate * error

print("Training complete")
print("Weights:", weights)
print("Bias:", bias)

# Predict a new applicant: income, credit, employment, debt, loan
new_applicant = np.array([[9.0, 740, 4, 2.0, 3.5]])
new_scaled = (new_applicant - mean) / (std + 1e-8)
result = 1 if np.dot(new_scaled[0], weights) + bias >= 0 else 0
print("New applicant decision:", "APPROVED" if result == 1 else "REJECTED")
