"""
EXPERIMENT 10: ANFIS - Vehicle Fuel Consumption Prediction
A compact first-order Sugeno ANFIS demonstration.
Inputs: engine speed and vehicle load. Output: fuel consumption.
Gaussian membership functions are fixed; consequent parameters are learned
using least squares (a simplified educational ANFIS implementation).
"""

import numpy as np

# Small sample dataset: [engine speed, vehicle load] -> fuel consumption
data = np.array([
    [1000, 20,  4.0], [1500, 30, 5.0], [2000, 40, 6.5],
    [2500, 50, 8.0], [3000, 60, 9.5], [3500, 70, 11.0],
    [1200, 80, 7.0], [2200, 20, 6.0], [3200, 35, 8.5],
    [1800, 75, 8.0], [2800, 85, 11.0], [3800, 90, 13.0]
], dtype=float)

X = data[:, :2]
y = data[:, 2]

# Normalize inputs to make membership functions easier to define
mean = X.mean(axis=0)
std = X.std(axis=0)
Xn = (X - mean) / std

# Two Gaussian membership functions for each input: low and high
centres = [-0.7, 0.7]
sigma = 0.8

def gaussian(x, centre):
    return np.exp(-0.5 * ((x - centre) / sigma) ** 2)

def rule_strengths(row):
    e_low, e_high = gaussian(row[0], centres[0]), gaussian(row[0], centres[1])
    l_low, l_high = gaussian(row[1], centres[0]), gaussian(row[1], centres[1])
    return np.array([e_low*l_low, e_low*l_high, e_high*l_low, e_high*l_high])

# Each Sugeno rule has output: p*engine_speed + q*load + r
A = []
for row in Xn:
    strengths = rule_strengths(row)
    normalized = strengths / (strengths.sum() + 1e-9)
    rule_features = []
    for w in normalized:
        rule_features.extend([w * row[0], w * row[1], w])
    A.append(rule_features)
A = np.array(A)

# Learn consequent parameters by least squares
parameters = np.linalg.lstsq(A, y, rcond=None)[0]

def predict(engine_speed, vehicle_load):
    row = (np.array([engine_speed, vehicle_load]) - mean) / std
    strengths = rule_strengths(row)
    weights = strengths / (strengths.sum() + 1e-9)
    features = []
    for w in weights:
        features.extend([w * row[0], w * row[1], w])
    return float(np.dot(features, parameters))

predictions = np.array([predict(a, b) for a, b in X])
mse = np.mean((predictions - y) ** 2)

print("Training MSE:", round(mse, 4))
speed = float(input("Enter engine speed (RPM): "))
load = float(input("Enter vehicle load: "))
print("Predicted fuel consumption:", round(predict(speed, load), 2))
