"""
EXPERIMENT 9: Fuzzy Logic Controller - Room Cooling/Fan Controller
Mamdani inference with temperature and humidity inputs.
Uses simple triangular membership functions and centroid defuzzification.
"""

import numpy as np

def trimf(x, a, b, c):
    left = (x - a) / (b - a + 1e-9)
    right = (c - x) / (c - b + 1e-9)
    return np.maximum(0, np.minimum(left, right))

def trapmf(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum(
        (x-a)/(b-a+1e-9), 1), (d-x)/(d-c+1e-9)))

# Universes
T = np.arange(0, 51, 1)       # temperature in Celsius
H = np.arange(0, 101, 1)      # humidity percentage
S = np.arange(0, 101, 1)      # fan speed percentage

# Membership functions
cold = trapmf(T, 0, 0, 15, 25)
warm = trimf(T, 15, 27, 38)
hot = trapmf(T, 30, 40, 50, 50)

dry = trapmf(H, 0, 0, 25, 50)
normal = trimf(H, 25, 50, 75)
humid = trapmf(H, 50, 75, 100, 100)

low = trapmf(S, 0, 0, 20, 45)
medium = trimf(S, 25, 50, 75)
high = trapmf(S, 55, 80, 100, 100)

def membership(universe, mf, value):
    return float(np.interp(value, universe, mf))

def controller(temp, humidity):
    tc = [membership(T, cold, temp), membership(T, warm, temp), membership(T, hot, temp)]
    hm = [membership(H, dry, humidity), membership(H, normal, humidity),
          membership(H, humid, humidity)]

    # Rules: (temperature category, humidity category) -> fan category
    # Output category: 0=low, 1=medium, 2=high
    rules = [
        (0,0,0), (0,1,0), (0,2,1),
        (1,0,1), (1,1,1), (1,2,2),
        (2,0,2), (2,1,2), (2,2,2)
    ]
    output_mfs = [low, medium, high]
    aggregated = np.zeros_like(S, dtype=float)

    for ti, hi, out in rules:
        strength = min(tc[ti], hm[hi])
        aggregated = np.maximum(aggregated, np.minimum(strength, output_mfs[out]))

    # Centroid defuzzification
    crisp = np.sum(S * aggregated) / (np.sum(aggregated) + 1e-9)
    return crisp

temp = float(input("Enter room temperature (0-50 C): "))
humidity = float(input("Enter humidity (0-100 %): "))
speed = controller(temp, humidity)

print("Recommended fan speed:", round(speed, 2), "%")
