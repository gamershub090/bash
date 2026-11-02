"""
EXPERIMENT 8: Fuzzy Sets and Fuzzy Relations
Room temperature -> fan speed using fuzzy operations and max-min composition.
"""

import numpy as np

temperature = np.array([10, 15, 20, 25, 30, 35, 40])
fan_speed = np.array([0, 25, 50, 75, 100])

# Classical sets: membership is either 0 or 1
classical_low = (temperature <= 20).astype(int)
classical_medium = ((temperature > 20) & (temperature <= 30)).astype(int)
classical_high = (temperature > 30).astype(int)

# Fuzzy sets: membership values range from 0 to 1
low = np.array([1, 1, .75, .25, 0, 0, 0])
medium = np.array([0, .25, .75, 1, .75, .25, 0])
high = np.array([0, 0, 0, .25, .75, 1, 1])

print("Temperature:", temperature)
print("Classical low:", classical_low)
print("Fuzzy low:", low)
print("Fuzzy medium:", medium)
print("Fuzzy high:", high)

# Fuzzy set operations
print("\nUnion (low OR medium):", np.maximum(low, medium))
print("Intersection (low AND medium):", np.minimum(low, medium))
print("Complement (NOT low):", 1 - low)

# Fuzzy fan-speed sets
slow = np.array([1, .75, .25, 0, 0])
moderate = np.array([0, .25, .75, 1, .75])
fast = np.array([0, 0, .25, .75, 1])

# Cartesian fuzzy relations using minimum (Mamdani relation)
R_low = np.minimum(low[:, None], slow[None, :])
R_med = np.minimum(medium[:, None], moderate[None, :])
R_high = np.minimum(high[:, None], fast[None, :])
R = np.maximum.reduce([R_low, R_med, R_high])

# Max-Min composition for an example temperature membership vector
input_temp = np.array([0, 0, .2, .8, 1, .2, 0])
output = np.max(np.minimum(input_temp[:, None], R), axis=0)

print("\nFuzzy relation matrix:\n", R)
print("Max-Min output membership:", output)
print("Suggested fan speed:", fan_speed[np.argmax(output)], "%")
