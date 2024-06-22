from math import sqrt

from scipy.stats import norm

# Calculate the p-value

z_score = 6.9

p_value = 2 * norm.sf(abs(z_score))

# Print the p-value

print("p-value:", 56*56*p_value)

z_score = 5

p_value = 2 * norm.sf(abs(z_score))

# Print the p-value

print("p-value:", p_value)
