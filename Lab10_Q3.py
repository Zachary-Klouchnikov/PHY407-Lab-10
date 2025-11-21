__authors__ = "Zachary Klouchnikov and Hannah Semple"

# The following code estimates integrals using both mean value sampling and
# importance sampling methods. It then plots histograms of the results.

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""
FUNCTIONS
"""
def mean_value(f: Callable, a: float, b: float, n: int) -> float:
    """Returns the mean value estimate of the integral of f from a to b using n
    sample points.
    
    Arguments:
    f -- function to integrate
    a -- lower integration limit
    b -- upper integration limit
    n -- number of sample points
    """
    k = 0.0 # Stores the average of f(x)

    for _ in range(n):
        x = (b - a) * np.random.random()
        k += f(x)

    return k * (b - a) / n

def importance_sampling(f: Callable, p: Callable, s: Callable, n: int) -> float:
    """Returns the importance sampling estimate of the integral of f using n
    sample points with a probability density function p. Pointes are sampled
    using s.
    
    Arguments:
    f -- function to integrate
    p -- probability density function
    s -- sampling function
    n -- number of sample points
    """
    k = 0.0 # Stores the average of f(x) / p(x)

    for _ in range(n):
        x = s()
        k += f(x) / p(x)

    return k / n

"""
PART A)
"""
"Constants"
N = 10000 # Number of sample points
A = 0 # Lower integration limit
B = 1 # Upper integration limit

"Initialize arrays to store results"
mean_value_array = np.empty(100) # Stores mean value estimates
importance_sampling_array = np.empty(100) # Stores importance sampling estimates

"Define functions"
f = lambda x: (x ** -0.5) / (1 + np.exp(x)) # Equation (8)
p = lambda x: 1 / (2 * np.sqrt(x)) # Equation (9)
s = lambda: np.random.random() ** 2 # Sampling function for p(x)

"Compute estimates"
for i in range(100):
    mean_value_array[i] = mean_value(f, A, B, N)
    importance_sampling_array[i] = importance_sampling(f, p, s, N)

"Plotting Integrand Estimates of Equation (8)"
plt.figure()

# Plotting integrand estimates of Equation (8)
plt.hist(mean_value_array, bins = 10, range = [0.8, 0.88], color = 'Teal', label = "Mean Value Sampling")
plt.hist(importance_sampling_array, bins = 10, range = [0.8, 0.88], alpha = 0.7, color = 'Coral', label = "Importance Sampling")

# Labels
plt.title("Integrand Estimates of Equation (8)", fontsize = 12)
plt.xlabel("Integral Estimate", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Integrand Estimates of Equation (8).pdf')
plt.show()

"""
PART B)
"""
"Constants"
N = 10000 # Number of sample points
A = 0 # Lower integration limit
B = 10 # Upper integration limit

"Initialize arrays to store results"
mean_value_array = np.empty(100) # Stores mean value estimates
importance_sampling_array = np.empty(100) # Stores importance sampling estimates

"Define functions"
f = lambda x: np.exp(-2 * np.abs(x - 5)) # Equation (10)
p = lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 5) ** 2) # Equation (12)
s = lambda: np.random.normal(5, 1) # Sampling function for p(x)

"Compute estimates"
for i in range(100):
    mean_value_array[i] = mean_value(f, A, B, N)
    importance_sampling_array[i] = importance_sampling(f, p, s, N)

"Plotting Integrand Estimates of Equation (10)"
plt.figure()

# Plotting integrand estimates of Equation (10)
plt.hist(mean_value_array, bins = 10, range = [0.96, 1.04], color = 'Teal', label = "Mean Value Sampling")
plt.hist(importance_sampling_array, bins = 10, range = [0.96, 1.04], color = 'Coral', alpha = 0.7, label = "Importance Sampling")

# Labels
plt.title("Integrand Estimates of Equation (10)", fontsize = 12)
plt.xlabel("Integral Estimate", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# plt.savefig('Figures\\Integrand Estimates of Equation (10).pdf')
plt.show()
