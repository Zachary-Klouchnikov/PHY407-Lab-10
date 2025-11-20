__authors__ = "Zachary Klouchnikov and Hannah Semple"

# HEADER

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize.curve_fit as curve_fit

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""
FUNCTIONS
"""
def get_tau_step():
    """Calculate how far a photon travels before it gets scattered. Returns the
    optical depth traveled.
    """

    return -np.log(np.random.random())

def emit_photon(tau_max):
    """Emit a photon from the stellar core. Returns the optical depth and
    directional cosine of the photon emitted.

    Arguments:
    tau_max -- maximum optical depth of the atmosphere
    """
    mu = np.random.random()

    return tau_max - get_tau_step() * mu, mu

def scatter_photon(tau):
    """Scatter a photon. Returns the new optical depth and directional cosine
    of the photon after scattering.

    Arguments:
    tau -- optical depth of the atmosphere before scattering
    """
    mu = 2 * np.random.random() - 1 # Sample mu uniformly from -1 to 1

    return tau - get_tau_step() * mu, mu

def random_walk(tau_max):
    """Simulate the random walk of a photon through a stellar atmosphere.
    Returns the number of steps taken by the photon.

    Arguments:
    tau_max -- maximum optical depth of the atmosphere
    """
    tau, mu = emit_photon(tau_max)

    steps = 0
    while tau >= 0 and tau < tau_max:
        tau, mu = scatter_photon(tau)
        steps += 1

        # Re-emit photon if it scatters back into the core
        if tau >= tau_max:
            tau, mu = emit_photon(tau_max)
            steps = 0
    
    print(f"Photon escaped after {steps} steps with directional cosine {mu}")

    return steps, mu

def intensity(mu_list):
    """Returns the binned intensity I(mu) for a list of directional cosines.

    Arguments:
    mu_list -- list of directional cosines of escaping photons
    """
    counts, bin_edge = np.histogram(mu_list, bins = 20)

    intensity_mu = np.array([], dtype = float)
    for i in range(len(counts)):
        intensity_mu = np.append(intensity_mu, counts[i] / ((bin_edge[i + 1] - bin_edge[i]) / 2)) # I(mu) proportional to N(mu) / mu

    return intensity_mu

"""
PART B)
"""
"Constants"
N = 1e5 # Number of photons to simulate
TAU_MAX = 10 # Maximum optical depth of the atmosphere

"Simulating N Photons Scattering"
mu_list = np.array([], dtype = float)

for i in range(int(N)):
    steps, mu = random_walk(TAU_MAX)
    mu_list = np.append(mu_list, mu)

"Plotting Photons Escaping With Respective Directional Cosine"
plt.figure()

# Plotting photons escaping with respective directional cosine
plt.hist(mu_list, bins = 20, color = 'Teal')

# Labels
plt.title("Photons Escaping With Respective Directional Cosine Distribution", fontsize = 12)
plt.xlabel("$\mu$", fontsize = 12)
plt.ylabel("$N(\mu)$", fontsize = 12)

plt.grid()

# Limits
plt.xlim([0, 1])

plt.savefig('Figures\\Photons_Escaping_With_Respective_Directional_Cosine_Distribution.pdf')
plt.show()

mu_binned = np.linspace(0, 1, 20) # Binned mu values for plotting
intensity_mu = intensity(mu_list) # Intensity

"Plotting The Analytical vs. Numerical Intensity Ratio"
plt.figure()

# Plotting the analytical vs. numerical intensity ratio
plt.plot(mu_binned, intensity_mu / intensity_mu[-1], ls = 'o', color = 'Teal', label = "Numerical $I(\mu) / I(1)$")
plt.plot(mu_binned, (0.4 + 0.6 * mu_binned), ls = 'o', color = 'Coral', label = "Analytical $I(\mu) / I(1)$")

# Labels
plt.title("The Analytical vs. Numerical Intensity Ratio", fontsize = 12)
plt.xlabel("$\mu$", fontsize = 12)
plt.ylabel("$I(\mu) / I(1)$", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# Limits
plt.xlim([0, 1])
plt.ylim([0, 1.1])

plt.savefig('Figures\\The_Analytical_vs._Numerical_Intensity_Ratio.pdf')
plt.show()
