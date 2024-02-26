import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_probability_density_function(x, pdf, x_threshold, label_threshold, probability_label):
    """
    Plot the probability density function and shade the region under the curve.

    Parameters:
    x (array-like): Array of x values.
    pdf (array-like): Array of probability density function values.
    x_threshold (float): Threshold value for shading the region.
    label_threshold (str): Label for the threshold vertical line.
    probability_label (str): Label for the shaded probability region.
    """
    plt.plot(x, pdf, label='PDF')
    plt.axvline(x=x_threshold, color='r', linestyle='--', label=label_threshold)
    plt.fill_between(x, pdf, where=(x < x_threshold), alpha=0.3, label=probability_label)
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'Probability Density Function: {probability_label}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the parameters for the normal distribution
mu = 0
sigma = 1

# Calculate the probability P(x < 1 | N(0, 1))
x = np.linspace(-5, 5, 1000)  # Array of x values
pdf = stats.norm.pdf(x, loc=mu, scale=sigma)  # Probability density function
cdf = stats.norm.cdf(1, loc=mu, scale=sigma)  # Cumulative distribution function

# Plot the probability density function
plot_probability_density_function(x, pdf, 1, 'x = 1', 'P(x < 1 | N(0, 1))')

# Define the parameters for the second normal distribution
mu2 = 175
sigma2 = 3

# Calculate the probability P(x > μ + 2σ | N(175, 3))
x2 = np.linspace(170, 180, 1000)  # Array of x values
pdf2 = stats.norm.pdf(x2, loc=mu2, scale=sigma2)  # Probability density function
cdf2 = 1 - stats.norm.cdf(mu2 + 2 * sigma2, loc=mu2, scale=sigma2)  # Cumulative distribution function

# Plot the probability density function
plot_probability_density_function(x2, pdf2, mu2 + 2 * sigma2, 'x = μ + 2σ', 'P(x > μ + 2σ | N(175, 3))')