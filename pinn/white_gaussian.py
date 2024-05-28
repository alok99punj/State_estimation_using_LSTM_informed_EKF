import numpy as np
def white_gaussian(std_dev):
    # Define parameters
    num_samples = 1  # Number of samples
    mean = 0            # Mean of the Gaussian distribution
             # Standard deviation of the Gaussian distribution

    # Generate white Gaussian noise
    noise = np.random.normal(mean, std_dev, num_samples)
    return noise 