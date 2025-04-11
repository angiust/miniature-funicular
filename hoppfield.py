import numpy as np
"""
N=1000 # number of neurons
p=8 # number of patterns
t_max=3000 # number of temporal steps
a=0 # parameter of the distribution of probability
T=0 # temperature of the system
s=20 # number of samples, i'll use it later
"""

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    if x == 0:
        return np.random.choice([-1, 1])


def sample_mixture(a): # Sample a random number from the distribution: p(ξ) = a/(2√2) * exp(-|ξ|/√2) + (1-a)/2 * [δ(ξ-1) + δ(ξ+1)]
    u = np.random.rand()
    if u < a:
        return np.random.laplace(loc=0, scale=np.sqrt(2))
    else:
        return np.random.choice([-1, 1])


def extract_pattern(neurons_number, patterns_number, distribution_param):
    patterns=np.zeros((neurons_number, patterns_number))
    for mu in range(patterns_number):
        for i in range(neurons_number):
            patterns[i, mu] = sample_mixture(distribution_param)
    return patterns


def compute_couplings(patterns):
    couplings = patterns @ patterns.T
    np.fill_diagonal(couplings, 0)
    return couplings


def init_net_first_pattern(neurons_number, patterns):
    neurons=np.zeros(neurons_number)
    for i in range(neurons_number):
        neurons[i]=sign(patterns[i,0])
    return neurons


def compute_magn_first_pattern(number_neurons, neurons, patterns):
    magn = 0
    for i in range(number_neurons):
        magn += neurons[i] * patterns[i,0]
    return magn / number_neurons


def run_dynamic(neurons, neurons_number, couplings, steps, temperature):

    return 