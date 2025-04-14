import numpy as np

"""
This code implements a Hopfield network with a mixture of distributions for the patterns.
The network is initialized with a set of patterns, and the neurons are updated based on the couplings between them.
The patterns are sampled from a mixture of Laplace and delta distributions.
The network dynamics are run for a specified number of steps, and the magnitude of the first pattern is computed at each step.
"""


def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    if x == 0:
        return 2 * (np.random.rand() < 0.5) - 1  # np.random.choice([-1, 1])


def sample_mixture(a):
    """
    Sample a random number from the distribution:
    p(ξ) = a/(2√2) * exp(-|ξ|/√2) + (1-a)/2 * [δ(ξ-1) + δ(ξ+1)]
    """
    u = np.random.rand()
    if u < a:
        return np.random.laplace(loc=0, scale=np.sqrt(2))
    else:
        return 2 * (np.random.rand() < 0.5) - 1  # np.random.choice([-1, 1])


def extract_pattern(neurons_number, patterns_number, distribution_param):
    """Sample a random number from the distribution: p(ξ) = a/(2√2) * exp(-|ξ|/√2) + (1-a)/2 * [δ(ξ-1) + δ(ξ+1)]"""
    mask = np.random.rand(neurons_number, patterns_number) < distribution_param
    laplace = np.random.laplace(loc=0, scale=np.sqrt(2), size=(neurons_number, patterns_number))
    delta = np.random.choice([-1.0, 1.0], size=(neurons_number, patterns_number))
    return np.where(mask, laplace, delta)


def compute_couplings(neurons_number, patterns):
    couplings = (patterns @ patterns.T) / neurons_number
    np.fill_diagonal(couplings, 0)
    return couplings


def init_net_first_pattern(patterns):
    return np.sign(patterns[:, 0])


def first_pattern_magnetization(neurons, patterns):
    return np.average(neurons * patterns[:, 0] * neurons)


def number_of_neurons_aligned(neurons, patterns):
    return np.sum(np.sign(neurons) == np.sign(patterns[:, 0]))


def updated_value(temperature, local_field):
    if temperature == 0:
        return np.sign(local_field)
    return 2 * (np.random.rand() < (1 + np.tanh(local_field / temperature)) / 2) - 1


def update_neurons(neurons, couplings, temperature):
    neuron_picked = np.random.randint(neurons.size)
    local_field = np.dot(couplings[neuron_picked, :], neurons)
    neurons[neuron_picked] = updated_value(temperature, local_field)
    return neurons


def dynamic(neurons, couplings, patterns, steps, temperature):
    magnetizations = []
    for t in range(steps):
        neurons = update_neurons(neurons, couplings, temperature)
        magnetization = first_pattern_magnetization(neurons, patterns)
        magnetizations.append(magnetization)
    return magnetizations


def bare_simulation(N, p, t_max, a, T):
    patterns = extract_pattern(N, p, a)
    couplings = compute_couplings(N, patterns)
    neurons = init_net_first_pattern(patterns)

    return dynamic(neurons, couplings, patterns, t_max, T)


def wrap_into_array(t_max, magnetizations):
    dtype = [("t", int), ("magnetization", float)]
    array = np.empty(t_max, dtype=dtype)
    array["t"] = range(t_max)
    array["magnetization"] = magnetizations
    return array


def simulation(N, p, t_max, a, T):
    evolution = bare_simulation(N, p, t_max, a, T)
    return wrap_into_array(t_max, evolution)


def multiple_simulation(N, p, t_max, a, T, s):
    """run different simulation with resampling"""
    return [simulation(N, p, t_max, a, T) for _ in range(s)]
