import numpy as np

"""
This code implements a Hopfield network with a mixture of distributions for the patterns.
The network is initialized with a set of patterns, and the neurons are updated based on the couplings between them.
The patterns are sampled from a mixture of Laplace and delta distributions.
The network dynamics are run for a specified number of steps, and the magnitude of the first pattern is computed at each step.
"""


def sign(x): # i keep it because maybe i'll use it
    if x < 0:
        return -1
    if x > 0:
        return 1
    if x == 0:
        return 2 * (np.random.rand() < 0.5) - 1  # np.random.choice([-1, 1])


def extract_pattern(neurons_number, patterns_number, distribution_param):
    """Sample patterns_number random patterns such that 
    every component is a number sampled from the distribution:
    p(ξ) = a/(2√2) * exp(-|ξ|/√2) + (1-a)/2 * [δ(ξ-1) + δ(ξ+1)]"""
    size = (neurons_number, patterns_number)
    mask = np.random.rand(*size) < distribution_param
    laplace = np.random.laplace(loc=0, scale=np.sqrt(2), size=size)
    delta = np.random.choice([-1.0, 1.0], size=size) # maybe 2 * (np.random.rand() < 0.5) - 1 it's better
    return np.where(mask, laplace, delta)


def compute_couplings(neurons_number, patterns):
    couplings = (patterns @ patterns.T) / neurons_number
    np.fill_diagonal(couplings, 0)
    return couplings


def compute_mixture(patterns):
    if patterns.shape[1] < 3:
        raise ValueError("Need at least 3 patterns to compute this mixture.")
    mixture = np.sign(patterns[:, 0] + patterns[:, 1] + patterns[:, 2])
    assert np.all(mixture != 0), "sign of mixture should be non-zero"
    return mixture


def mixture_magnetization(neurons, mixture):
    return np.average(neurons * mixture)


def init_net_first_pattern(patterns):
    assert np.all(patterns[:,0] != 0), "sign of patterns should be non-zero"
    return np.sign(patterns[:, 0])


def first_pattern_magnetization(neurons, patterns):
    return np.average(neurons * patterns[:, 0])


def updated_value(temperature, local_field):
    if temperature == 0:
        assert local_field != 0, "sign of 0 local field is concerning!"
        return np.sign(local_field)
    return 2 * (np.random.rand() < (1 + np.tanh(local_field / temperature)) / 2) - 1


def update_neurons(neurons, couplings, temperature):
    for _ in range(neurons.size):
        neuron_picked = np.random.randint(neurons.size)
        local_field = np.dot(couplings[neuron_picked, :], neurons)
        neurons[neuron_picked] = updated_value(temperature, local_field)


def dynamic(neurons, couplings, patterns, steps, temperature):
    for _ in range(steps):
        update_neurons(neurons, couplings, temperature)
        yield first_pattern_magnetization(neurons, patterns)


def mixture_dynamic(neurons, couplings, mixture, sweeps, temperature):
    for _ in range(sweeps):
        update_neurons(neurons, couplings, temperature)
        yield mixture_magnetization(neurons, mixture)


def bare_simulation(N, p, t_max, a, T):
    patterns = extract_pattern(N, p, a)
    couplings = compute_couplings(N, patterns)
    # assert np.max(np.abs(couplings)) < 1.0, "couplings should be less than 1"
    neurons = init_net_first_pattern(patterns)

    return np.fromiter(dynamic(neurons, couplings, patterns, t_max, T), float)


def simulation(N, p, t_max, a, T):
    return bare_simulation(N, p, t_max, a, T)


def mixture_simulation(N, p, sweep_max, a, T):
    patterns = extract_pattern(N, p, a)
    couplings = compute_couplings(N, patterns)
    mixture = compute_mixture(patterns)
    neurons = mixture

    return np.fromiter(mixture_dynamic(neurons, couplings, mixture, sweep_max, T), float)


def multiple_simulation(N, p, t_max, a, T, s):
    """run different simulation with resampling of the patterns
    and return the average magnetization"""
    average_magnetization = np.zeros(t_max)
    sampled_magnetization = np.array([simulation(N, p, t_max, a, T) for _ in range(s)])
    average_magnetization = np.mean(sampled_magnetization, axis=0)
    standard_deviation = np.std(sampled_magnetization, axis=0)
    return np.column_stack((average_magnetization, standard_deviation))


def multiple_mixture_simulation(N, p, sweep_max, a, T, s):
    """run different simulation starting from the mixture,
    with resampling of the patterns and 
    return the average magnetization respect to the mixture"""
    average_magnetization = np.zeros(t_max)
    sampled_magnetization = np.array([mixture_simulation(N, p, sweep_max, a, T) for _ in range(s)])
    average_magnetization = np.mean(sampled_magnetization, axis=0)
    standard_deviation = np.std(sampled_magnetization, axis=0)
    return np.column_stack((average_magnetization, standard_deviation))
