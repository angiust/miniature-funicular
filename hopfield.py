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
    mixture = (patterns[:, 0] + patterns[:, 1] + patterns[:, 2])
    assert np.all(mixture != 0), "sign of mixture should be non-zero"
    return mixture


def magnetization(neurons, patterns):
    return np.average(neurons[:, None] * patterns, axis=0)


def init_net_first_pattern(patterns):
    assert np.all(patterns[:,0] != 0), "sign of patterns should be non-zero"
    return np.sign(patterns[:, 0])


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


def dynamic(neurons, couplings, patterns, sweeps, temperature):
    for _ in range(sweeps):
        update_neurons(neurons, couplings, temperature)
        yield magnetization(neurons, patterns)


def simulation(N, p, sweep_max, a, T, mixture):
    """run a simulation with with one sample of the patterns
    and return the magnetization at each sweep and respect
    each pattern"""
    patterns = extract_pattern(N, p, a)
    couplings = compute_couplings(N, patterns)
    # assert np.max(np.abs(couplings)) < 1.0, "couplings should be less than 1"
    if mixture:
        mixture_vector = compute_mixture(patterns)
        neurons = np.sign(mixture_vector)
    else:
        neurons = init_net_first_pattern(patterns)

    return np.fromiter(dynamic(neurons, couplings, patterns, sweep_max, T), dtype = np.dtype((float, p)))


def multiple_simulation(N, p, sweep_max, a, T, s, mixture):
    """run different simulation with resampling of the patterns
    and return the average magnetization, it can start from the first pattern
    or from the mixture and return the average magnetization respectevely 
    from the first pattern or from the mixture"""
    sampled_magnetization = np.array([simulation(N, p, sweep_max, a, T, mixture) for _ in range(s)])
    average_magnetization = np.mean(sampled_magnetization, axis=0)
    standard_deviation = np.std(sampled_magnetization, axis=0)
    return np.column_stack((average_magnetization, standard_deviation))
