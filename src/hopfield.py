import numpy as np
from typing import Literal, Optional
from itertools import combinations

"""
This code implements a Hopfield network with a mixture of distributions for the patterns.
The network is initialized with a set of patterns, and the neurons are updated based on the couplings between them.
The patterns are sampled from a mixture of Laplace and delta distributions.
The network dynamics are run for a specified number of steps, and the magnitude of the first pattern is computed at each step.
"""


def sign(x):  # i keep it because maybe i'll use it
    if x < 0:
        return -1
    if x > 0:
        return 1
    if x == 0:
        return 2 * (np.random.rand() < 0.5) - 1  # np.random.choice([-1, 1])


def extract_pattern_old(neurons_number, patterns_number, distribution_param):
    """Sample patterns_number random patterns such that
    every component is a number sampled from the distribution:
    p(ξ) = a/(2√2) * exp(-|ξ|/√2) + (1-a)/2 * [δ(ξ-1) + δ(ξ+1)]"""
    size = (neurons_number, patterns_number)
    mask = np.random.rand(*size) < distribution_param
    laplace = np.random.laplace(loc=0, scale=np.sqrt(2), size=size)
    delta = np.random.choice(
        [-1.0, 1.0], size=size
    )  # maybe 2 * (np.random.rand() < 0.5) - 1 it's better
    return np.where(mask, laplace, delta)


def extract_pattern(neurons_number, patterns_number, distribution_param, delta: Optional[bool] = False):
    """Sample patterns_number random patterns.
    If delta=False (default):
        Each component is drawn from:
        p(ξ) = a/(2√2) * exp(-|ξ|/√2) + (1-a)/2 * [δ(ξ-1) + δ(ξ+1)]
    If delta=True:
        Each component is:
            0 with probability 'a',
           -1 or 1 with probability (1-a)/2 each.
    """
    size = (neurons_number, patterns_number)
    mask = np.random.rand(*size) < distribution_param

    if delta:
        delta_values = np.random.choice([-1.0, 1.0], size=size)
        return np.where(mask, 0.0, delta_values) # shape (N, p)
    else:
        laplace = np.random.laplace(loc=0, scale=np.sqrt(2), size=size)
        delta_values = np.random.choice([-1.0, 1.0], size=size)
        return np.where(mask, laplace, delta_values) # shape (N, p)


def compute_couplings(neurons_number, patterns):
    couplings = (patterns @ patterns.T) / neurons_number
    np.fill_diagonal(couplings, 0)
    return couplings


def compute_mixture(patterns):
    if patterns.shape[1] < 3:
        raise ValueError("Need at least 3 patterns to compute this mixture.")
    mixture = patterns[:, 0] + patterns[:, 1] + patterns[:, 2]
    # assert np.all(mixture != 0), "sign of mixture should be non-zero"
    return mixture


def comb_matrix(N, n):
    I = np.eye(N, dtype=int)
    vecs = [np.sum(cols, axis=0) for cols in combinations(I, n)]
    return np.stack(vecs, axis=1)  # shape: (N, C(N, n))


def compute_all_n_mixtures(patterns, n):
    assert patterns.shape[1] >= n, "Need at least n patterns to compute this mixture."
    comb = comb_matrix(patterns.shape[1], n)
    mixtures_continuous = patterns @ comb
    return np.sign(mixtures_continuous)  # shape (N, C(p, n))


def compute_all_three_mixtures(patterns):
    """compute all the three mixtures of the patterns"""
    if patterns.shape[1] < 3:
        raise ValueError("Need at least 3 patterns to compute this mixture.")
    mixtures = []
    for i in range(patterns.shape[1]):
        for j in range(i + 1, patterns.shape[1]):
            for k in range(j + 1, patterns.shape[1]):
                mixtures.append(patterns[:, i] + patterns[:, j] + patterns[:, k])
    return np.array(np.sign(mixtures)).T  # shape (N, (p choose 3))


def magnetization(neurons, patterns):
    N = neurons.shape[0]
    magnetization = (patterns.T @ neurons) / N
    return np.abs(magnetization)


def energy(neurons, couplings):
    return -0.5 * neurons @ couplings @ neurons


def init_net_on_a_pattern(patterns, which_pattern):
    if which_pattern < 0 & patterns.shape[1] < which_pattern + 2:
        raise ValueError("Choosen a pattern that doesn't exist")
    # assert np.all(patterns[:, which_pattern] != 0), ("sign of patterns should be non-zero")
    neurons = np.array([sign(s) for s in patterns[:, which_pattern]], dtype=float)
    return neurons # np.sign(patterns[:, which_pattern])


def init_neurons(patterns, init_type):
    if init_type == "pattern":
        return init_net_on_a_pattern(patterns, 0)
    elif init_type == "mixture":
        if patterns.shape[1] < 3:
            raise ValueError("Need at least 3 patterns to compute this mixture.")
        mixture_vector = compute_mixture(patterns)
        return np.array([sign(s) for s in mixture_vector], dtype=float) # np.sign(mixture_vector)
    elif init_type == "random":
        return np.sign(np.random.rand(patterns.shape[0]) - 0.5)
    else:
        raise ValueError(
            "init_type should be 'pattern', 'mixture' or 'random', not {}".format(
                init_type
            )
        )


def updated_value(temperature, local_field):
    if temperature == 0:
        #assert local_field != 0, "sign of 0 local field is concerning!"
        return sign(local_field) #np.sign(local_field)
    return 2 * (np.random.rand() < (1 + np.tanh(local_field / temperature)) / 2) - 1


def update_neurons(neurons, couplings, temperature):
    for _ in range(neurons.size):
        neuron_picked = np.random.randint(neurons.size)
        local_field = np.dot(couplings[neuron_picked, :], neurons)
        neurons[neuron_picked] = updated_value(temperature, local_field)


def dynamic(neurons, couplings, patterns, sweeps, temperature):
    for _ in range(sweeps):
        update_neurons(neurons, couplings, temperature)
        yield np.append(
            magnetization(neurons, patterns), energy(neurons, couplings)
        )  # shape (p + 1,)


def simulation(
    N, p, sweep_max, a, T, init_type: Literal["pattern", "mixture", "random"], delta: Optional[bool] = False
):
    patterns = extract_pattern(N, p, a, delta)
    couplings = compute_couplings(N, patterns)
    # assert np.max(np.abs(couplings)) < 1.0, "couplings should be less than 1"
    neurons = init_neurons(patterns, init_type)
    if init_type == "mixture":
        mixture = np.sign(compute_mixture(patterns))
        patterns = np.column_stack((patterns, mixture))  # now shape (N, p+1)
        p = p + 1

    return np.fromiter(
        dynamic(neurons, couplings, patterns, sweep_max, T),
        dtype=np.dtype((float, p + 1)),
    )  # shape (sweep_max, p + 2) if mixture, else (sweep_max, p + 1)


def simulation_all_pattern_init(N, p, sweep_max, a, T, delta: Optional[bool] = False):
    """run a simulation with with one sample of the patterns
    and return the magnetization at each sweep and respect
    each pattern"""
    patterns = extract_pattern(N, p, a, delta)
    couplings = compute_couplings(N, patterns)
    story = [
        list(
            dynamic(
                neurons=init_net_on_a_pattern(patterns, i),
                couplings=couplings,
                patterns=patterns,
                sweeps=sweep_max,
                temperature=T,
            )
        )
        for i in range(p)
    ]

    return np.array(story)  # shape (p, sweep_max, p + 1)


def multiple_simulation(
    N, p, sweep_max, a, T, s, init_type: Literal["pattern", "mixture", "random"], delta: Optional[bool] = False
):
    """run different simulation with resampling of the patterns
    and return the average magnetization, it can start from the first pattern
    or from the mixture or from a random pattern and return the average magnetization
    respectevely from the first pattern or from the mixture"""
    sampled_magnetization = np.array(
        [simulation(N, p, sweep_max, a, T, init_type, delta) for _ in range(s)]
    )
    average_magnetization = np.mean(sampled_magnetization, axis=0)
    standard_deviation = np.std(sampled_magnetization, axis=0)
    return np.column_stack(
        (average_magnetization, standard_deviation)
    )  # shape (sweep_max, 2 * (p + 1))


def multiple_simulation_all_story(
    N, p, sweep_max, a, T, s, init_type: Literal["pattern", "mixture", "random"], delta: Optional[bool] = False
): # it returns the last magnetization of each simulation with resampling of the patterns
    """run different simulation with resampling of the patterns
    and return the last magnetization, it can start from the first pattern
    or from the mixture or from a random pattern"""
    return np.array(
        [simulation(N, p, sweep_max, a, T, init_type, delta)[-1] for _ in range(s)]
    )  # shape (s, p + 1)


def multiple_simulation_random(
    N, p, sweep_max, a, T, s, mixture: Optional[bool] = False, delta: Optional[bool] = False
):
    if mixture:
        return np.array(
            [simulation(N, p, sweep_max, a, T, init_type="mixture", delta=delta) for _ in range(s)]
        ) # shape (s, sweep_max, p + 1)
    else:
        return np.array(
            [simulation(N, p, sweep_max, a, T, init_type="random", delta=delta) for _ in range(s)]
        ) # shape (s, sweep_max, p + 1)


def pattern_energy(patterns, couplings):
    return np.array([energy(pattern, couplings) for pattern in patterns.T])  # shape (p,)


def mixture_energy(patterns, couplings):
    if patterns.shape[1] < 3:
        raise ValueError("Need at least 3 patterns to compute this mixture.")
    mixture = compute_all_three_mixtures(patterns)
    return np.array([energy(mixture_vector, couplings) for mixture_vector in mixture.T])  # shape (n,)


def compute_energy(N, p, distrib_param, delta: Optional[bool] = False):
    """compute the energy of the patterns and the mixture of the patterns"""
    patterns = extract_pattern(N, p, distrib_param, delta)
    couplings = compute_couplings(N, patterns)
    patterns_energy = pattern_energy(np.sign(patterns), couplings)
    mixtures_energy = mixture_energy(patterns, couplings)
    return np.concatenate( (patterns_energy, mixtures_energy) )  # shape (p + n)


def varying_a_energy(N, p, delta: Optional[bool] = False):
    """compute the energy of the patterns and the mixture of the patterns
    for different values of a"""
    a_values = np.linspace(0, 1, 10)
    energies = []
    for a in a_values:
        energies.append(compute_energy(N, p, a, delta))
    return np.array(energies)  # shape (10, p + n)


def varying_a_energy_stat(N, p, delta: Optional[bool] = False):
    """compute the energy of the patterns and the mixture of the patterns
    for different values of a"""
    a_values = np.linspace(0, 1, 25)
    energies = []
    for a in a_values:
        energies.append(compute_energy(N, p, a, delta))
    energies = np.array(energies)  # shape: (10, p + n_mixtures)
    energies_patterns = energies[:, :p]       # shape: (10, p)
    energies_mixtures = energies[:, p:]       # shape: (10, n_mixtures)

    stats = {
        "patterns_mean": np.mean(energies_patterns, axis=1),  # shape: (10,)
        "patterns_std": np.std(energies_patterns, axis=1),    # shape: (10,)
        "mixtures_mean": np.mean(energies_mixtures, axis=1),  # shape: (10,)
        "mixtures_std": np.std(energies_mixtures, axis=1),    # shape: (10,)
    }

    return stats