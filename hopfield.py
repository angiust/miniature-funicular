import argparse
import numpy as np

# This code implements a Hopfield network with a mixture of distributions for the patterns.
# The network is initialized with a set of patterns, and the neurons are updated based on the couplings between them.
# The patterns are sampled from a mixture of Laplace and delta distributions.
# The network dynamics are run for a specified number of steps, and the magnitude of the first pattern is computed at each step.

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    if x == 0:
        return 2 * (np.random.rand() < 0.5) - 1 # np.random.choice([-1, 1])


def sample_mixture(a): # Sample a random number from the distribution: p(ξ) = a/(2√2) * exp(-|ξ|/√2) + (1-a)/2 * [δ(ξ-1) + δ(ξ+1)]
    u = np.random.rand()
    if u < a:
        return np.random.laplace(loc=0, scale=np.sqrt(2))
    else:
        return 2 * (np.random.rand() < 0.5) - 1 # np.random.choice([-1, 1])


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


def compute_magn_first_pattern(neurons_number, neurons, patterns):
    magn = 0
    for i in range(neurons_number):
        magn += neurons[i] * patterns[i,0]
    return magn / neurons_number


def number_of_neurons_aligned(neurons_number, neurons, patterns):
    aligned = 0
    for i in range(neurons_number):
        if neurons[i] == sign(patterns[i, 0]):
            aligned += 1
    return aligned


def update_neurons(neurons_number, neurons, couplings, temperature):
    neuron_picked = np.random.randint(0, neurons_number)
    local_field = np.sum(couplings[neuron_picked, :] * neurons)
    if temperature == 0:
        neurons[neuron_picked] = sign(local_field)
    else:
        neurons[neuron_picked] = 2 * (np.random.rand() < (1+np.tanh(local_field/temperature))/2) - 1
    return neurons


def run_dynamic(neurons, neurons_number, couplings, steps, temperature):
    for t in range(steps):
        neurons = update_neurons(neurons_number, neurons, couplings, temperature)
        magn = compute_magn_first_pattern(neurons_number, neurons, patterns)
        neurons_aligned = number_of_neurons_aligned(neurons_number, neurons, patterns)
        if t % 100 == 0:
            print(f"Step {t}: Magnetization = {magn} | Aligned Neurons = {neurons_aligned}")
    return neurons


def run_simulation(N, p, t_max, a, T):
    patterns = extract_pattern(N, p, a)
    couplings = compute_couplings(patterns)
    neurons = init_net_first_pattern(N, patterns)
    neurons = run_dynamic(neurons, N, couplings, t_max, T)
    return neurons


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-N", type=int, default=1000, help="number of neurons")
parser.add_argument("-p", type=int, default=8, help="number of patterns")
parser.add_argument("-t", type=int, default=3000, help="number of temporal steps")
parser.add_argument("-a", type=float, default=0, help="parameter of the distribution of probability")
parser.add_argument("-T", type=float, default=0, help="temperature of the system")
parser.add_argument("-s", type=int, default=20, help="number of samples")
# many more arguments

arguments = parser.parse_args()

print(f"And now you can play with these: {arguments}.")

run_simulation(
    N=arguments.N,
    p=arguments.p,
    t_max=arguments.t,
    a=arguments.a,
    T=arguments.T
)


'''
# Example usage 
N=1000 # number of neurons
p=8 # number of patterns
t_max=3000 # number of temporal steps
a=0.7 # parameter of the distribution of probability
T=0.1 # temperature of the system
# s=20 # number of samples, i'll use it later
   
patterns = extract_pattern(N, p, a)
couplings = compute_couplings(patterns)
neurons = init_net_first_pattern(N, patterns)
neurons = run_dynamic(neurons, N, couplings, t_max, T)
"""
print("First pattern:", patterns[:, 0])
print("Final state of neurons:", neurons)
"""
'''