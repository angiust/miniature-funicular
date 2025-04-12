import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", type=str, default="output.dat", help="name of the input file")
# many more arguments

arguments = parser.parse_args()
filename = arguments.i

print(f"Plot of the following file: {arguments.i}.")


# Load the data, skipping the first line (which starts with '#')
data = np.loadtxt(filename, delimiter=",", comments="#")

# Extract time and magnetization
time = data[:, 0]
magnetization = data[:, 1]
#aligned_neurons = data[:, 2] / 1000

# Plot
plt.figure(figsize=(8, 5))
plt.plot(time, magnetization, marker='o', linestyle='-', color='blue')
#plt.plot(time, aligned_neurons, label="Aligned Neurons", color="green", linestyle='--')
plt.title("Magnetization vs Time")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.tight_layout()
plt.show()
