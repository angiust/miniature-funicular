"""
plt.figure(figsize=(8, 5))
plt.errorbar(time, average_magnetization, yerr=std_deviation, color='blue', ecolor='lightgray', elinewidth=1, capsize=3)
plt.title(f"Magnetization vs Time {arguments.title}")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot
plt.figure(figsize=(8, 5))
plt.plot(time, magnetization, linestyle='-', color='blue')
plt.title("Magnetization vs Time")
plt.xlabel("Time Step")
plt.ylabel("Magnetization")
plt.grid(True)
plt.tight_layout()
plt.show()
"""