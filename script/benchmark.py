import argparse

import numpy as np
import matplotlib.pyplot as plt

def plot(file_name):
    data = np.genfromtxt(
        f"../benchmark/{file_name}.csv",
        delimiter=",",
        skip_header=1,
        dtype=None,
        encoding=None
    )

    n = np.array([int(row[1]) for row in data])
    cpu_time = np.array([float(row[2]) for row in data])
    gpu_time = np.array([float(row[3]) for row in data])
    speedup = np.array([float(row[4]) for row in data])

    ticks_x = np.arange(100000, 1000001, 100000)

    plt.figure(figsize=(10, 6))
    plt.plot(n, cpu_time, label='CPU Time', marker='o', color='blue')
    plt.plot(n, gpu_time, label='GPU Time', marker='x', color='red')
    plt.title('Execution time: CPU vs GPU')
    plt.xlabel('Problem size (n)')
    plt.ylabel('Time (seconds)')
    plt.xticks(ticks_x, ticks_x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(n, speedup, label='Speedup', marker='s', color='green')
    plt.title('Speedup (CPU Time / GPU Time)')
    plt.xlabel('Problem size (n)')
    plt.ylabel('Speedup')
    plt.xticks(ticks_x, ticks_x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark info.")
    parser.add_argument("-bfn", default="benchmark", help="Benchmark file name (without extension)")
    args = parser.parse_args()
    plot(args.bfn)

