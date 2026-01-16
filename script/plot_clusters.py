from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def plot_clusters(filename):
    data = np.loadtxt(filename)

    x = data[:, 0]
    y = data[:, 1]
    clusters = data[:, 2].astype(int)
    cluster_dict = defaultdict(list)

    for xi, yi, ci in zip(x, y, clusters):
        cluster_dict[ci].append((xi, yi))

    plt.figure(figsize=(8, 8))

    for cluster_id, points in cluster_dict.items():
        points = np.array(points)
        if cluster_id == -1:
            plt.scatter(points[:, 0], points[:, 1], c="black", s=20, label="Outlier")
        else:
            plt.scatter(points[:, 0], points[:, 1], s=20, label=f"Cluster {cluster_id}")

    plt.title("DBSCAN")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    plot_clusters("../data/output.txt")
