import argparse

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


def plot_clusters(file_name):
    data = np.loadtxt(f"../data/{file_name}.txt")

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
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", default="output")
    args = parser.parse_args()

    plot_clusters(args.fn)
