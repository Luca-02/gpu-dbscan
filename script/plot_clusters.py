import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def plot_clusters(file_name):
    """
    Plots clustered 2D points from a text file.
    Points with a negative cluster label are treated as outliers
    and plotted in black. All other points are colored by cluster.

    :param file_name: Name of the input file, located in the ../data/ directory
    """
    data = np.loadtxt(f"../data/{file_name}.csv", delimiter=",", skiprows=1)

    x = data[:, 0]
    y = data[:, 1]

    # If the file has 3 column use cluster column, otherwise set default cluster to -1
    if data.shape[1] >= 3:
        clusters = data[:, 2].astype(int)
    else:
        clusters = -1 * np.ones(x.shape, dtype=int)

    cluster_dict = defaultdict(list)

    for xi, yi, ci in zip(x, y, clusters):
        cluster_dict[ci].append((xi, yi))

    plt.figure(figsize=(8, 8))

    for cluster_id, points in cluster_dict.items():
        points = np.array(points)
        if cluster_id < 0:
            plt.scatter(points[:, 0], points[:, 1], c="black", s=1, label="Outlier")
        else:
            plt.scatter(points[:, 0], points[:, 1], s=1, label=f"Cluster {cluster_id}")

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
