import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot(title, x, y, clusters=None):
    if clusters is None:
        clusters = -1 * np.ones(x.shape, dtype=int)

    cluster_dict = defaultdict(list)

    for xi, yi, ci in zip(x, y, clusters):
        cluster_dict[ci].append((xi, yi))

    plt.figure(title, figsize=(8, 8))

    legend_elements = []
    for cluster_id, points in cluster_dict.items():
        points = np.array(points)
        if cluster_id <= 0:
            plt.scatter(points[:, 0], points[:, 1], c="black", s=1, label="Outlier")
            legend_elements.append(
                Line2D([0], [0], marker='o', color='black',
                       linestyle='None', markersize=10, label="Outlier")
            )
        else:
            sc = plt.scatter(points[:, 0], points[:, 1], s=1, label=f"Cluster {cluster_id}")
            legend_elements.append(
                Line2D([0], [0], marker='o', color=sc.get_facecolor()[0],
                       linestyle='None', markersize=10, label=f"Cluster {cluster_id}")
            )

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    # plt.legend(
    #     handles=legend_elements,
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.5),
    #     borderaxespad=0
    # )
    # plt.tight_layout()
    plt.show()


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
        plot(file_name, x, y, data[:, 2].astype(int))
    else:
        plot(file_name, x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", default="output")
    args = parser.parse_args()
    plot_clusters(args.fn)
