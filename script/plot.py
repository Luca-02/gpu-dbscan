import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot(x, y, clusters=None, title="Space"):
    """
    Plots 2D points colored by cluster.
    Points with cluster label <= 0 are treated as outliers and colored black.

    :param x: Numpy array of x-coordinates
    :param y: Numpy array of y-coordinates
    :param clusters: Optional numpy array of cluster labels; if None, all points treated as outliers
    :param title: Title of the plot
    """
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
            plt.scatter(points[:, 0], points[:, 1], c="black", s=0.1, label="Outlier")
            legend_elements.append(
                Line2D([0], [0], marker='o', color='black',
                       linestyle='None', markersize=10, label="Outlier")
            )
        else:
            sc = plt.scatter(points[:, 0], points[:, 1], s=0.1, label=f"Cluster {cluster_id}")
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


def read_dataset(file_name):
    """
    Reads a CSV dataset and returns a numpy array (n,2) or (n,3) if cluster column exists.

    :param file_name: Name of the dataset file (without extension)
    :return: Numpy array of shape (n,2) or (n,3)
    """
    return np.loadtxt(f"../data/{file_name}.csv", delimiter=",", skiprows=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D dataset.")
    parser.add_argument("-fn", default="output", help="CSV dataset file name (without extension)")
    args = parser.parse_args()

    data = read_dataset(file_name=args.fn)
    plot(
        x=data[:, 0],
        y=data[:, 1],
        clusters=data[:, 2] if data.shape[1] >= 3 else None,
        title=args.fn
    )
