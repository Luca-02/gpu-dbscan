import argparse
import csv
import os
from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import make_blobs, make_moons


class ClusterGenerator(ABC):
    """
    Abstract base class for all cluster generators.

    Subclasses must implement the generate() method.
    """

    def __init__(
            self,
            n: int,
            c: int,
            center_scale: float,
            std_scale: float,
            noise_ratio: float,
            random_state: int,
    ):
        self.n = n
        self.c = c
        self.center_scale = center_scale
        self.std_scale = std_scale
        self.noise_ratio = noise_ratio
        self.random_state = random_state

    @abstractmethod
    def generate(
            self,
            size: int,
            center: np.ndarray,
            thickness: float,
            rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generates a cluster of 2D points.

        :param size: Number of points in the cluster
        :param center: 2D coordinates of the cluster center
        :param thickness: Standard deviation or spread of the cluster
        :param rng: Numpy random number generator
        :return: Numpy array of shape (size, 2) representing cluster points
        """
        pass


class CircleGenerator(ClusterGenerator):
    """
    Generates circular clusters.
    """

    def generate(self, size, center, thickness, rng):
        points, _ = make_blobs(
            n_samples=size,
            centers=[center],
            cluster_std=thickness,
            random_state=rng.integers(1e9),
        )
        return points


class EllipseGenerator(ClusterGenerator):
    """
    Generates elliptical clusters with random rotation and scaling.
    """

    def generate(self, size, center, thickness, rng):
        points, _ = make_blobs(
            n_samples=size,
            centers=[[0, 0]],
            cluster_std=thickness,
            random_state=rng.integers(1e9)
        )

        scale_x = rng.uniform(1.5, 4.0)
        scale_y = rng.uniform(0.3, 1.2)
        theta = rng.uniform(0, 2 * np.pi)

        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        stretch = np.array([
            [scale_x, 0],
            [0, scale_y]
        ])

        return points @ stretch @ rotation + center


class BananaGenerator(ClusterGenerator):
    """
    Generates banana-shaped clusters with random rotation.
    """

    def generate(self, size, center, thickness, rng):
        points, _ = make_blobs(
            n_samples=size,
            centers=[[0, 0]],
            cluster_std=thickness,
            random_state=rng.integers(1e9)
        )

        reference_scale = 10.0
        scale_factor = reference_scale / self.center_scale

        curvature = rng.uniform(1.0, 2.0) * scale_factor
        theta = rng.uniform(0, 2 * np.pi)

        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        points[:, 1] += curvature * points[:, 0] ** 2
        return points @ rotation + center


class MoonGenerator(ClusterGenerator):
    """
    Generates moon-shaped clusters with rotation and scaling.
    """

    def generate(self, size, center, thickness, rng):
        reference_scale = 10.0
        scale_factor = reference_scale / self.center_scale

        points, _ = make_moons(
            n_samples=size,
            noise=thickness * 0.3 * scale_factor,
            random_state=rng.integers(1e9)
        )
        scale = rng.uniform(1.0, 2.0) / scale_factor
        theta = rng.uniform(0, 2 * np.pi)

        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        return points @ rotation * scale + center


def gen_noise(n_noise, points, rng):
    """
    Generates uniform noise points covering the bounding box of the existing points.

    :param n_noise: Number of noise points to generate
    :param points: Existing cluster points to define bounding box
    :param rng: Numpy random number generator
    :return: Numpy array of shape (n_noise, 2) representing noise points
    """
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)

    return rng.uniform(
        low=p_min,
        high=p_max,
        size=(n_noise, 2)
    )


def generate_dataset(
        n: int,
        c: int,
        center_scale: float = 10.0,
        std_scale: float = 0.3,
        noise_ratio: float = 0.001,
        random_state: int | None = None,
):
    """
    Generates a 2D dataset with multiple clusters of varying shapes and noise.

    :param n: Total number of points
    :param c: Number of clusters
    :param center_scale: Scale for random cluster center positions
    :param std_scale: Base standard deviation for clusters
    :param noise_ratio: Fraction of points to generate as noise
    :param random_state: Optional random seed for reproducibility
    :return: Numpy array of shape (n, 2) containing the generated dataset
    """
    print(f"Generating dataset with {n} points and {c} clusters...")

    rng = np.random.default_rng(random_state)

    n_noise = int(n * noise_ratio)
    n_clusters = n - n_noise
    base_cluster_size = n_clusters // c
    remaining_cluster_size = n_clusters % c

    centers = rng.uniform(
        low=-center_scale,
        high=center_scale,
        size=(c, 2)
    )

    param = [n, c, center_scale, std_scale, noise_ratio, random_state]

    generators: list[ClusterGenerator] = [
        CircleGenerator(*param),
        EllipseGenerator(*param),
        BananaGenerator(*param),
        MoonGenerator(*param),
    ]

    clusters = []

    for i in range(c):
        gen: ClusterGenerator = rng.choice(generators)
        thickness = rng.uniform(0.8, 1.2) * std_scale
        extra_size = 1 if i < remaining_cluster_size else 0

        points = gen.generate(
            size=base_cluster_size + extra_size,
            center=centers[i],
            thickness=thickness,
            rng=rng
        )

        clusters.append(points)

    points = np.vstack(clusters)
    noise = gen_noise(n_noise, points, rng)

    return np.vstack([points, noise])


def save_dataset(file_name, data, n, c, center_scale, std_scale, noise_ratio):
    """
    Saves a 2D dataset to a CSV file with headers "x,y".

    :param file_name: Name of the dataset file (without extension)
    :param data: Numpy array of shape (n, 2) containing dataset points
    :param n: Total number of points
    :param c: Number of clusters
    :param center_scale: Scale for random cluster center positions
    :param std_scale: Base standard deviation for clusters
    :param noise_ratio: Fraction of points to generate as noise
    """
    folder = "../data_in"
    os.makedirs(folder, exist_ok=True)

    center_scale_str = str(center_scale).replace('.', 'd')
    std_scale_str = str(std_scale).replace('.', 'd')
    noise_ratio_str = str(noise_ratio).replace('.', 'd')

    path = os.path.join(
        folder,
        f"{file_name}_{n}n_{c}c_{center_scale_str}cs_{std_scale_str}std_{noise_ratio_str}nr.csv"
    )
    print(f"Saving in {str(path).replace('\\', '/')}")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(data)


if __name__ == "__main__":
    default_n = 10000,
    parser = argparse.ArgumentParser(description="Generate synthetic 2D dataset.")
    parser.add_argument("-fn", default="input", help="CSV dataset file name (without extension)")
    parser.add_argument("-n", type=int, nargs="+", default=default_n, help="List of total points for each dataset")
    parser.add_argument("-c", type=int, default=30, help="Number of clusters")
    parser.add_argument("-cs", type=float, default=10.0, help="Scale for random cluster centers")
    parser.add_argument("-std", type=float, default=0.3, help="Base cluster standard deviation")
    parser.add_argument("-nr", type=float, default=0.001, help="Noise ratio")
    parser.add_argument("-r", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    for n in args.n:
        dataset = generate_dataset(
            n=n,
            c=args.c,
            center_scale=args.cs,
            std_scale=args.std,
            noise_ratio=args.nr,
            random_state=args.r
        )
        save_dataset(
            file_name=args.fn,
            data=dataset,
            n=n,
            c=args.c,
            center_scale=args.cs,
            std_scale=args.std,
            noise_ratio=args.nr
        )
    print("All datasets generated successfully.")
