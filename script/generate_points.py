import argparse
import csv
import math
import random

import numpy as np


def generate_banana_cluster(cx, cy, radius, angle_start, angle_end, thickness, n):
    """
    Generates a curved cluster of points.
    Points are sampled along a circular arc between two angles.
    Gaussian noise is applied to the radius to control the thickness
    of the cluster.

    :param cx: X coordinate of the cluster center
    :param cy: Y coordinate of the cluster center
    :param radius: Base radius of the arc
    :param angle_start: Starting angle of the arc
    :param angle_end: Ending angle of the arc
    :param thickness: Standard deviation of radial noise
    :param n: Number of points to generate
    :return: List of [x, y] points forming a banana-shaped cluster
    """
    angles = [random.uniform(angle_start, angle_end) for _ in range(n)]
    radii = [radius + random.gauss(0, thickness) for _ in range(n)]
    return [[cx + r * math.cos(theta), cy + r * math.sin(theta)] for r, theta in zip(radii, angles)]


def generate_elliptical_cluster(cx, cy, a, b, angle, n):
    """
    Generates an elliptical cluster of points with rotation.
    Points are sampled from an ellipse, scaled by Gaussian noise,
    rotated by a given angle, and translated to the desired center.

    :param cx: X coordinate of the cluster center
    :param cy: Y coordinate of the cluster center
    :param a: Semi-major axis of the ellipse
    :param b: Semi-minor axis of the ellipse
    :param angle: Rotation angle of the ellipse
    :param n: Number of points to generate
    :return: List of [x, y] points forming an elliptical cluster
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ts = [random.uniform(0, 2 * math.pi) for _ in range(n)]
    rs = [random.gauss(1, 0.1) for _ in range(n)]
    points = []
    for t, r in zip(ts, rs):
        x = a * r * math.cos(t)
        y = b * r * math.sin(t)
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        points.append([cx + xr, cy + yr])
    return points


def generate_elongated_cluster(cx, cy, length, angle, thickness, n):
    """
    Generates an elongated cluster of points.
    Points are distributed along a main axis with Gaussian noise
    applied perpendicularly, then rotated and translated.

    :param cx: X coordinate of the cluster center
    :param cy: Y coordinate of the cluster center
    :param length: Total length of the cluster
    :param angle: Rotation angle of the cluster
    :param thickness: Standard deviation of perpendicular noise
    :param n: Number of points to generate
    :return: List of [x, y] points forming an elongated cluster
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ts = [random.uniform(-length / 2, length / 2) for _ in range(n)]
    offsets = [random.gauss(0, thickness) for _ in range(n)]
    return [[cx + t * cos_a - o * sin_a, cy + t * sin_a + o * cos_a] for t, o in zip(ts, offsets)]


def generate_noise(n, xmin, xmax, ymin, ymax):
    """
    Generates uniformly distributed random noise points.

    :param n: Number of noise points
    :param xmin: Minimum x coordinate
    :param xmax: Maximum x coordinate
    :param ymin: Minimum y coordinate
    :param ymax: Maximum y coordinate
    :return: List of [x, y] noise points
    """
    xs = [random.uniform(xmin, xmax) for _ in range(n)]
    ys = [random.uniform(ymin, ymax) for _ in range(n)]
    return [[x, y] for x, y in zip(xs, ys)]


def random_center(xmin, xmax, ymin, ymax, margin=10):
    """
    Generates a random center point inside given bounds,
    keeping a margin from the borders.

    :param xmin: Minimum x coordinate
    :param xmax: Maximum x coordinate
    :param ymin: Minimum y coordinate
    :param ymax: Maximum y coordinate
    :param margin: Minimum distance from the borders
    :return: Tuple [cx, cy] representing the center point
    """
    return random.uniform(xmin + margin, xmax - margin), random.uniform(ymin + margin, ymax - margin)


def generate_random_cluster(xmin, xmax, ymin, ymax, n, scale=1.0):
    """
    Generates a randomly selected cluster type within the given bounds.

    :param xmin: Minimum x coordinate of the dataset
    :param xmax: Maximum x coordinate of the dataset
    :param ymin: Minimum y coordinate of the dataset
    :param ymax: Maximum y coordinate of the dataset
    :param n: Number of points to generate
    :param scale: Scaling factor for cluster size and density
    :return: List of [x, y] points forming the cluster
    """
    cx, cy = random_center(xmin, xmax, ymin, ymax)

    cluster_type = random.choice(
        ["banana", "ellipse", "elongated", "compact"]
    )

    if cluster_type == "banana":
        radius = random.uniform(4, 8) * scale
        thickness = random.uniform(0.2, 0.4) * scale
        angle_start = random.uniform(-math.pi, 0)
        angle_end = angle_start + random.uniform(2, 3)
        return generate_banana_cluster(cx, cy, radius, angle_start, angle_end, thickness, n)

    elif cluster_type == "ellipse":
        a = random.uniform(2, 4) * scale
        b = random.uniform(2, 4) * scale
        angle = random.uniform(0, math.pi)
        return generate_elliptical_cluster(cx, cy, a, b, angle, n)

    elif cluster_type == "elongated":
        length = random.uniform(8, 12) * scale
        thickness = random.uniform(0.2, 0.4) * scale
        angle = random.uniform(-math.pi / 2, math.pi / 2)
        return generate_elongated_cluster(cx, cy, length, angle, thickness, n)

    a = random.uniform(1, 3) * scale
    b = random.uniform(1, 3) * scale
    angle = random.uniform(0, math.pi)
    return generate_elliptical_cluster(cx, cy, a, b, angle, n)


def uniform_cluster_sizes(total_points, n_clusters, noise_frac=0.001, jitter_frac=0.05):
    noise_points = int(total_points * noise_frac)
    cluster_points = total_points - noise_points

    base = cluster_points // n_clusters
    jitter = int(base * jitter_frac)

    sizes = [
        base + random.randint(-jitter, jitter)
        for _ in range(n_clusters)
    ]

    # Add remaining points to the first cluster
    diff = cluster_points - sum(sizes)
    sizes[0] += diff

    return sizes, noise_points


def generate_dataset(total_points, n_clusters):
    """
    Generates a synthetic dataset composed of multiple clusters
    and background noise, then saves it to a file.

    :param total_points: Total number of points to generate
    :param n_clusters: Number of clusters to generate
    :return: Generated dataset as a numpy array
    """
    scale = math.sqrt(total_points)
    xmin, xmax = -10 * scale, 10 * scale
    ymin, ymax = -10 * scale, 10 * scale

    cluster_sizes, noise_points = uniform_cluster_sizes(total_points, n_clusters)

    points = []

    for size in cluster_sizes:
        cluster = generate_random_cluster(xmin, xmax, ymin, ymax, size, scale)
        points.extend(cluster)

    if points:
        x_vals, y_vals = zip(*points)
        xmin = min(xmin, min(x_vals))
        xmax = max(xmax, max(x_vals))
        ymin = min(ymin, min(y_vals))
        ymax = max(ymax, max(y_vals))

    points += generate_noise(noise_points, xmin, xmax, ymin, ymax)

    return np.array(points)


def save_dataset(file_name, points):
    """
    Saves the dataset to a CSV file.

    :param file_name: Name of the output file
    :param points: Generated dataset as a numpy array
    """
    with open(f"../data/{file_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", default="input")
    parser.add_argument("-n", type=int, default=100000)
    parser.add_argument("-c", type=int, default=5)
    args = parser.parse_args()
    data = generate_dataset(args.n, args.c)
    save_dataset(args.fn, data)
