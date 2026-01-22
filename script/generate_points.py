import argparse
import csv
import math
import random


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
    :return: List of (x, y) points forming a banana-shaped cluster
    """
    angles = [random.uniform(angle_start, angle_end) for _ in range(n)]
    radii = [radius + random.gauss(0, thickness) for _ in range(n)]
    return [(cx + r * math.cos(theta), cy + r * math.sin(theta)) for r, theta in zip(radii, angles)]


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
    :return: List of (x, y) points forming an elliptical cluster
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
        points.append((cx + xr, cy + yr))
    return points


def generate_elongated_cluster(cx, cy, length, thickness, angle, n):
    """
    Generates an elongated cluster of points.
    Points are distributed along a main axis with Gaussian noise
    applied perpendicularly, then rotated and translated.

    :param cx: X coordinate of the cluster center
    :param cy: Y coordinate of the cluster center
    :param length: Total length of the cluster
    :param thickness: Standard deviation of perpendicular noise
    :param angle: Rotation angle of the cluster
    :param n: Number of points to generate
    :return: List of (x, y) points forming an elongated cluster
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ts = [random.uniform(-length / 2, length / 2) for _ in range(n)]
    offsets = [random.gauss(0, thickness) for _ in range(n)]
    return [(cx + t * cos_a - o * sin_a, cy + t * sin_a + o * cos_a) for t, o in zip(ts, offsets)]


def generate_noise(n, xmin, xmax, ymin, ymax):
    """
    Generates uniformly distributed random noise points.

    :param n: Number of noise points
    :param xmin: Minimum x coordinate
    :param xmax: Maximum x coordinate
    :param ymin: Minimum y coordinate
    :param ymax: Maximum y coordinate
    :return: List of (x, y) noise points
    """
    xs = [random.uniform(xmin, xmax) for _ in range(n)]
    ys = [random.uniform(ymin, ymax) for _ in range(n)]
    return list(zip(xs, ys))


def random_center(xmin, xmax, ymin, ymax, margin=10):
    """
    Generates a random center point inside given bounds,
    keeping a margin from the borders.

    :param xmin: Minimum x coordinate
    :param xmax: Maximum x coordinate
    :param ymin: Minimum y coordinate
    :param ymax: Maximum y coordinate
    :param margin: Minimum distance from the borders
    :return: Tuple (cx, cy) representing the center point
    """
    return random.uniform(xmin + margin, xmax - margin), random.uniform(ymin + margin, ymax - margin)


def generate_random_cluster(xmin, xmax, ymin, ymax, scale=1.0):
    """
    Generates a randomly selected cluster type within the given bounds.

    :param xmin: Minimum x coordinate of the dataset
    :param xmax: Maximum x coordinate of the dataset
    :param ymin: Minimum y coordinate of the dataset
    :param ymax: Maximum y coordinate of the dataset
    :param scale: Scaling factor for cluster size and density
    :return: List of (x, y) points forming the cluster
    """
    cx, cy = random_center(xmin, xmax, ymin, ymax)

    cluster_type = random.choice(
        ["banana", "ellipse", "elongated", "compact"]
    )

    if cluster_type == "banana":
        radius = random.uniform(4, 8) * scale
        thickness = random.uniform(0.2, 0.8) * scale
        angle_start = random.uniform(-math.pi, 0)
        angle_end = angle_start + random.uniform(2, 3)
        n = int(random.randint(300, 800) * scale)
        return generate_banana_cluster(cx, cy, radius, angle_start, angle_end, thickness, n)

    elif cluster_type == "ellipse":
        a = random.uniform(2, 6) * scale
        b = random.uniform(2, 5) * scale
        angle = random.uniform(0, math.pi)
        n = int(random.randint(300, 600) * scale)
        return generate_elliptical_cluster(cx, cy, a, b, angle, n)

    elif cluster_type == "elongated":
        length = random.uniform(12, 18) * scale
        thickness = random.uniform(0.3, 0.6) * scale
        angle = random.uniform(-math.pi / 2, math.pi / 2)
        n = int(random.randint(300, 800) * scale)
        return generate_elongated_cluster(cx, cy, length, thickness, angle, n)

    a = random.uniform(1, 3) * scale
    b = random.uniform(1, 3) * scale
    angle = random.uniform(0, math.pi)
    n = int(random.randint(200, 300) * scale)
    return generate_elliptical_cluster(cx, cy, a, b, angle, n)


def generate_dataset(file_name, xmin, xmax, ymin, ymax, n_cluster=20):
    """
    Generates a synthetic dataset composed of multiple clusters
    and background noise, then saves it to a file.

    :param file_name: Name of the output file
    :param xmin: Minimum x coordinate
    :param xmax: Maximum x coordinate
    :param ymin: Minimum y coordinate
    :param ymax: Maximum y coordinate
    :param n_cluster: Number of clusters to generate
    :return: None
    """
    points = []
    scale = max(xmax - xmin, ymax - ymin) / 50

    for _ in range(n_cluster):
        points += generate_random_cluster(xmin, xmax, ymin, ymax, scale)

    if points:
        x_vals, y_vals = zip(*points)
        xmin = min(xmin, min(x_vals))
        xmax = max(xmax, max(x_vals))
        ymin = min(ymin, min(y_vals))
        ymax = max(ymax, max(y_vals))

    area = (xmax - xmin) * (ymax - ymin)
    n_noise = int(area * 0.0001)
    points += generate_noise(n_noise, xmin, xmax, ymin, ymax)

    with open(f"../data/{file_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", default="input")
    parser.add_argument("-xmin", type=float, default=-60)
    parser.add_argument("-xmax", type=float, default=60)
    parser.add_argument("-ymin", type=float, default=-60)
    parser.add_argument("-ymax", type=float, default=60)
    parser.add_argument("-nc", type=int, default=20)
    args = parser.parse_args()
    generate_dataset(args.fn, args.xmin, args.xmax, args.ymin, args.ymax, args.nc)
