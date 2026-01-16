import random


def generate_cluster(cx, cy, n, spread):
    return [
        (random.gauss(cx, spread), random.gauss(cy, spread))
        for _ in range(n)
    ]


def generate_noise(n, xmin, xmax, ymin, ymax):
    return [
        (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
        for _ in range(n)
    ]


def generate_dataset():
    points = []
    points += generate_cluster(0, 0, 100, 0.3)
    points += generate_cluster(5, 5, 120, 0.4)
    points += generate_cluster(-4, 4, 80, 0.2)
    points += generate_noise(40, -10, 10, -10, 10)

    with open("../data/input.txt", "w") as f:
        for x, y in points:
            f.write(f"{x} {y}\n")


if __name__ == "__main__":
    generate_dataset()
