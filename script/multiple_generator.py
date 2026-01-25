import argparse

from generator import generate_dataset, save_dataset

if __name__ == "__main__":
    default_n = default = [10000, 20000, 30000, 40000, 50000],
    parser = argparse.ArgumentParser(description="Generate multiple datasets.")
    parser.add_argument("-fn", default="input", help="CSV datasets file name (prefix)")
    parser.add_argument("-n", type=int, nargs="+", default=default_n, help="List of total points for each dataset")
    parser.add_argument("-c", type=int, default=30, help="Number of clusters per dataset")
    parser.add_argument("-cs", type=float, default=10.0, help="Scale for random cluster centers")
    parser.add_argument("-std", type=float, default=0.3, help="Base cluster standard deviation")
    parser.add_argument("-nr", type=float, default=0.001, help="Noise ratio")
    parser.add_argument("-r", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    for n in args.n:
        file_name = f"{args.fn}_{n}"
        dataset = generate_dataset(
            n=n,
            c=args.c,
            center_scale=args.cs,
            std_scale=args.std,
            noise_ratio=args.nr,
            random_state=args.r
        )
        save_dataset(file_name=file_name, data=dataset)
    print("All datasets generated successfully.")
