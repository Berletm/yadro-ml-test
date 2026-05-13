import matplotlib.pyplot as plt
import numpy as np
import argparse
import ast

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finds point with min euclidean dist to other points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-o", "--output", type=str, default="result.png",
                        help="Path for saving plot.")
    return parser.parse_args()

def parse_and_validate(input_str: str) -> np.ndarray:
    try:
        data = ast.literal_eval(input_str)
        if not isinstance(data, list):
            raise TypeError("Expected List object.")
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in data):
            raise ValueError("Every element must be a tuple of 2 elements.")
        return np.array(data, dtype=float)
    except (SyntaxError, ValueError, TypeError) as e:
        raise RuntimeError(f"Wrong format: {e}")

def solve(args: argparse.Namespace) -> None:
    points_str = input("Input points:")
    points = parse_and_validate(points_str)
    arr = np.array(points)
    centroid = np.mean(arr, axis=0)

    print(centroid)
    plt.scatter(arr[:, 0], arr[:, 1], c="blue")
    plt.scatter(*centroid, c="red", s=100)
    plt.savefig(args.output)

def main() -> None:
    args = parse_arguments()
    solve(args)



if __name__ == "__main__":
    main()