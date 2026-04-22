import argparse


def train(args):
    print("=== FRAG Placeholder: Train Mode ===")
    print(f"Dataset: {args.dataset}")
    print("Training pipeline is under development.")


def test(args):
    print("=== FRAG Placeholder: Test Mode ===")
    print(f"Dataset: {args.dataset}")
    print("Evaluation pipeline is under development.")


def main():
    parser = argparse.ArgumentParser(
        description="FRAG: Fair Retrieval-Augmented Generation for Sequential Recommendation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="Running mode: train or test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="movielens",
        help="Dataset name, e.g., movielens, steam, lastfm, goodreads",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
