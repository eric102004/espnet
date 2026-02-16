import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the EuroSpeech Portugal subset from Hugging Face and save it to disk."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the Hugging Face dataset will be saved",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset revision (branch, tag, or commit)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "google/fleurs",
        "pt_br",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        revision=args.revision,
        trust_remote_code=True,
    )

    dataset.save_to_disk(str(output_dir))
    print(f"Saved EuroSpeech Portugal dataset to: {output_dir}")


if __name__ == "__main__":
    main()
