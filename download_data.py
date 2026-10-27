"""Fetch the 4 EMNIST Letters files into DATA_DIR if they are not already there."""
import os
import urllib.request
from pathlib import Path

DATA_DIR = Path(os.environ.get("EMNIST_DIR", "data"))

FILES = [
    "emnist-letters-train-images-idx3-ubyte.gz",
    "emnist-letters-train-labels-idx1-ubyte.gz",
    "emnist-letters-test-images-idx3-ubyte.gz",
    "emnist-letters-test-labels-idx1-ubyte.gz",
]

MIRROR = "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main/{}"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (DATA_DIR / f).exists()]
    if not missing:
        print(f"All 4 EMNIST Letters files already present in {DATA_DIR}/")
        return

    failed = []
    for name in missing:
        url = MIRROR.format(name)
        dest = DATA_DIR / name
        print(f"Downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"  failed: {exc}")
            failed.append(name)

    if failed:
        print(
            "\nCould not download: " + ", ".join(failed) +
            "\nPlease place these files manually in the data/ folder "
            "(official source: https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip, files under gzip/)."
        )
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
