"""Download datasets for bayes-causal-rec.

Run directly:  python -m bcr.data.download --data-dir data/raw
Or via the installed entry point:  bcr-download --data-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

# Coat Shopping dataset — primary Cornell host, with GitHub mirror fallback
COAT_URLS: dict[str, list[str]] = {
    "train.ascii": [
        "https://www.cs.cornell.edu/~schnabts/mnar/train.ascii",
        "https://raw.githubusercontent.com/usaito/unbiased-implicit-rec-real/master/data/coat/train.ascii",
    ],
    "test.ascii": [
        "https://www.cs.cornell.edu/~schnabts/mnar/test.ascii",
        "https://raw.githubusercontent.com/usaito/unbiased-implicit-rec-real/master/data/coat/test.ascii",
    ],
}
COAT_FILES = list(COAT_URLS.keys())


def download_coat(data_dir: str = "data/raw") -> None:
    """Download Coat train/test rating matrices from Cornell.

    Args:
        data_dir: Directory where raw files will be saved.

    Raises:
        RuntimeError: If any file cannot be downloaded after retries.
    """
    coat_dir = Path(data_dir) / "coat"
    coat_dir.mkdir(parents=True, exist_ok=True)

    for fname in COAT_FILES:
        dest = coat_dir / fname
        if dest.exists():
            print(f"[coat] {fname} already downloaded, skipping.")
            continue

        last_exc: Exception | None = None
        for url in COAT_URLS[fname]:
            print(f"[coat] Trying {url} ...")
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, timeout=30, stream=True, headers=headers)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with (
                    open(dest, "wb") as f,
                    tqdm(total=total, unit="B", unit_scale=True, desc=fname) as bar,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
                print(f"[coat] Saved to {dest}")
                last_exc = None
                break
            except Exception as exc:
                if dest.exists():
                    dest.unlink()
                last_exc = exc
                print(f"[coat]   → failed: {exc}")

        if last_exc is not None:
            raise RuntimeError(
                f"All mirrors failed for {fname}.\n"
                "Please download train.ascii and test.ascii manually from "
                "https://www.cs.cornell.edu/~schnabts/mnar/ "
                f"and place them in {coat_dir}/"
            ) from last_exc


def coat_is_available(data_dir: str = "data/raw") -> bool:
    """Return True if both Coat files are present on disk."""
    coat_dir = Path(data_dir) / "coat"
    return all((coat_dir / f).exists() for f in COAT_FILES)


def main() -> None:
    """CLI entry point: download the Coat dataset, or fall back to synthetic."""
    parser = argparse.ArgumentParser(description="Download datasets.")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    args = parser.parse_args()

    try:
        download_coat(args.data_dir)
        print("\n✓ Coat dataset ready.")
    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        print("Falling back to synthetic MNAR data (see bcr.data.preprocess).")


if __name__ == "__main__":
    main()
