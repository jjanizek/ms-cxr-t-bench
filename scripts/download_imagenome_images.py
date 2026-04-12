"""Download MIMIC-CXR-JPG images needed for Chest ImaGenome training.

Reads data/imagenome_pairs/image_urls.tsv and downloads each image to
/data/imagenome_images/ using 16 parallel wget subprocesses.

wget handles PhysioNet's auth redirects correctly (requests does not).
Safe to interrupt and re-run — already-downloaded files are skipped (-nc).

Usage:
    python scripts/download_imagenome_images.py
    python scripts/download_imagenome_images.py --workers 24 --dest /data/imagenome_images
"""
import argparse
import getpass
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_one(url: str, dest_path: Path, user: str, password: str) -> tuple[str, bool]:
    """Download url → dest_path via wget. Returns (filename, success)."""
    fname = dest_path.name
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return fname, True  # already done

    result = subprocess.run(
        [
            "wget", "-q",
            f"--user={user}",
            f"--password={password}",
            "-O", str(dest_path),
            url,
        ],
        capture_output=True,
    )
    if result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0:
        return fname, True
    # Clean up zero-byte file on failure
    if dest_path.exists() and dest_path.stat().st_size == 0:
        dest_path.unlink()
    return fname, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_list", default="data/imagenome_pairs/image_urls.tsv")
    parser.add_argument("--dest", default="/data/imagenome_images")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--physionet_user", default="jjanizek")
    args = parser.parse_args()

    url_list = Path(args.url_list)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Read URL list
    entries = []
    with open(url_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            url, fname = parts
            entries.append((url, dest / fname))

    already = sum(1 for _, p in entries if p.exists() and p.stat().st_size > 0)
    remaining = len(entries) - already
    logger.info("Total: %d  Already done: %d  Remaining: %d", len(entries), already, remaining)

    if remaining == 0:
        logger.info("All images already downloaded.")
        return

    password = getpass.getpass(f"PhysioNet password for {args.physionet_user}: ")

    # Quick auth test
    test_url, test_dest = next((u, p) for u, p in entries if not (p.exists() and p.stat().st_size > 0))
    test_result = subprocess.run(
        ["wget", "-q", "--spider", f"--user={args.physionet_user}", f"--password={password}", test_url],
        capture_output=True,
    )
    if test_result.returncode != 0:
        logger.error("Auth test failed (exit %d). Check your password.", test_result.returncode)
        logger.error("wget stderr: %s", test_result.stderr.decode()[:200])
        return

    logger.info("Auth OK. Downloading with %d parallel workers...", args.workers)
    failures = []

    todo = [(url, path) for url, path in entries if not (path.exists() and path.stat().st_size > 0)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, url, path, args.physionet_user, password): path.name
            for url, path in todo
        }
        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                fname, success = future.result()
                pbar.update(1)
                if not success:
                    failures.append(fname)
                    pbar.set_postfix(fails=len(failures))

    total_done = sum(1 for _, p in entries if p.exists() and p.stat().st_size > 0)
    logger.info("Done. %d/%d images in %s. %d failures.", total_done, len(entries), dest, len(failures))

    if failures:
        fail_log = dest / "download_failures.txt"
        with open(fail_log, "w") as f:
            f.write("\n".join(failures) + "\n")
        logger.warning("%d failures written to %s", len(failures), fail_log)


if __name__ == "__main__":
    main()
