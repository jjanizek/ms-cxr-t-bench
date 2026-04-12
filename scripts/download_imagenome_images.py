"""Download MIMIC-CXR-JPG images needed for Chest ImaGenome training.

Reads data/imagenome_pairs/image_urls.tsv and downloads each image to
/data/imagenome_images/ using 16 parallel threads.

Safe to interrupt and re-run — already-downloaded files are skipped.

Usage:
    python scripts/download_imagenome_images.py
    python scripts/download_imagenome_images.py --workers 32 --dest /data/imagenome_images
"""
import argparse
import getpass
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_one(url: str, dest_path: Path, session: requests.Session) -> tuple[str, bool, str]:
    """Download url → dest_path. Returns (filename, success, error_msg)."""
    fname = dest_path.name
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return fname, True, "skipped"
    try:
        resp = session.get(url, timeout=30, stream=True)
        if resp.status_code == 200:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return fname, True, "ok"
        else:
            return fname, False, f"HTTP {resp.status_code}"
    except Exception as e:
        return fname, False, str(e)


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
                logger.warning("Skipping malformed line: %r", line[:80])
                continue
            url, fname = parts
            entries.append((url, dest / fname))

    # Count already done
    already = sum(1 for _, p in entries if p.exists() and p.stat().st_size > 0)
    logger.info(
        "Total images: %d  Already downloaded: %d  Remaining: %d",
        len(entries), already, len(entries) - already,
    )

    if already == len(entries):
        logger.info("All images already downloaded.")
        return

    password = getpass.getpass(f"PhysioNet password for {args.physionet_user}: ")

    session = requests.Session()
    session.auth = (args.physionet_user, password)
    # Verify credentials with a test request
    test_url, _ = entries[0]
    resp = session.head(test_url, timeout=10)
    if resp.status_code == 401:
        logger.error("Authentication failed — check username/password.")
        sys.exit(1)

    logger.info("Starting downloads with %d workers...", args.workers)
    failures = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, url, path, session): fname
            for url, path in entries
            if not (path.exists() and path.stat().st_size > 0)
        }
        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                fname, success, msg = future.result()
                pbar.update(1)
                completed += 1
                if not success:
                    failures.append((fname, msg))
                    pbar.set_postfix(fails=len(failures))

    total_done = sum(1 for _, p in entries if p.exists() and p.stat().st_size > 0)
    logger.info(
        "Done. %d/%d images in %s. %d failures.",
        total_done, len(entries), dest, len(failures),
    )
    if failures:
        fail_log = dest / "download_failures.txt"
        with open(fail_log, "w") as f:
            for fname, msg in failures:
                f.write(f"{fname}\t{msg}\n")
        logger.warning("Failures written to %s", fail_log)


if __name__ == "__main__":
    main()
