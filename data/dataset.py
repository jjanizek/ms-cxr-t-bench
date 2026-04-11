"""PyTorch Dataset for MS-CXR-T image pairs.

Images are stored flat in the images_root dir as {dicom_id}.jpg where
dicom_id is the last path component of the dicom_id column in the CSV
(e.g. "p10/p10002428/s55758034/abc123" → "abc123.jpg").

The CSV is wide format with per-finding progression columns:
  {finding}_progression ∈ {improving, stable, worsening} or NaN (not annotated).

Use `load_labels()` to get a long-format DataFrame suitable for per-finding splits.
"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


FINDINGS = [
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]

LABEL_MAP = {"improving": 0, "stable": 1, "worsening": 2}
# Display names matching CLAUDE.md
FINDING_DISPLAY = {
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
}


def dicom_id_to_filename(dicom_id: str) -> str:
    """Extract just the filename from the full path-like dicom_id column.

    e.g. "p10/p10002428/s55758034/3bea0373-..." → "3bea0373-....jpg"
    """
    return Path(dicom_id).name + ".jpg"


def load_labels(labels_path: str, images_root: Optional[str] = None) -> pd.DataFrame:
    """Load and reshape the wide-format CSV to long format.

    Returns a DataFrame with columns:
        subject_id, study_id, dicom_id, previous_study_id, previous_dicom_id,
        finding, progression, label_quality
    Only rows where the finding is annotated (non-null progression) are kept.

    Args:
        labels_path:  Path to MS_CXR_T_temporal_image_classification_v1.0.0.csv.
        images_root:  If provided, filter to pairs where both images exist on disk.
                      This removes non-AP (PA/lateral) views that were not downloaded.
    """
    import logging
    logger = logging.getLogger(__name__)

    df = pd.read_csv(labels_path)

    records = []
    for finding in FINDINGS:
        prog_col = f"{finding}_progression"
        qual_col = f"{finding}_label_quality"
        if prog_col not in df.columns:
            continue
        sub = df[df[prog_col].notna()].copy()
        sub = sub.rename(columns={prog_col: "progression", qual_col: "label_quality"})
        sub["finding"] = finding
        keep = [
            "subject_id", "study_id", "dicom_id",
            "previous_study_id", "previous_dicom_id",
            "finding", "progression", "label_quality",
        ]
        records.append(sub[[c for c in keep if c in sub.columns]])

    long = pd.concat(records, ignore_index=True)
    long["progression"] = long["progression"].str.lower()
    long["label"] = long["progression"].map(LABEL_MAP)

    if images_root is not None:
        root = Path(images_root)
        available = {p.name for p in root.glob("*.jpg")}

        def both_present(row) -> bool:
            f1 = Path(row["dicom_id"]).name + ".jpg"
            f2 = Path(row["previous_dicom_id"]).name + ".jpg"
            return f1 in available and f2 in available

        before = len(long)
        mask = long.apply(both_present, axis=1)
        long = long[mask].reset_index(drop=True)
        logger.info(
            "Image filter: kept %d / %d (pair, finding) rows "
            "(removed %d with missing AP images)",
            len(long), before, before - len(long),
        )
    return long


class PairDataset(Dataset):
    """Dataset yielding (img1, img2, label) for temporal classification.

    img1 = previous study image (earlier date).
    img2 = current study image (later date).

    Args:
        df:           Long-format DataFrame from load_labels() (or a subset).
        images_root:  Directory containing flat .jpg files.
        transform:    torchvision transforms applied to each image independently.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_root: str,
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, dicom_id: str) -> Image.Image:
        fname = dicom_id_to_filename(dicom_id)
        path = self.images_root / fname
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        row = self.df.iloc[idx]
        img1 = self._load(row["previous_dicom_id"])  # prior (earlier) study
        img2 = self._load(row["dicom_id"])            # current (later) study
        label = int(row["label"])
        return img1, img2, label
