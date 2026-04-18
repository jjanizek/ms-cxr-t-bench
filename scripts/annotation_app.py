"""Streamlit app for labeling MAIRA-2 temporal reports.

Displays for each (pair, finding):
  - Prior and current chest X-rays, side by side
  - MS-CXR-T ground-truth progression label
  - MIMIC-CheXpert labels for both studies (for the 5 MS-CXR-T findings)
  - MAIRA-2 generated report
  - LLM-extracted structured label ("predicted")
  - Annotation buttons: Agree / Disagree (→ choose correct label) / Skip

Writes back to the annotation_sheet.csv in place after each click.

Run:
    cd /home/joseph/ms-cxr-t-bench
    conda activate ms-cxr-bench
    streamlit run scripts/annotation_app.py
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

SHEET_PATH = REPO_ROOT / "data" / "maira2_judge_validation" / "annotation_sheet.csv"
IMAGES_ROOT = REPO_ROOT / "data" / "raw" / "images"
CHEXPERT_PATH = Path("/data/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv.gz")

# The 5 findings we care about, mapped to CheXpert column names
FINDING_CHEXPERT = {
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
}
CHEXPERT_COLS = list(FINDING_CHEXPERT.values())

LABEL_OPTIONS = ["improving", "stable", "worsening"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_sheet() -> pd.DataFrame:
    df = pd.read_csv(SHEET_PATH, dtype=str)
    for col in ["human_label", "notes"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df


@st.cache_data
def load_chexpert() -> pd.DataFrame | None:
    if not CHEXPERT_PATH.exists():
        return None
    with gzip.open(CHEXPERT_PATH, "rt") as f:
        df = pd.read_csv(f)
    df["study_id"] = df["study_id"].astype(str)
    return df


def study_id_from_dicom(dicom_id: str) -> str:
    """Extract study_id from a path-like dicom_id.

    e.g. 'p10/p10002428/s55758034/3bea0373-...' → '55758034'
    """
    parts = Path(dicom_id).parts
    for p in parts:
        if p.startswith("s") and p[1:].isdigit():
            return p[1:]
    return ""


def image_path_for(dicom_id: str) -> Path:
    """Flat-layout filename: last path component + '.jpg'."""
    return IMAGES_ROOT / (Path(dicom_id).name + ".jpg")


def chexpert_value_to_str(v) -> str:
    if pd.isna(v) or v == "":
        return "—"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if f == 1.0:
        return "✓ present"
    if f == 0.0:
        return "✗ absent"
    if f == -1.0:
        return "? uncertain"
    return str(v)


def chexpert_row(chex_df: pd.DataFrame, dicom_id: str) -> pd.Series | None:
    if chex_df is None:
        return None
    sid = study_id_from_dicom(dicom_id)
    match = chex_df[chex_df["study_id"] == sid]
    if len(match) == 0:
        return None
    return match.iloc[0]


def save_annotation(df: pd.DataFrame, idx: int, human_label: str, notes: str = "") -> None:
    df.at[idx, "human_label"] = human_label
    if notes:
        df.at[idx, "notes"] = notes
    df.to_csv(SHEET_PATH, index=False)
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="MAIRA-2 Judge Validation", layout="wide")

    df = load_sheet().copy()
    chex_df = load_chexpert()

    # --- sidebar: navigation + progress ---
    n = len(df)
    done_mask = df["human_label"].isin(LABEL_OPTIONS + ["skip"])
    n_done = int(done_mask.sum())

    st.sidebar.title("Progress")
    st.sidebar.metric("Annotated", f"{n_done} / {n}")
    st.sidebar.progress(n_done / n if n else 0.0)

    # init index: jump to first unlabeled sample
    if "idx" not in st.session_state:
        unlabeled = df.index[~done_mask].tolist()
        st.session_state.idx = unlabeled[0] if unlabeled else 0

    # Sample picker
    st.sidebar.markdown("### Jump to sample")
    picker_options = [
        f"{i+1:3d}. [{'✓' if df.at[i, 'human_label'] in LABEL_OPTIONS + ['skip'] else ' '}] {df.at[i, 'finding']}"
        for i in range(n)
    ]
    picked = st.sidebar.selectbox(
        "Sample", range(n),
        format_func=lambda i: picker_options[i],
        index=st.session_state.idx,
        key="_picker",
    )
    if picked != st.session_state.idx:
        st.session_state.idx = picked
        st.rerun()

    idx = st.session_state.idx
    row = df.iloc[idx]

    # --- header + nav ---
    col_prev, col_counter, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=(idx == 0), use_container_width=True):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
    with col_counter:
        finding = row["finding"]
        gt = row["ground_truth"] if "ground_truth" in row.index else "—"
        pred = row["predicted"]
        st.markdown(
            f"### Sample {idx+1}/{n}  —  **{finding}**  \n"
            f"MS-CXR-T ground truth: **`{gt}`**  |  LLM judge predicted: **`{pred}`**"
        )
    with col_next:
        if st.button("Next ▶", disabled=(idx == n - 1), use_container_width=True):
            st.session_state.idx = min(n - 1, idx + 1)
            st.rerun()

    st.divider()

    # --- images ---
    img_col_prior, img_col_curr = st.columns(2)
    prior_path = image_path_for(row["previous_dicom_id"])
    curr_path = image_path_for(row["dicom_id"])

    with img_col_prior:
        st.markdown(f"**Prior study** (`s{study_id_from_dicom(row['previous_dicom_id'])}`)")
        if prior_path.exists():
            st.image(Image.open(prior_path), use_container_width=True)
        else:
            st.warning(f"Image missing: {prior_path}")

    with img_col_curr:
        st.markdown(f"**Current study** (`s{study_id_from_dicom(row['dicom_id'])}`)")
        if curr_path.exists():
            st.image(Image.open(curr_path), use_container_width=True)
        else:
            st.warning(f"Image missing: {curr_path}")

    # --- CheXpert labels ---
    st.markdown("#### MIMIC-CheXpert labels (from the original radiology reports)")
    prior_chex = chexpert_row(chex_df, row["previous_dicom_id"])
    curr_chex = chexpert_row(chex_df, row["dicom_id"])

    if prior_chex is None and curr_chex is None:
        st.info("No CheXpert labels available for these studies.")
    else:
        highlight = FINDING_CHEXPERT.get(finding, "")
        rows = []
        for c in CHEXPERT_COLS:
            pv = chexpert_value_to_str(prior_chex[c]) if prior_chex is not None else "—"
            cv = chexpert_value_to_str(curr_chex[c]) if curr_chex is not None else "—"
            marker = "← target" if c == highlight else ""
            rows.append({"Finding": c, "Prior": pv, "Current": cv, "": marker})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- MAIRA-2 report ---
    st.markdown("#### MAIRA-2 generated report")
    st.info(row["maira2_report"])

    # --- Annotation interface ---
    st.markdown("#### Your label")
    st.markdown(
        f"LLM judge extracted: **`{pred}`**. "
        f"Do you agree that this is the correct temporal progression for **{finding}**, "
        f"based on the MAIRA-2 report above?"
    )

    already = row["human_label"]
    if already:
        st.caption(f"Currently labeled: **{already}**")

    notes_key = f"notes_{idx}"
    notes_val = st.text_input("Notes (optional)", value=row["notes"], key=notes_key)

    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
    advance = False

    with btn_col1:
        if st.button(f"✓ Agree ({pred})", type="primary", use_container_width=True):
            save_annotation(df, idx, pred, notes_val)
            advance = True
    with btn_col2:
        if st.button("Disagree → Improving", use_container_width=True,
                     disabled=(pred == "improving")):
            save_annotation(df, idx, "improving", notes_val)
            advance = True
    with btn_col3:
        if st.button("Disagree → Stable", use_container_width=True,
                     disabled=(pred == "stable")):
            save_annotation(df, idx, "stable", notes_val)
            advance = True
    with btn_col4:
        if st.button("Disagree → Worsening", use_container_width=True,
                     disabled=(pred == "worsening")):
            save_annotation(df, idx, "worsening", notes_val)
            advance = True
    with btn_col5:
        if st.button("Skip / Can't tell", use_container_width=True):
            save_annotation(df, idx, "skip", notes_val)
            advance = True

    if advance:
        # Advance to next unlabeled sample if possible
        df2 = load_sheet()
        remaining = df2.index[~df2["human_label"].isin(LABEL_OPTIONS + ["skip"])].tolist()
        if remaining:
            nxt = next((r for r in remaining if r > idx), remaining[0])
            st.session_state.idx = nxt
        else:
            st.session_state.idx = min(n - 1, idx + 1)
        st.rerun()


if __name__ == "__main__":
    main()
