import s3fs
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE           = 128
T_PAST             = 13
T_FUTURE           = 12
TOTAL_FRAMES       = T_PAST + T_FUTURE   # 25
START_FRAME        = 12   # which frame in the 49-frame sequence to start from
NUM_EVENTS_TO_FETCH = 5   # increase to get more events for the demo


# ─── Raw SEVIR layout reference ───────────────────────────────────────────────
# Bulk HDF5 file shape:  (N_events, H, W, T)
#   N_events ≈ varies per file
#   H = W = 384  (for VIL)  or  192 / 208 (for some IR products)
#   T = 49  (one frame every ~5 min over ~4 hours)
# After indexing one event:  (H, W, T)  →  need to transpose to (T, H, W)

def _normalise_s3_path(file_name: str) -> str:
    """Ensure the path starts with  data/  as required by the S3 bucket layout."""
    file_name = str(file_name).strip().lstrip("/")
    if not file_name.startswith("data/"):
        file_name = "data/" + file_name
    return file_name


def fetch_event_from_s3(event_id: str, modality: str,
                         catalog: pd.DataFrame,
                         fs: s3fs.S3FileSystem) -> np.ndarray:
    """
    Download one event for one modality from the public SEVIR S3 bucket.

    Returns
    -------
    np.ndarray  shape (TOTAL_FRAMES, IMG_SIZE, IMG_SIZE)  dtype float32
    """
    meta = catalog[(catalog["id"] == event_id) &
                   (catalog["img_type"] == modality)]
    if meta.empty:
        raise ValueError(f"Event '{event_id}' / modality '{modality}' "
                         f"not found in catalog.")
    meta     = meta.iloc[0]
    s3_path  = "s3://sevir/" + _normalise_s3_path(meta["file_name"])
    h5_idx   = int(meta["file_index"])

    print(f"  [{modality.upper():5s}] {s3_path}  (event index {h5_idx}) … ", end="", flush=True)

    with fs.open(s3_path, "rb") as raw:
        with h5py.File(raw, "r") as hf:
            # ── Shape of dataset: (N_events, H, W, T) ────────────────────
            ds    = hf[modality]
            shape = ds.shape          # e.g. (1649, 384, 384, 49)

            if len(shape) == 4:
                # Most common case: (N, H, W, T)
                raw_frame = ds[h5_idx]          # → (H, W, T)
                arr = raw_frame.transpose(2, 0, 1).astype(np.float32)  # (T, H, W)

            elif len(shape) == 3:
                # Occasionally the event axis is already stripped: (H, W, T)
                arr = ds[:].transpose(2, 0, 1).astype(np.float32)      # (T, H, W)

            else:
                raise RuntimeError(f"Unexpected dataset shape {shape} "
                                   f"for modality '{modality}'")

    T, H, W = arr.shape
    print(f"raw shape (T,H,W) = {arr.shape}", flush=True)

    # ── Temporal slice: grab TOTAL_FRAMES consecutive frames ──────────────
    t_start = min(START_FRAME, max(0, T - TOTAL_FRAMES))
    t_end   = t_start + TOTAL_FRAMES
    if t_end > T:
        # Pad with the last frame if the sequence is shorter than expected
        pad    = t_end - T
        arr    = np.concatenate([arr[t_start:], np.repeat(arr[[-1]], pad, axis=0)], axis=0)
    else:
        arr = arr[t_start:t_end]          # (TOTAL_FRAMES, H, W)

    # ── Spatial center-crop to IMG_SIZE × IMG_SIZE ────────────────────────
    _, H, W = arr.shape
    cy, cx  = H // 2, W // 2
    half    = IMG_SIZE // 2

    # Guard: if the spatial size is already ≤ IMG_SIZE, pad instead of crop
    if H < IMG_SIZE or W < IMG_SIZE:
        pad_h = max(0, IMG_SIZE - H)
        pad_w = max(0, IMG_SIZE - W)
        arr   = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        _, H, W = arr.shape
        cy, cx  = H // 2, W // 2

    arr_cropped = arr[:, cy - half: cy + half,
                         cx - half: cx + half]    # (TOTAL_FRAMES, 128, 128)

    assert arr_cropped.shape == (TOTAL_FRAMES, IMG_SIZE, IMG_SIZE), \
        f"Unexpected shape after crop: {arr_cropped.shape}"

    print(f"  → cropped to {arr_cropped.shape}  ✓")
    return arr_cropped


def save_event_h5(vil_data: np.ndarray, ir_data: np.ndarray,
                   out_dir: Path, event_id: str) -> None:
    """
    Save one event to two H5 files in the format expected by app.py.

    Dataset layout (matches app.py load_event_from_h5 'Case A'):
        past   : (13, 128, 128)   frames 0–12
        future : (12, 128, 128)   frames 13–24
    """
    past_sl   = slice(0,  T_PAST)
    future_sl = slice(T_PAST, TOTAL_FRAMES)

    for mod, arr in [("vil", vil_data), ("ir069", ir_data)]:
        path = out_dir / f"real_event_{event_id}_{mod}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("past",   data=arr[past_sl],   compression="gzip")
            f.create_dataset("future", data=arr[future_sl], compression="gzip")
            f.attrs["event_id"] = event_id
            f.attrs["img_type"] = mod
        print(f"  Saved → {path.name}  "
              f"(past={arr[past_sl].shape}, future={arr[future_sl].shape})")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path("sample_data")
    out_dir.mkdir(exist_ok=True)

    print("Connecting to AWS S3 (anonymous) …")
    fs = s3fs.S3FileSystem(anon=True)

    print("Loading SEVIR CATALOG.csv from S3 …")
    catalog = pd.read_csv(
        "s3://sevir/CATALOG.csv",
        storage_options={"anon": True},
        low_memory=False,
    )
    print(f"  Catalog loaded: {len(catalog):,} rows, "
          f"columns: {list(catalog.columns)}")

    # ── Find events that have BOTH VIL and IR069 ──────────────────────────
    print("\nSearching for paired VIL + IR069 events …")

    # Try 2019 first; fall back to any year if needed
    for year in ["2019", "2018", "2020", "2017"]:
        cat_yr  = catalog[catalog["time_utc"].astype(str).str.startswith(year, na=False)]
        vil_ids = set(cat_yr[cat_yr["img_type"] == "vil"]["id"])
        ir_ids  = set(cat_yr[cat_yr["img_type"] == "ir069"]["id"])
        paired  = sorted(vil_ids & ir_ids)
        if paired:
            print(f"  Found {len(paired):,} paired events in {year}. "
                  f"Fetching first {NUM_EVENTS_TO_FETCH}.")
            break
    else:
        print("ERROR: No paired VIL+IR069 events found in catalog.")
        return

    selected = paired[:NUM_EVENTS_TO_FETCH]

    success, failed = 0, []
    for event_id in selected:
        print(f"\n{'─'*60}")
        print(f"Event: {event_id}")
        try:
            vil_data = fetch_event_from_s3(event_id, "vil",   catalog, fs)
            ir_data  = fetch_event_from_s3(event_id, "ir069", catalog, fs)
            save_event_h5(vil_data, ir_data, out_dir, event_id)
            success += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append((event_id, str(e)))

    print(f"\n{'═'*60}")
    print(f"Done.  Success: {success}  Failed: {len(failed)}")
    if failed:
        for eid, err in failed:
            print(f"  ✗ {eid}: {err}")
    print(f"\nH5 files saved to '{out_dir}/'")
    print("Point the Streamlit app sidebar → 'Local H5 directory' → 'sample_data'")


if __name__ == "__main__":
    main()
