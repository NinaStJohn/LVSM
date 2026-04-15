# DL3DV-10K -> LVSM-style preprocessing script
# Supports all subsets (1K-10K), auto train/test split, and outputs:
#   - LVSM format: per-scene JSON metadata + full_list.txt
#   - AE format:   flat image_list.txt with one absolute image path per line

import os
import re
import json
import shutil
import time
import argparse
import logging
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FRAME_ID_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
SUBSET_NAMES = [f"{i}K" for i in range(1, 11)]  # 1K .. 10K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_frame_id(file_path: str) -> Optional[int]:
    m = FRAME_ID_RE.search(os.path.basename(file_path))
    return int(m.group(1)) if m else None


def safe_load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_split_dirs(output_root: str, split: str) -> Tuple[str, str]:
    images_dir = os.path.join(output_root, split, "images")
    meta_dir = os.path.join(output_root, split, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    return images_dir, meta_dir


def make_train_test_split(
    frame_ids: List[int],
    test_fraction: float = 0.1,
    min_test: int = 1,
) -> Tuple[set, set]:
    """Hold out every N-th frame as test when no split JSON is present."""
    sorted_ids = sorted(frame_ids)
    n = len(sorted_ids)
    n_test = max(min_test, int(round(n * test_fraction)))
    step = max(1, n // n_test)
    test_ids = set(sorted_ids[i] for i in range(0, n, step))
    train_ids = set(sorted_ids) - test_ids
    return train_ids, test_ids


def link_or_copy(src: str, dst: str, use_symlink: bool) -> None:
    if use_symlink:
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Per-frame record builder
# ---------------------------------------------------------------------------

def build_frame_record(
    scene_dir: str,
    frame_entry: Dict,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    seq_images_dir: str,
    use_symlink: bool,
) -> Optional[Dict]:
    """
    Returns an LVSM-format record dict, or None on failure.
    Also creates the image file/symlink at seq_images_dir/<img_name>.
    """
    rel_img_path = frame_entry.get("file_path")
    if rel_img_path is None:
        return None

    src_img_path = os.path.join(scene_dir, rel_img_path)
    if not os.path.isfile(src_img_path):
        logging.warning("Missing image file: %s", src_img_path)
        return None

    img_name = os.path.basename(rel_img_path)
    dst_img_path = os.path.join(seq_images_dir, img_name)
    link_or_copy(src_img_path, dst_img_path, use_symlink)

    transform_matrix = frame_entry.get("transform_matrix")
    if transform_matrix is None:
        logging.warning("Missing transform_matrix for %s", src_img_path)
        return None

    c2w = np.array(transform_matrix, dtype=np.float32)
    if c2w.shape != (4, 4):
        logging.warning("Bad transform_matrix shape in %s", src_img_path)
        return None

    # NeRF stores camera-to-world; LVSM wants world-to-camera
    w2c = np.linalg.inv(c2w)

    return {
        "image_path": os.path.abspath(dst_img_path),
        "fxfycxcy": [float(fx), float(fy), float(cx), float(cy)],
        "w2c": w2c.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-scene processor  (runs in worker processes)
# ---------------------------------------------------------------------------

def process_scene(
    scene_dir: str,
    output_root: str,
    use_symlink: bool,
    test_fraction: float,
    mode: str,           # "lvsm" | "ae" | "both"
) -> Tuple[bool, str, List[str], List[str]]:
    """
    Returns (success, message, train_image_paths, test_image_paths).
    train/test_image_paths are absolute paths used to build the flat AE lists.
    """
    try:
        scene_name = os.path.basename(scene_dir.rstrip("/"))
        transforms_path = os.path.join(scene_dir, "transforms.json")
        split_json_path = os.path.join(scene_dir, "train_test_split_2.json")

        if not os.path.isfile(transforms_path):
            return False, f"{scene_dir}: missing transforms.json", [], []

        transforms = safe_load_json(transforms_path)
        frames = transforms.get("frames", [])
        if not isinstance(frames, list) or len(frames) == 0:
            return False, f"{scene_dir}: transforms.json has no frames", [], []

        def get_intrinsic(key, frame=None):
            if frame and key in frame:
                return float(frame[key])
            return float(transforms[key])

        # --- resolve split ---
        if os.path.isfile(split_json_path):
            split_info = safe_load_json(split_json_path)
            train_ids = set(split_info.get("train_ids", []))
            test_ids = set(split_info.get("test_ids", []))
        else:
            all_ids = [
                parse_frame_id(fe.get("file_path", ""))
                for fe in frames
                if parse_frame_id(fe.get("file_path", "")) is not None
            ]
            train_ids, test_ids = make_train_test_split(all_ids, test_fraction=test_fraction)

        need_lvsm = mode in ("lvsm", "both")

        # Always need image dirs (both modes write images/symlinks)
        train_images_dir, train_meta_dir = ensure_split_dirs(output_root, "train")
        test_images_dir,  test_meta_dir  = ensure_split_dirs(output_root, "test")
        train_seq_dir = os.path.join(train_images_dir, scene_name)
        test_seq_dir  = os.path.join(test_images_dir,  scene_name)
        os.makedirs(train_seq_dir, exist_ok=True)
        os.makedirs(test_seq_dir,  exist_ok=True)

        train_records: List[Dict] = []
        test_records:  List[Dict] = []
        train_img_paths: List[str] = []
        test_img_paths:  List[str] = []

        for frame_entry in frames:
            rel_img_path = frame_entry.get("file_path")
            if rel_img_path is None:
                continue

            frame_id = parse_frame_id(rel_img_path)
            if frame_id is None:
                logging.warning("Could not parse frame id from %s", rel_img_path)
                continue

            try:
                fx = get_intrinsic("fl_x", frame_entry)
                fy = get_intrinsic("fl_y", frame_entry)
                cx = get_intrinsic("cx",   frame_entry)
                cy = get_intrinsic("cy",   frame_entry)
            except (KeyError, TypeError) as e:
                logging.warning("Missing intrinsics in %s: %s", scene_dir, e)
                continue

            if frame_id in train_ids:
                record = build_frame_record(
                    scene_dir, frame_entry, fx, fy, cx, cy,
                    train_seq_dir, use_symlink,
                )
                if record is not None:
                    train_records.append(record)
                    train_img_paths.append(record["image_path"])

            elif frame_id in test_ids:
                record = build_frame_record(
                    scene_dir, frame_entry, fx, fy, cx, cy,
                    test_seq_dir, use_symlink,
                )
                if record is not None:
                    test_records.append(record)
                    test_img_paths.append(record["image_path"])

        train_records.sort(key=lambda x: os.path.basename(x["image_path"]))
        test_records.sort( key=lambda x: os.path.basename(x["image_path"]))
        train_img_paths.sort()
        test_img_paths.sort()

        # --- write LVSM per-scene JSON ---
        if need_lvsm:
            if train_records:
                with open(os.path.join(train_meta_dir, f"{scene_name}.json"), "w") as f:
                    json.dump({"scene_name": scene_name, "frames": train_records}, f, indent=2)
            if test_records:
                with open(os.path.join(test_meta_dir, f"{scene_name}.json"), "w") as f:
                    json.dump({"scene_name": scene_name, "frames": test_records}, f, indent=2)

        return True, scene_dir, train_img_paths, test_img_paths

    except Exception as e:
        import traceback
        return False, f"{scene_dir}: {e}\n{traceback.format_exc()}", [], []


def _worker(args):
    return process_scene(*args)


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------

def find_scene_dirs(base_path: str) -> List[str]:
    """
    Accept either the DL3DV-10K root (containing 1K/..10K) or a single subset dir.
    A valid scene must have transforms.json and images_4/.
    """
    scene_dirs = []
    EXCLUDED_DIRS = {"bad_scenes"}

    def _scan(directory: str):
        try:
            entries = sorted(os.listdir(directory))
        except PermissionError:
            return
        for entry in entries:
            sd = os.path.join(directory, entry)
            if (os.path.isdir(sd)
                    and os.path.isfile(os.path.join(sd, "transforms.json"))
                    and os.path.isdir(os.path.join(sd, "images_4"))):
                scene_dirs.append(sd)

    entries = sorted(os.listdir(base_path))
    has_subsets = any(e in SUBSET_NAMES for e in entries)

    if has_subsets:
        for entry in entries:
            if entry in EXCLUDED_DIRS or entry not in SUBSET_NAMES:
                continue
            subset_path = os.path.join(base_path, entry)
            if os.path.isdir(subset_path):
                logging.info("Scanning subset: %s", subset_path)
                _scan(subset_path)
    else:
        _scan(base_path)

    return scene_dirs


def write_full_list(meta_dir: str, out_dir: str) -> None:
    """LVSM full_list.txt — absolute paths to every per-scene JSON."""
    json_files = sorted(
        os.path.abspath(os.path.join(meta_dir, f))
        for f in os.listdir(meta_dir)
        if f.endswith(".json")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "full_list.txt"), "w") as f:
        for p in json_files:
            f.write(p + "\n")


def write_image_list(image_paths: List[str], out_path: str) -> None:
    """AE flat list — one absolute image path per line."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for p in sorted(image_paths):
            f.write(p + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_dataset_root(
    base_path: str,
    output_root: str,
    num_processes: int,
    chunk_size: int,
    use_symlink: bool,
    test_fraction: float,
    mode: str,
) -> None:
    scene_dirs = find_scene_dirs(base_path)
    total = len(scene_dirs)
    logging.info("Found %d scene directories", total)

    if total == 0:
        logging.warning("No valid scene directories found in %s", base_path)
        return

    worker_args = [
        (sd, output_root, use_symlink, test_fraction, mode)
        for sd in scene_dirs
    ]

    start = time.time()
    with mp.Pool(num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(_worker, worker_args, chunksize=chunk_size),
                total=total,
                desc=f"Processing ({num_processes} workers, mode={mode})",
            )
        )

    ok_count = sum(1 for r in results if r[0])
    failed   = [r for r in results if not r[0]]
    logging.info("Done in %.1f s — %d/%d scenes OK", time.time() - start, ok_count, total)

    if failed:
        logging.warning("%d scenes failed:", len(failed))
        for r in failed:
            logging.warning("  %s", r[1])

    # Collect all image paths from successful scenes
    all_train_imgs = [p for r in results if r[0] for p in r[2]]
    all_test_imgs  = [p for r in results if r[0] for p in r[3]]

    need_lvsm = mode in ("lvsm", "both")
    need_ae   = mode in ("ae",   "both")

    # --- LVSM outputs ---
    if need_lvsm:
        for split in ("train", "test"):
            meta_dir = os.path.join(output_root, split, "metadata")
            if os.path.isdir(meta_dir):
                write_full_list(meta_dir, os.path.join(output_root, split))
                logging.info(
                    "Wrote LVSM full_list.txt  ->  %s/%s/full_list.txt",
                    output_root, split,
                )

    # --- AE outputs ---
    if need_ae:
        ae_dir = os.path.join(output_root, "ae")
        train_list = os.path.join(ae_dir, "train_image_list.txt")
        test_list  = os.path.join(ae_dir, "test_image_list.txt")

        write_image_list(all_train_imgs, train_list)
        write_image_list(all_test_imgs,  test_list)

        logging.info(
            "Wrote AE image lists ->  %s  (%d train / %d test images)",
            ae_dir, len(all_train_imgs), len(all_test_imgs),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess DL3DV-10K into LVSM and/or AE format.\n\n"
            "Modes:\n"
            "  lvsm  ->  per-scene JSON + full_list.txt  (for LVSM training)\n"
            "  ae    ->  flat image_list.txt              (for autoencoder training)\n"
            "  both  ->  both outputs in one pass         (recommended)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_path", required=True,
        help=(
            "DL3DV-10K root containing 1K/.../10K subdirs, "
            "OR a single subset dir (e.g. .../DL3DV-10K/4K)"
        ),
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root output directory",
    )
    parser.add_argument(
        "--mode", choices=["lvsm", "ae", "both"], default="both",
        help="What to output (default: both)",
    )
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--chunk_size",    type=int, default=10)
    parser.add_argument(
        "--symlink_images", action="store_true",
        help=(
            "Symlink images instead of copying. "
            "STRONGLY recommended — avoids duplicating terabytes of image data."
        ),
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.1,
        help="Fraction of frames held out as test when no split JSON exists (default: 0.1)",
    )

    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("DL3DV-10K preprocessing")
    logging.info("  base_path     : %s", args.base_path)
    logging.info("  output_dir    : %s", args.output_dir)
    logging.info("  mode          : %s", args.mode)
    logging.info("  num_processes : %d", args.num_processes)
    logging.info("  symlink_images: %s", args.symlink_images)
    logging.info("  test_fraction : %.2f", args.test_fraction)
    logging.info("=" * 60)

    if not args.symlink_images:
        logging.warning(
            "WARNING: --symlink_images not set. Images will be COPIED, "
            "duplicating all data. Pass --symlink_images to avoid this."
        )

    process_dataset_root(
        base_path=args.base_path,
        output_root=args.output_dir,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size,
        use_symlink=args.symlink_images,
        test_fraction=args.test_fraction,
        mode=args.mode,
    )
    logging.info("All done.")