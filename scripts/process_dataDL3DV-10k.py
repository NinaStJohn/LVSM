# DL3DV-10K -> LVSM-style preprocessing script

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


def parse_frame_id(file_path: str) -> Optional[int]:
    m = FRAME_ID_RE.search(os.path.basename(file_path))
    if not m:
        return None
    return int(m.group(1))


def generate_full_list(base_path: str, output_dir: str) -> None:
    json_files = [
        os.path.join(base_path, f)
        for f in os.listdir(base_path)
        if f.endswith(".json")
    ]
    json_files = [os.path.abspath(f) for f in json_files]
    json_files.sort()

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "full_list.txt"), "w") as f:
        for file in json_files:
            f.write(file + "\n")


def safe_load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_split_dirs(output_root: str, split: str) -> Tuple[str, str]:
    images_dir = os.path.join(output_root, split, "images")
    meta_dir = os.path.join(output_root, split, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    return images_dir, meta_dir


def build_frame_record(
    scene_dir: str,
    frame_entry: Dict,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    seq_images_dir: str,
    copy_images: bool,
) -> Optional[Dict]:
    rel_img_path = frame_entry.get("file_path")
    if rel_img_path is None:
        return None

    src_img_path = os.path.join(scene_dir, rel_img_path)
    if not os.path.isfile(src_img_path):
        logging.warning("Missing image file: %s", src_img_path)
        return None

    img_name = os.path.basename(rel_img_path)
    dst_img_path = os.path.join(seq_images_dir, img_name)

    if copy_images:
        shutil.copy2(src_img_path, dst_img_path)

    transform_matrix = frame_entry.get("transform_matrix")
    if transform_matrix is None:
        logging.warning("Missing transform_matrix for %s", src_img_path)
        return None

    c2w = np.array(transform_matrix, dtype=np.float32)
    if c2w.shape != (4, 4):
        logging.warning("Bad transform_matrix shape in %s", src_img_path)
        return None

    # NeRF-style transforms.json usually stores camera-to-world.
    # LVSM metadata wants world-to-camera.
    w2c = np.linalg.inv(c2w)

    return {
        "image_path": dst_img_path,
        "fxfycxcy": [float(fx), float(fy), float(cx), float(cy)],
        "w2c": w2c.tolist()
    }


def process_scene(scene_dir: str, output_root: str, copy_images: bool = True) -> Tuple[bool, str]:
    try:
        scene_name = os.path.basename(scene_dir.rstrip("/"))
        split_json_path = os.path.join(scene_dir, "train_test_split_2.json")
        transforms_path = os.path.join(scene_dir, "transforms.json")

        if not os.path.isfile(split_json_path):
            return False, f"{scene_dir}: missing train_test_split_2.json"
        if not os.path.isfile(transforms_path):
            return False, f"{scene_dir}: missing transforms.json"

        split_info = safe_load_json(split_json_path)
        transforms = safe_load_json(transforms_path)

        train_ids = set(split_info.get("train_ids", []))
        test_ids = set(split_info.get("test_ids", []))

        fx = float(transforms["fl_x"])
        fy = float(transforms["fl_y"])
        cx = float(transforms["cx"])
        cy = float(transforms["cy"])

        frames = transforms.get("frames", [])
        if not isinstance(frames, list):
            return False, f"{scene_dir}: transforms.json frames is not a list"

        train_records: List[Dict] = []
        test_records: List[Dict] = []

        train_images_dir, train_meta_dir = ensure_split_dirs(output_root, "train")
        test_images_dir, test_meta_dir = ensure_split_dirs(output_root, "test")

        train_seq_images_dir = os.path.join(train_images_dir, scene_name)
        test_seq_images_dir = os.path.join(test_images_dir, scene_name)

        os.makedirs(train_seq_images_dir, exist_ok=True)
        os.makedirs(test_seq_images_dir, exist_ok=True)

        for frame_entry in frames:
            rel_img_path = frame_entry.get("file_path")
            if rel_img_path is None:
                continue

            frame_id = parse_frame_id(rel_img_path)
            if frame_id is None:
                logging.warning("Could not parse frame id from %s", rel_img_path)
                continue

            if frame_id in train_ids:
                record = build_frame_record(
                    scene_dir=scene_dir,
                    frame_entry=frame_entry,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    seq_images_dir=train_seq_images_dir,
                    copy_images=copy_images,
                )
                if record is not None:
                    train_records.append(record)

            elif frame_id in test_ids:
                record = build_frame_record(
                    scene_dir=scene_dir,
                    frame_entry=frame_entry,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    seq_images_dir=test_seq_images_dir,
                    copy_images=copy_images,
                )
                if record is not None:
                    test_records.append(record)

        train_records.sort(key=lambda x: os.path.basename(x["image_path"]))
        test_records.sort(key=lambda x: os.path.basename(x["image_path"]))

        if train_records:
            train_meta = {
                "scene_name": scene_name,
                "frames": train_records
            }
            with open(os.path.join(train_meta_dir, f"{scene_name}.json"), "w") as f:
                json.dump(train_meta, f, indent=4)

        if test_records:
            test_meta = {
                "scene_name": scene_name,
                "frames": test_records
            }
            with open(os.path.join(test_meta_dir, f"{scene_name}.json"), "w") as f:
                json.dump(test_meta, f, indent=4)

        return True, scene_dir

    except Exception as e:
        return False, f"{scene_dir}: {str(e)}"


def process_single_scene(args):
    return process_scene(*args)


def find_scene_dirs(base_path: str) -> List[str]:
    scene_dirs = []
    for entry in sorted(os.listdir(base_path)):
        scene_dir = os.path.join(base_path, entry)
        if not os.path.isdir(scene_dir):
            continue

        split_json = os.path.join(scene_dir, "train_test_split_2.json")
        transforms = os.path.join(scene_dir, "transforms.json")
        images_dir = os.path.join(scene_dir, "images_4")

        if os.path.isfile(split_json) and os.path.isfile(transforms) and os.path.isdir(images_dir):
            scene_dirs.append(scene_dir)

    return scene_dirs


def process_dataset_root(
    base_path: str,
    output_root: str,
    num_processes: Optional[int] = None,
    chunk_size: int = 1,
    copy_images: bool = True,
) -> None:
    scene_dirs = find_scene_dirs(base_path)
    total_scenes = len(scene_dirs)
    logging.info("Found %d scene directories in %s", total_scenes, base_path)

    if total_scenes == 0:
        logging.warning("No valid scene directories found in %s", base_path)
        return

    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)

    args = [(scene_dir, output_root, copy_images) for scene_dir in scene_dirs]

    start_time = time.time()
    with mp.Pool(num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_scene, args, chunksize=chunk_size),
                total=total_scenes,
                desc=f"Processing scenes with {num_processes} processes"
            )
        )

    successful = sum(1 for success, _ in results if success)
    failed = [(success, msg) for success, msg in results if not success]

    elapsed_time = time.time() - start_time
    logging.info("Processing completed in %.2f seconds", elapsed_time)
    logging.info("Successfully processed %d/%d scenes", successful, total_scenes)

    if failed:
        logging.warning("Failed to process %d scenes:", len(failed))
        for _, msg in failed:
            logging.warning("  - %s", msg)

    for split in ["train", "test"]:
        meta_dir = os.path.join(output_root, split, "metadata")
        save_dir = os.path.join(output_root, split)
        if os.path.isdir(meta_dir):
            generate_full_list(meta_dir, save_dir)
            logging.info("Generated full_list.txt for %s", split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to one DL3DV subset directory, e.g. /work/hdd/bdxf/dl3dv/DL3DV-10K/4K"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory root"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=32
    )
    parser.add_argument(
        "--symlink_images",
        action="store_true",
        help="Use symlinks instead of copying images"
    )

    args = parser.parse_args()

    copy_images = not args.symlink_images

    if args.symlink_images:
        # Swap behavior if you want symlinks instead of copies
        def build_frame_record_symlink(
            scene_dir: str,
            frame_entry: Dict,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            seq_images_dir: str,
            copy_images: bool,
        ) -> Optional[Dict]:
            rel_img_path = frame_entry.get("file_path")
            if rel_img_path is None:
                return None

            src_img_path = os.path.join(scene_dir, rel_img_path)
            if not os.path.isfile(src_img_path):
                logging.warning("Missing image file: %s", src_img_path)
                return None

            img_name = os.path.basename(rel_img_path)
            dst_img_path = os.path.join(seq_images_dir, img_name)

            if not os.path.exists(dst_img_path):
                os.symlink(src_img_path, dst_img_path)

            transform_matrix = frame_entry.get("transform_matrix")
            if transform_matrix is None:
                logging.warning("Missing transform_matrix for %s", src_img_path)
                return None

            c2w = np.array(transform_matrix, dtype=np.float32)
            if c2w.shape != (4, 4):
                logging.warning("Bad transform_matrix shape in %s", src_img_path)
                return None

            w2c = np.linalg.inv(c2w)

            return {
                "image_path": dst_img_path,
                "fxfycxcy": [float(fx), float(fy), float(cx), float(cy)],
                "w2c": w2c.tolist()
            }

        build_frame_record = build_frame_record_symlink  # type: ignore

    logging.info("Starting DL3DV-10K processing...")
    process_dataset_root(
        base_path=args.base_path,
        output_root=args.output_dir,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size,
        copy_images=copy_images,
    )
    logging.info("Processing completed.")