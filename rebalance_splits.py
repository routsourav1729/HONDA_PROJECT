#!/usr/bin/env python3
"""
Rebalance train/test splits:
  1) For rare/tail classes, move images from train to test to achieve ~50/50 object count split.
  2) Swap all pole-containing images between train and test.

Uses multiprocessing for fast XML parsing.
No annotation files are modified — only t1.txt and test.txt are updated.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
import time

# ── Configuration ──────────────────────────────────────────────────────
ANNOTATIONS_DIR = Path(
    "/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/datasets/Annotations"
)
TRAIN_FILE = Path(
    "/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/datasets/ImageSets/Main/IDD/t1.txt"
)
TEST_FILE = Path(
    "/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/datasets/ImageSets/Main/IDD/test.txt"
)

# Rare/tail classes to balance to ~50/50 object‐count split
CLASSES_TO_BALANCE = [
    "road_roller",
    "pull_cart",
    "concrete_mixer",
    "crane_truck",
    "excavator",
    "tanker_vehicle",
    "tractor",
    "street_cart",
    "animal",
    "bicycle",
]

# Class whose images will be swapped entirely between train ↔ test
SWAP_CLASS = "pole"

SEED = 42


# ── XML parsing helpers (multiprocessing friendly) ─────────────────────
def _parse_single(xml_path: str):
    """Return (image_id, {class_name: count})"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        counts = defaultdict(int)
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is not None and name_elem.text:
                counts[name_elem.text] += 1
        return (Path(xml_path).stem, dict(counts))
    except Exception:
        return (Path(xml_path).stem, {})


def _parse_batch(xml_paths):
    """Parse a batch – called by each worker."""
    return [_parse_single(p) for p in xml_paths]


def parse_all_annotations(image_ids, num_workers=16):
    """Parse annotations for *image_ids* in parallel.
    Returns dict  image_id → {class_name: object_count}.
    """
    xml_paths = [str(ANNOTATIONS_DIR / f"{iid}.xml") for iid in image_ids]

    batch_size = max(1, len(xml_paths) // (num_workers * 4))
    batches = [xml_paths[i : i + batch_size] for i in range(0, len(xml_paths), batch_size)]

    print(f"  Parsing {len(xml_paths)} annotations with {num_workers} workers, "
          f"{len(batches)} batches …")

    with Pool(num_workers) as pool:
        batch_results = pool.map(_parse_batch, batches)

    image_classes = {}
    for batch in batch_results:
        for img_id, counts in batch:
            image_classes[img_id] = counts
    return image_classes


# ── Counting helpers ───────────────────────────────────────────────────
def class_counts_for(image_set, image_classes):
    """Return {class_name: total_object_count} over *image_set*."""
    counts = defaultdict(int)
    for img_id in image_set:
        for cls, n in image_classes.get(img_id, {}).items():
            counts[cls] += n
    return dict(counts)


def count_class_in(cls, image_set, image_classes):
    return sum(image_classes.get(i, {}).get(cls, 0) for i in image_set)


# ── Main logic ─────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    t0 = time.time()

    # Read current splits
    with open(TRAIN_FILE) as f:
        train_ids = set(line.strip() for line in f if line.strip())
    with open(TEST_FILE) as f:
        test_ids = set(line.strip() for line in f if line.strip())

    all_ids = train_ids | test_ids
    print(f"Initial — Train: {len(train_ids):,}  Test: {len(test_ids):,}  "
          f"Total: {len(all_ids):,}")

    # Parse all annotations
    num_workers = min(16, cpu_count())
    image_classes = parse_all_annotations(all_ids, num_workers)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── Phase 1: Balance rare classes ──────────────────────────────────
    print("=" * 64)
    print("PHASE 1 — Balance rare classes to ~50/50 object‐count split")
    print("=" * 64)

    # We'll accumulate images to move; process rarest class first so that
    # if a moved image also helps a less‐rare class, we account for it.
    images_to_move = set()  # train → test

    # Sort target classes by combined rarity (rarest first)
    def _combined(cls):
        return sum(image_classes.get(i, {}).get(cls, 0) for i in all_ids)

    for cls in sorted(CLASSES_TO_BALANCE, key=_combined):
        # Current effective sets (account for already‐scheduled moves)
        eff_train = train_ids - images_to_move
        eff_test  = test_ids | images_to_move

        train_cnt = count_class_in(cls, eff_train, image_classes)
        test_cnt  = count_class_in(cls, eff_test, image_classes)
        combined  = train_cnt + test_cnt
        target    = combined // 2
        deficit   = target - test_cnt

        if deficit <= 0:
            print(f"  {cls:<20s}  train={train_cnt:>6,}  test={test_cnt:>6,}  "
                  f"combined={combined:>6,}  ✓ already ≥50 %")
            continue

        # Candidates: train images (not yet scheduled) that contain this class
        candidates = [
            i for i in eff_train
            if image_classes.get(i, {}).get(cls, 0) > 0
        ]
        random.shuffle(candidates)

        moved_objs = 0
        moved_imgs = 0
        for img_id in candidates:
            if moved_objs >= deficit:
                break
            images_to_move.add(img_id)
            moved_objs += image_classes[img_id][cls]
            moved_imgs += 1

        new_test_cnt = test_cnt + moved_objs
        print(f"  {cls:<20s}  train={train_cnt:>6,}  test={test_cnt:>6,}  "
              f"combined={combined:>6,}  deficit={deficit:>5,}  "
              f"→ moved {moved_imgs:,} imgs ({moved_objs:,} objs)  "
              f"new_test={new_test_cnt:,}")

    # Apply phase‑1 moves
    train_ids -= images_to_move
    test_ids  |= images_to_move
    print(f"\nPhase 1 total: moved {len(images_to_move):,} images from train → test")
    print(f"  Train: {len(train_ids):,}  Test: {len(test_ids):,}")

    # ── Phase 2: Swap pole images ──────────────────────────────────────
    print("\n" + "=" * 64)
    print("PHASE 2 — Swap all pole‐containing images between train ↔ test")
    print("=" * 64)

    train_pole = {i for i in train_ids if image_classes.get(i, {}).get(SWAP_CLASS, 0) > 0}
    test_pole  = {i for i in test_ids  if image_classes.get(i, {}).get(SWAP_CLASS, 0) > 0}

    pole_train_objs = count_class_in(SWAP_CLASS, train_pole, image_classes)
    pole_test_objs  = count_class_in(SWAP_CLASS, test_pole, image_classes)
    print(f"  Before swap —  Train: {len(train_pole):,} imgs ({pole_train_objs:,} pole objs)  "
          f"Test: {len(test_pole):,} imgs ({pole_test_objs:,} pole objs)")

    # Swap: train‑pole → test,  test‑pole → train
    train_ids = (train_ids - train_pole) | test_pole
    test_ids  = (test_ids  - test_pole)  | train_pole

    pole_train_after = count_class_in(SWAP_CLASS, train_ids, image_classes)
    pole_test_after  = count_class_in(SWAP_CLASS, test_ids,  image_classes)
    print(f"  After  swap —  Train: {sum(1 for i in train_ids if image_classes.get(i,{}).get(SWAP_CLASS,0)>0):,} imgs ({pole_train_after:,} pole objs)  "
          f"Test: {sum(1 for i in test_ids if image_classes.get(i,{}).get(SWAP_CLASS,0)>0):,} imgs ({pole_test_after:,} pole objs)")
    print(f"  Train: {len(train_ids):,}  Test: {len(test_ids):,}")

    # ── Write updated split files ──────────────────────────────────────
    print("\n" + "=" * 64)
    print("Writing updated split files")
    print("=" * 64)

    train_sorted = sorted(train_ids)
    test_sorted  = sorted(test_ids)

    with open(TRAIN_FILE, "w") as f:
        f.write("\n".join(train_sorted) + "\n")
    with open(TEST_FILE, "w") as f:
        f.write("\n".join(test_sorted) + "\n")

    print(f"  {TRAIN_FILE}  →  {len(train_sorted):,} images")
    print(f"  {TEST_FILE}  →  {len(test_sorted):,} images")

    # ── Final distribution summary ─────────────────────────────────────
    print("\n" + "=" * 64)
    print("FINAL CLASS DISTRIBUTION")
    print("=" * 64)

    train_counts = class_counts_for(train_ids, image_classes)
    test_counts  = class_counts_for(test_ids, image_classes)
    all_classes  = sorted(set(list(train_counts) + list(test_counts)),
                          key=lambda c: -(train_counts.get(c, 0) + test_counts.get(c, 0)))

    print(f"\n{'Class':<22s} {'Train':>10s} {'Test':>10s} {'Combined':>10s} {'Test%':>7s}  Note")
    print("-" * 72)
    for cls in all_classes:
        tc  = train_counts.get(cls, 0)
        tec = test_counts.get(cls, 0)
        comb = tc + tec
        pct  = tec / comb * 100 if comb else 0
        note = ""
        if cls in CLASSES_TO_BALANCE:
            note = "balanced"
        elif cls == SWAP_CLASS:
            note = "swapped"
        print(f"{cls:<22s} {tc:>10,} {tec:>10,} {comb:>10,} {pct:>6.1f}%  {note}")

    total_train = sum(train_counts.values())
    total_test  = sum(test_counts.values())
    print("-" * 72)
    print(f"{'TOTAL':<22s} {total_train:>10,} {total_test:>10,} "
          f"{total_train + total_test:>10,}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
