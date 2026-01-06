"""
Randomly sample PGM face images from Extended Yale B subfolders.

Features:
- Randomly pick N folders (default 15) among subfolders like 'yaleB11', 'yaleB12', ...
- From each selected folder, randomly pick K PGM files (default 11)
- Try to avoid picking adjacent files in lexicographic order (min gap=1)
- Copy all selected images into a single output directory
- If output directory already exists, it will be CLEANED (all files/subfolders removed) before copying
- Generate a manifest.csv with metadata

Usage (PowerShell example):
    python sample_extended_yaleb.py --src "data/ExtendedYaleB" --out "data/faces" --folders 15 --per-folder 11 --seed 42

This script uses only Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


YALEB_DIR_PATTERN = re.compile(r"^yaleB\d+\Z", re.IGNORECASE)


def list_yaleb_person_dirs(src: Path) -> List[Path]:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source directory not found or not a directory: {src}")
    dirs = [p for p in src.iterdir() if p.is_dir() and YALEB_DIR_PATTERN.match(p.name)]
    # Stable sort by numeric id inside name
    def key_fn(p: Path) -> Tuple[int, str]:
        m = re.search(r"(\d+)", p.name)
        return (int(m.group(1)) if m else 0, p.name.lower())

    return sorted(dirs, key=key_fn)


def list_pgm_files(person_dir: Path) -> List[Path]:
    files = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pgm"]
    return sorted(files, key=lambda p: p.name.lower())


def sample_without_adjacent(total: int, k: int, rng: random.Random, min_gap: int = 1, max_attempts: int = 5000) -> List[int]:
    """
    Return k indices from range(total) attempting to ensure a minimum gap in sorted order.
    If strict sampling fails after many attempts, fall back to evenly spaced then random adjust.
    """
    if k <= 0:
        return []
    if total <= 0:
        return []
    if k > total:
        # Best-effort fallback
        return list(range(total))

    # Fast path: if spacing allows deterministic even spacing
    # Even spacing target positions
    if (k - 1) * (min_gap + 1) < total:
        # Choose evenly spaced anchors + small random jitter
        step = total / k
        candidates = set()
        for i in range(k):
            base = int(round(i * step + step / 2.0))
            # Clamp into range
            base = max(0, min(total - 1, base))
            candidates.add(base)
        # If collisions, fill randomly with constraint
        chosen = sorted(list(candidates))
        # Greedy adjust to satisfy min_gap by moving away a few steps where possible
        for _ in range(3):  # limited smoothing passes
            chosen_sorted = sorted(chosen)
            ok = True
            for i in range(1, len(chosen_sorted)):
                if abs(chosen_sorted[i] - chosen_sorted[i - 1]) <= min_gap - 0:
                    ok = False
                    # try to push right if possible else left
                    shift_right = chosen_sorted[i] + (min_gap + 1)
                    if shift_right < total:
                        chosen_sorted[i] = shift_right
                    else:
                        shift_left = chosen_sorted[i - 1] - (min_gap + 1)
                        if shift_left >= 0:
                            chosen_sorted[i - 1] = shift_left
            if ok:
                chosen = chosen_sorted
                break
            chosen = chosen_sorted
        if len(chosen) >= k:
            return sorted(chosen)[:k]
        # else continue to random approach

    # Random greedy with retries
    indices = list(range(total))
    for _ in range(max_attempts):
        rng.shuffle(indices)
        picked = []
        for idx in sorted(indices[: min(total, max(k * 3, 64))]):
            if not picked:
                picked.append(idx)
            else:
                if all(abs(idx - p) > min_gap for p in picked):
                    picked.append(idx)
            if len(picked) == k:
                break
        if len(picked) == k:
            return sorted(picked)

    # Fallback: evenly spaced without strict gap
    step = total / k
    approx = [int(i * step + step / 2.0) for i in range(k)]
    return sorted({max(0, min(total - 1, x)) for x in approx})[:k]


def ensure_unique_name(dst_dir: Path, file_name: str) -> str:
    """Return a unique filename under dst_dir; append numeric suffix if needed."""
    base = Path(file_name).stem
    ext = Path(file_name).suffix
    candidate = file_name
    i = 1
    while (dst_dir / candidate).exists():
        candidate = f"{base}__{i}{ext}"
        i += 1
    return candidate


def clean_directory(dir_path: Path) -> None:
    """Remove all files and subdirectories under dir_path (but keep dir_path itself)."""
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except Exception as e:
            raise RuntimeError(f"Failed to clean '{dir_path}': {e}")


def copy_selection(
    selections: List[Tuple[Path, Path]], out_dir: Path, manifest_path: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dest_file", "person_folder", "file_name", "src_path"])
        for src_file, person_dir in selections:
            dest_name = ensure_unique_name(out_dir, src_file.name)
            dest_path = out_dir / dest_name
            shutil.copy2(src_file, dest_path)
            writer.writerow([dest_name, person_dir.name, src_file.name, str(src_file)])


def main():
    parser = argparse.ArgumentParser(description="Sample PGM images from Extended Yale B folders")
    parser.add_argument("--src", type=str, required=True, help="Path to ExtendedYaleB root directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory to store sampled images")
    parser.add_argument("--folders", type=int, default=15, help="Number of person folders to sample (default: 15)")
    parser.add_argument("--per-folder", type=int, default=11, help="Number of images per folder (default: 11)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--min-gap", type=int, default=1, help="Minimum index gap to avoid adjacency in sorted order (default: 1)")

    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    rng = random.Random(args.seed)

    person_dirs = list_yaleb_person_dirs(src)
    if not person_dirs:
        raise SystemExit(f"No 'yaleB*' folders found in {src}")
    if args.folders > len(person_dirs):
        print(f"[warn] Requested {args.folders} folders but only {len(person_dirs)} available; reducing to {len(person_dirs)}")
    num_folders = min(args.folders, len(person_dirs))

    selected_persons = rng.sample(person_dirs, k=num_folders)

    selections: List[Tuple[Path, Path]] = []
    skipped_persons = []
    for pdir in selected_persons:
        pgms = list_pgm_files(pdir)
        total = len(pgms)
        if total == 0:
            skipped_persons.append((pdir, "no pgm files"))
            continue
        k = min(args.per_folder, total)
        idxs = sample_without_adjacent(total, k, rng=rng, min_gap=max(0, args.min_gap))
        for i in idxs:
            selections.append((pgms[i], pdir))

    # Prepare output directory: clean if it exists
    if out_dir.exists():
        clean_directory(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    copy_selection(selections, out_dir, manifest_path)

    # Summary
    by_person = {}
    for _, pdir in selections:
        by_person[pdir.name] = by_person.get(pdir.name, 0) + 1

    print("==== Sampling Summary ====")
    print(f"Source: {src}")
    print(f"Output: {out_dir}")
    print(f"Folders selected: {len(selected_persons)}")
    print(f"Total images copied: {len(selections)}")
    for person, cnt in sorted(by_person.items()):
        print(f"  {person}: {cnt}")
    if skipped_persons:
        print("[info] Skipped persons:")
        for pdir, reason in skipped_persons:
            print(f"  {pdir.name}: {reason}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
