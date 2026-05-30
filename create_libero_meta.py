#!/usr/bin/env python
"""
Create LIBERO Dataset Training Metadata Configuration

Usage:
    python create_libero_meta.py \\
        --data_dir /datasets/metas \\
        --output ./datasets/metas/libero_train.json

This will scan the LIBERO dataset directory and generate a metadata file 
containing all HDF5 file paths and task descriptions.

Note: Each LIBERO HDF5 file contains 50 demos (episodes).
"""

import argparse
import json
import os
import glob
import re
from typing import List, Dict

import h5py


EXPECTED_SUBSET_FILES = {
    "libero_10": 10,
    "libero_goal": 10,
    "libero_object": 10,
    "libero_spatial": 10,
    "libero_90": 90,
}
EXPECTED_DEMOS_PER_FILE = 50
REQUIRED_DEMO_KEYS = (
    "actions",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/ee_pos",
    "obs/ee_ori",
    "obs/gripper_states",
)


def resolve_subset_dirs(data_dir: str, subsets: List[str]) -> List[tuple[str, str]]:
    """Resolve LIBERO subset names to existing directories.

    The repo may store LIBERO-100 as either flat directories:
      datasets/metas/libero_10, datasets/metas/libero_90

    or the grouped layout:
      datasets/metas/libero_100/libero_10
      datasets/metas/libero_100/libero_90
    """
    resolved: List[tuple[str, str]] = []

    def add_subset(label: str, rel_path: str) -> None:
        subset_dir = os.path.join(data_dir, rel_path)
        if os.path.exists(subset_dir):
            resolved.append((label, subset_dir))
        else:
            print(f"Warning: Skipping non-existent subset: {rel_path}")

    for subset in subsets:
        normalized = subset.strip("/").replace("\\", "/")

        if normalized == "libero_100":
            add_subset("libero_10", "libero_100/libero_10")
            add_subset("libero_90", "libero_100/libero_90")
        elif normalized in {"libero_10", "libero_90"}:
            flat_dir = os.path.join(data_dir, normalized)
            if os.path.exists(flat_dir):
                resolved.append((normalized, flat_dir))
            else:
                add_subset(normalized, f"libero_100/{normalized}")
        else:
            label = os.path.basename(normalized)
            add_subset(label, normalized)

    return resolved


def parse_task_from_filename(filepath: str) -> str:
    """Parse task description from filename."""
    base = os.path.basename(filepath)
    # e.g., KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
    task = re.sub(r"_demo\.hdf5$", "", base)
    m = re.search(r"SCENE\d+_", task)
    if m:
        task = task[m.end():]
    task = task.replace("_", " ")
    return task


def format_h5_errors(errors: List[Dict[str, str]]) -> str:
    lines = ["Unreadable or invalid HDF5 files:"]
    for item in errors:
        lines.append(f"  - [{item['subset']}] {item['path']}: {item['error']}")
    return "\n".join(lines)


def count_demos_in_h5(h5_path: str) -> int:
    """Count demos in HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise ValueError("missing data group")

        demo_keys = [k for k in f["data"].keys() if k.startswith("demo")]
        if not demo_keys:
            raise ValueError("no demo_* groups found")

        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            for required_key in REQUIRED_DEMO_KEYS:
                if required_key not in demo:
                    raise ValueError(f"{demo_key} missing {required_key}")

        return len(demo_keys)


def validate_expected_counts(stats: Dict[str, Dict[str, int]], allow_incomplete: bool) -> None:
    if allow_incomplete:
        return

    problems = []
    for subset, expected_files in EXPECTED_SUBSET_FILES.items():
        if subset not in stats:
            continue

        expected_demos = expected_files * EXPECTED_DEMOS_PER_FILE
        actual_files = stats[subset]["num_files"]
        actual_demos = stats[subset]["num_demos"]
        if actual_files != expected_files:
            problems.append(f"{subset}: {actual_files} files, expected {expected_files}")
        if actual_demos != expected_demos:
            problems.append(f"{subset}: {actual_demos} demos, expected {expected_demos}")

    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise RuntimeError(
            "LIBERO dataset looks incomplete. Re-download the affected subset/files "
            "or pass --allow_incomplete for debugging only.\n"
            f"{detail}"
        )


def create_libero_meta(
    data_dir: str,
    subsets: List[str] = None,
    output_path: str = None,
    skip_bad_files: bool = False,
    allow_incomplete: bool = False,
) -> Dict:
    """
    Create LIBERO dataset meta configuration.
    
    Args:
        data_dir: LIBERO dataset root directory
        subsets: List of subsets to include
        output_path: Output JSON path
        
    Returns:
        meta dictionary
    """
    if subsets is None:
        # Default 4 subsets (excluding libero_90)
        subsets = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    
    datalist = []
    stats = {}
    total_demos = 0
    bad_files = []
    
    print(f"Scanning LIBERO dataset: {data_dir}")
    
    resolved_subsets = resolve_subset_dirs(data_dir, subsets)

    for subset, subset_dir in resolved_subsets:
        h5_files = sorted(glob.glob(os.path.join(subset_dir, "*.hdf5")))
        subset_files = 0
        subset_demos = 0
        
        for h5_path in h5_files:
            task = parse_task_from_filename(h5_path)
            try:
                num_demos = count_demos_in_h5(h5_path)
            except Exception as e:
                bad_files.append({
                    "subset": subset,
                    "path": h5_path,
                    "error": str(e),
                })
                if skip_bad_files:
                    print(f"Warning: Skipping unreadable or invalid file: {h5_path}: {e}")
                    continue
                continue

            subset_files += 1
            subset_demos += num_demos
            
            datalist.append({
                "path": h5_path,
                "task": task,
                "subset": subset,
                "num_demos": num_demos,
            })
        
        stats[subset] = {
            "num_files": subset_files,
            "num_demos": subset_demos,
        }
        total_demos += subset_demos
        
        print(f"   {subset}: {subset_files} files, {subset_demos} demos")

    if bad_files and not skip_bad_files:
        raise RuntimeError(
            f"Found {len(bad_files)} bad HDF5 file(s). "
            "The official full LIBERO split should not skip any files.\n"
            f"{format_h5_errors(bad_files)}"
        )

    validate_expected_counts(stats, allow_incomplete or skip_bad_files)
    
    meta = {
        "dataset_name": "libero_hdf5",
        "data_dir": data_dir,
        "datalist": datalist,
        "num_files": len(datalist),
        "num_episodes": total_demos,
        "subsets": list(stats.keys()),
        "requested_subsets": subsets,
        "subset_stats": stats,
        "observation_key": ["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        "action_key": "actions",
        "state_dim": 8,
        "action_dim": 7,
        "fps": 10,
    }
    
    print(f"\nFound {len(datalist)} HDF5 files, {total_demos} episodes (demos)")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved to: {output_path}")
    
    return meta


def main():
    parser = argparse.ArgumentParser(description="Create LIBERO training metadata")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="LIBERO dataset root directory")
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=["libero_10", "libero_goal", "libero_object", "libero_spatial"],
                        help="Subsets to include (default 4 subsets, excluding libero_90)")
    parser.add_argument("--output", type=str,
                        default="./datasets/metas/libero_train.json",
                        help="Output file path")
    parser.add_argument("--validate_only", action="store_true",
                        help="Validate the dataset without writing metadata")
    parser.add_argument("--skip_bad_files", action="store_true",
                        help="Skip unreadable files instead of failing (debug only)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Do not enforce official LIBERO file/demo counts")
    
    args = parser.parse_args()
    
    create_libero_meta(
        data_dir=args.data_dir,
        subsets=args.subsets,
        output_path=None if args.validate_only else args.output,
        skip_bad_files=args.skip_bad_files,
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
