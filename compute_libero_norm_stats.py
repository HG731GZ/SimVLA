#!/usr/bin/env python
"""
Compute LIBERO Dataset Normalization Statistics

LIBERO data format:
- state (proprio): 8-dim [ee_pos(3), ee_ori(3), gripper_states(2)]
- actions: 7-dim [delta_xyz(3), delta_euler(3), gripper_cmd(1)]

Output format:
{
  "norm_stats": {
    "state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
    "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
  }
}

Usage:
    python compute_libero_norm_stats.py \\
        --data_dir /path/to/LIBERO/datasets \\
        --output ./norm_stats/libero_norm.json
"""

import argparse
import json
import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


EXPECTED_SUBSET_FILES = {
    "libero_10": 10,
    "libero_goal": 10,
    "libero_object": 10,
    "libero_spatial": 10,
    "libero_90": 90,
}
EXPECTED_DEMOS_PER_FILE = 50
EXPECTED_FULL_LIBERO_STEPS = 1007618
REQUIRED_NORM_KEYS = (
    "actions",
    "obs/ee_pos",
    "obs/ee_ori",
    "obs/gripper_states",
)


def _quat2axisangle_single(quat: np.ndarray) -> np.ndarray:
    """Convert one quaternion [x, y, z, w] to axis-angle."""
    import math

    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def euler_to_axisangle(euler: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to axis-angle, matching the LIBERO handler."""
    rot = R.from_euler("xyz", euler)
    quats = rot.as_quat()

    if quats.ndim == 1:
        return _quat2axisangle_single(quats)

    axis_angles = np.zeros((len(quats), 3), dtype=np.float32)
    for i in range(len(quats)):
        axis_angles[i] = _quat2axisangle_single(quats[i])
    return axis_angles


def resolve_subset_dirs(data_dir: str, subsets: List[str]) -> List[tuple[str, str]]:
    """Resolve LIBERO subset names to existing directories.

    Supports both flat layouts:
      datasets/metas/libero_10

    and the official grouped layout used here:
      datasets/metas/libero_100/libero_10
      datasets/metas/libero_100/libero_90
    """
    resolved: List[tuple[str, str]] = []

    def add_subset(label: str, rel_path: str) -> None:
        subset_dir = os.path.join(data_dir, rel_path)
        if os.path.exists(subset_dir):
            resolved.append((label, subset_dir))
        else:
            print(f"Warning: Skipping non-existent directory: {subset_dir}")

    for subset in subsets:
        normalized = subset.strip("/").replace("\\", "/")

        if normalized == "libero_100":
            add_subset("libero_10", "libero_100/libero_10")
            add_subset("libero_90", "libero_100/libero_90")
        elif normalized in {"libero_10", "libero_90"}:
            flat_dir = os.path.join(data_dir, normalized)
            nested_rel = f"libero_100/{normalized}"
            if os.path.exists(flat_dir):
                resolved.append((normalized, flat_dir))
            else:
                add_subset(normalized, nested_rel)
        else:
            label = os.path.basename(normalized)
            add_subset(label, normalized)

    return resolved


def format_h5_errors(errors: List[Dict[str, str]]) -> str:
    lines = ["Unreadable or invalid HDF5 files:"]
    for item in errors:
        lines.append(f"  - [{item['subset']}] {item['path']}: {item['error']}")
    return "\n".join(lines)


def inspect_h5_for_norm(h5_path: str) -> tuple[int, int]:
    """Validate one LIBERO HDF5 file and return (num_demos, num_steps)."""
    num_demos = 0
    num_steps = 0

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise ValueError("missing data group")

        demo_keys = [k for k in f["data"].keys() if k.startswith("demo")]
        if not demo_keys:
            raise ValueError("no demo_* groups found")

        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            for required_key in REQUIRED_NORM_KEYS:
                if required_key not in demo:
                    raise ValueError(f"{demo_key} missing {required_key}")

            lengths = [len(demo[key]) for key in REQUIRED_NORM_KEYS]
            T = min(lengths)
            if T <= 0:
                raise ValueError(f"{demo_key} has empty required data")

            num_demos += 1
            num_steps += T

    return num_demos, num_steps


def validate_expected_counts(
    subset_stats: Dict[str, Dict[str, int]],
    total_steps: int,
    allow_incomplete: bool,
) -> None:
    if allow_incomplete:
        return

    problems = []
    for subset, expected_files in EXPECTED_SUBSET_FILES.items():
        if subset not in subset_stats:
            continue

        expected_demos = expected_files * EXPECTED_DEMOS_PER_FILE
        actual_files = subset_stats[subset]["num_files"]
        actual_demos = subset_stats[subset]["num_demos"]
        if actual_files != expected_files:
            problems.append(f"{subset}: {actual_files} files, expected {expected_files}")
        if actual_demos != expected_demos:
            problems.append(f"{subset}: {actual_demos} demos, expected {expected_demos}")

    if set(subset_stats) == set(EXPECTED_SUBSET_FILES) and total_steps != EXPECTED_FULL_LIBERO_STEPS:
        problems.append(
            f"full LIBERO: {total_steps} steps, expected {EXPECTED_FULL_LIBERO_STEPS}"
        )

    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise RuntimeError(
            "LIBERO dataset/norm preflight does not match the official full split. "
            "Re-download the affected subset/files or pass --allow_incomplete for debugging only.\n"
            f"{detail}"
        )


class RunningStats:
    """Compute running statistics for large datasets."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self._count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._mean_of_squares = np.zeros(dim, dtype=np.float64)
        self._min = np.full(dim, np.inf, dtype=np.float64)
        self._max = np.full(dim, -np.inf, dtype=np.float64)
        
        # Sample collection for quantile computation
        self._samples: List[np.ndarray] = []
        self._max_samples = 100000
        
    def update(self, batch: np.ndarray) -> None:
        """Update statistics."""
        batch = batch.reshape(-1, batch.shape[-1]).astype(np.float64)
        n = batch.shape[0]
        
        if n == 0:
            return
            
        # Update min/max
        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)
        self._min = np.minimum(self._min, batch_min)
        self._max = np.maximum(self._max, batch_max)
        
        # Collect samples for quantile computation
        if len(self._samples) * 1000 < self._max_samples:
            sample_idx = np.random.choice(n, min(100, n), replace=False)
            self._samples.append(batch[sample_idx])
        
        # Update running mean and mean of squares
        batch_mean = np.mean(batch, axis=0)
        batch_mean_sq = np.mean(batch ** 2, axis=0)
        
        total = self._count + n
        self._mean = (self._mean * self._count + batch_mean * n) / total
        self._mean_of_squares = (self._mean_of_squares * self._count + batch_mean_sq * n) / total
        self._count = total
        
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Get statistics."""
        if self._count < 2:
            raise ValueError("Need at least 2 samples to compute statistics")
            
        variance = self._mean_of_squares - self._mean ** 2
        std = np.sqrt(np.maximum(0, variance))
        
        # Compute quantiles
        all_samples = np.concatenate(self._samples, axis=0) if self._samples else np.zeros((1, self.dim))
        q01 = np.percentile(all_samples, 1, axis=0)
        q99 = np.percentile(all_samples, 99, axis=0)
        
        return {
            "mean": self._mean.astype(np.float32),
            "std": std.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "min": self._min.astype(np.float32),
            "max": self._max.astype(np.float32),
            "count": int(self._count),
        }


def compute_norm_stats(
    data_dir: str,
    subsets: List[str] = None,
    output_path: Optional[str] = None,
    state_orientation_format: str = "axis_angle",
    skip_bad_files: bool = False,
    allow_incomplete: bool = False,
    validate_only: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute LIBERO dataset normalization statistics.
    
    Args:
        data_dir: LIBERO dataset root directory
        subsets: Subsets to include, default ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
        output_path: Output JSON path
        
    Returns:
        Dictionary containing state and actions statistics
    """
    if subsets is None:
        subsets = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    
    print(f"Computing LIBERO normalization statistics")
    print(f"   Data directory: {data_dir}")
    print(f"   Subsets: {subsets}")
    if state_orientation_format not in {"axis_angle", "euler"}:
        raise ValueError(f"Unsupported state_orientation_format: {state_orientation_format}")
    ori_label = "ee_ori_axis_angle" if state_orientation_format == "axis_angle" else "ee_ori_euler"
    print(f"   State dimension: 8 [ee_pos(3), {ori_label}(3), gripper(2)]")
    print(f"   Actions dimension: 7 [delta_xyz(3), delta_euler(3), gripper_cmd(1)]")
    
    resolved_subsets = resolve_subset_dirs(data_dir, subsets)
    valid_files_by_subset: Dict[str, List[str]] = {}
    subset_stats: Dict[str, Dict[str, int]] = {}
    bad_files: List[Dict[str, str]] = []
    expected_total_demos = 0
    expected_total_steps = 0

    print("\nPreflight validation")
    for subset, subset_dir in resolved_subsets:
        h5_files = sorted(glob.glob(os.path.join(subset_dir, "*.hdf5")))
        valid_files_by_subset[subset] = []
        subset_files = 0
        subset_demos = 0
        subset_steps = 0

        print(f"   Checking {subset}: {len(h5_files)} files")
        for h5_path in h5_files:
            try:
                num_demos, num_steps = inspect_h5_for_norm(h5_path)
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

            valid_files_by_subset[subset].append(h5_path)
            subset_files += 1
            subset_demos += num_demos
            subset_steps += num_steps

        subset_stats[subset] = {
            "num_files": subset_files,
            "num_demos": subset_demos,
            "num_steps": subset_steps,
        }
        expected_total_demos += subset_demos
        expected_total_steps += subset_steps
        print(f"      readable: {subset_files} files, {subset_demos} demos, {subset_steps} steps")

    if bad_files and not skip_bad_files:
        raise RuntimeError(
            f"Found {len(bad_files)} bad HDF5 file(s). "
            "The official full LIBERO split should not skip any files.\n"
            f"{format_h5_errors(bad_files)}"
        )

    validate_expected_counts(
        subset_stats,
        expected_total_steps,
        allow_incomplete or skip_bad_files,
    )

    if validate_only:
        print("\nValidation complete")
        print(f"   Total demos: {expected_total_demos}")
        print(f"   Total steps: {expected_total_steps}")
        return {}

    # Initialize statistics
    state_stats = RunningStats(dim=8)
    action_stats = RunningStats(dim=7)

    total_demos = 0
    total_steps = 0

    # Iterate through all subsets
    for subset, h5_files in valid_files_by_subset.items():
        print(f"\nProcessing {subset}: {len(h5_files)} files")
        
        for h5_path in tqdm(h5_files, desc=subset):
            try:
                with h5py.File(h5_path, "r") as f:
                    if "data" not in f:
                        continue
                    data_grp = f["data"]
                    
                    for demo_key in data_grp.keys():
                        demo = data_grp[demo_key]
                        
                        # Check required keys
                        if "actions" not in demo:
                            continue
                        
                        # Load data
                        actions = np.array(demo["actions"])  # [T, 7]
                        
                        # Build state
                        ee_pos = np.array(demo["obs/ee_pos"]) if "obs/ee_pos" in demo else np.zeros((len(actions), 3))
                        ee_ori_euler = np.array(demo["obs/ee_ori"]) if "obs/ee_ori" in demo else np.zeros((len(actions), 3))
                        if state_orientation_format == "axis_angle":
                            ee_ori = euler_to_axisangle(ee_ori_euler)
                        else:
                            ee_ori = ee_ori_euler
                        gripper = np.array(demo["obs/gripper_states"]) if "obs/gripper_states" in demo else np.zeros((len(actions), 2))
                        
                        T = min(len(actions), len(ee_pos), len(ee_ori), len(gripper))
                        
                        state = np.concatenate([
                            ee_pos[:T],
                            ee_ori[:T],
                            gripper[:T]
                        ], axis=-1).astype(np.float32)
                        
                        actions = actions[:T].astype(np.float32)
                        
                        # Update statistics
                        state_stats.update(state)
                        action_stats.update(actions)
                        
                        total_demos += 1
                        total_steps += T
                        
            except Exception as e:
                if skip_bad_files:
                    print(f"Error processing {h5_path}: {e}")
                    continue
                raise RuntimeError(f"Error processing {h5_path}: {e}") from e
    
    print(f"\nStatistics computation complete")
    print(f"   Total demos: {total_demos}")
    print(f"   Total steps: {total_steps}")

    if total_demos != expected_total_demos or total_steps != expected_total_steps:
        raise RuntimeError(
            "Computed sample counts differ from preflight validation: "
            f"{total_demos} demos/{total_steps} steps vs "
            f"{expected_total_demos} demos/{expected_total_steps} steps"
        )
    
    # Get statistics
    state_norm_stats = state_stats.get_statistics()
    action_norm_stats = action_stats.get_statistics()
    
    # Print results
    if state_orientation_format == "axis_angle":
        state_labels = ["ee_x", "ee_y", "ee_z", "ori_ax", "ori_ay", "ori_az", "grip_0", "grip_1"]
    else:
        state_labels = ["ee_x", "ee_y", "ee_z", "ori_r", "ori_p", "ori_y", "grip_0", "grip_1"]
    action_labels = ["dx", "dy", "dz", "dr", "dp", "dyaw", "gripper"]
    
    print(f"\nState (8-dim) statistics:")
    print(f"{'dim':<10} {'mean':>10} {'std':>10} {'q01':>10} {'q99':>10}")
    print("-" * 50)
    for i, label in enumerate(state_labels):
        print(f"{label:<10} {state_norm_stats['mean'][i]:>10.4f} {state_norm_stats['std'][i]:>10.4f} "
              f"{state_norm_stats['q01'][i]:>10.4f} {state_norm_stats['q99'][i]:>10.4f}")
    
    print(f"\nActions (7-dim) statistics:")
    print(f"{'dim':<10} {'mean':>10} {'std':>10} {'q01':>10} {'q99':>10}")
    print("-" * 50)
    for i, label in enumerate(action_labels):
        print(f"{label:<10} {action_norm_stats['mean'][i]:>10.4f} {action_norm_stats['std'][i]:>10.4f} "
              f"{action_norm_stats['q01'][i]:>10.4f} {action_norm_stats['q99'][i]:>10.4f}")
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "norm_stats": {
                "state": {
                    "mean": state_norm_stats["mean"].tolist(),
                    "std": state_norm_stats["std"].tolist(),
                    "q01": state_norm_stats["q01"].tolist(),
                    "q99": state_norm_stats["q99"].tolist(),
                },
                "actions": {
                    "mean": action_norm_stats["mean"].tolist(),
                    "std": action_norm_stats["std"].tolist(),
                    "q01": action_norm_stats["q01"].tolist(),
                    "q99": action_norm_stats["q99"].tolist(),
                },
            },
            "metadata": {
                "data_dir": data_dir,
                "subsets": subsets,
                "resolved_subsets": [
                    {"name": name, "path": path} for name, path in resolved_subsets
                ],
                "subset_stats": subset_stats,
                "num_demos": total_demos,
                "num_steps": total_steps,
                "state_dim": 8,
                "action_dim": 7,
                "state_orientation_source": "obs/ee_ori euler xyz",
                "state_orientation_format": state_orientation_format,
                "state_labels": state_labels,
                "action_labels": action_labels,
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
            
        print(f"\nSaved to: {output_path}")
        
    return {"state": state_norm_stats, "actions": action_norm_stats}


def main():
    parser = argparse.ArgumentParser(description="Compute LIBERO normalization statistics")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="LIBERO dataset root directory")
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=["libero_10", "libero_goal", "libero_object", "libero_spatial"],
                        help="Subsets to include (default 4 subsets, excluding libero_90)")
    parser.add_argument("--output", type=str,
                        default="./norm_stats/libero_norm.json",
                        help="Output file path")
    parser.add_argument("--state_orientation_format", type=str,
                        choices=["axis_angle", "euler"],
                        default="axis_angle",
                        help="State orientation format used for norm stats")
    parser.add_argument("--validate_only", action="store_true",
                        help="Validate LIBERO HDF5 files/counts without computing statistics")
    parser.add_argument("--skip_bad_files", action="store_true",
                        help="Skip unreadable files instead of failing (debug only)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Do not enforce official LIBERO file/demo/step counts")
    
    args = parser.parse_args()
    
    compute_norm_stats(
        data_dir=args.data_dir,
        subsets=args.subsets,
        output_path=args.output,
        state_orientation_format=args.state_orientation_format,
        skip_bad_files=args.skip_bad_files,
        allow_incomplete=args.allow_incomplete,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
