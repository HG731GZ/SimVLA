#!/usr/bin/env python3
"""
SimVLA LIBERO Evaluation Client

Observation format:
1. State: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
2. Action: delta action (7D)
3. Default delta control mode
4. Images rotated 180 degrees
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional


def _has_libero_benchmark_layout(path: Path) -> bool:
    return (
        (path / "bddl_files").is_dir()
        and (path / "init_files").is_dir()
        and (path / "assets").is_dir()
    )


def _iter_libero_root_candidates():
    seen = set()
    roots = [
        os.environ.get("LIBERO_ROOT"),
        os.environ.get("SIMVLA_LIBERO_ROOT"),
    ]
    roots.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    roots.extend(sys.path)

    for root in roots:
        if not root:
            continue
        candidate = Path(root).expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _add_libero_import_path(libero_root: Path, benchmark_root: Path) -> None:
    candidates = [libero_root]
    if benchmark_root.parent.name == "libero":
        candidates.append(benchmark_root.parent.parent)

    for candidate in candidates:
        if (candidate / "libero" / "libero" / "__init__.py").is_file():
            path = str(candidate.resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            return


def _resolve_libero_benchmark_root() -> Path:
    for root in _iter_libero_root_candidates():
        for benchmark_root in (root / "libero" / "libero", root / "libero", root):
            if _has_libero_benchmark_layout(benchmark_root):
                benchmark_root = benchmark_root.resolve()
                _add_libero_import_path(root, benchmark_root)
                return benchmark_root

    import importlib.util

    try:
        spec = importlib.util.find_spec("libero.libero")
    except ModuleNotFoundError:
        spec = None
    if spec and spec.origin:
        benchmark_root = Path(spec.origin).resolve().parent
        if _has_libero_benchmark_layout(benchmark_root):
            return benchmark_root

    raise RuntimeError(
        "Cannot locate LIBERO. Set LIBERO_ROOT to the LIBERO repository root "
        "(for example: export LIBERO_ROOT=/path/to/LIBERO)."
    )


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _default_libero_config_dir(benchmark_root: Path) -> Path:
    digest = hashlib.sha1(str(benchmark_root).encode("utf-8")).hexdigest()[:12]
    base_dir = Path(os.environ.get("TMPDIR", "/tmp")).expanduser()
    return base_dir / f"simvla_libero_config_{digest}"


def _resolve_libero_dataset_root(benchmark_root: Path) -> Path:
    for env_name in ("LIBERO_DATASETS", "LIBERO_DATASET_ROOT"):
        if os.environ.get(env_name):
            return Path(os.environ[env_name]).expanduser().resolve()

    simvla_datasets = Path(__file__).resolve().parents[2] / "datasets"
    if simvla_datasets.exists():
        return simvla_datasets.resolve()

    return (benchmark_root.parent / "datasets").resolve()


def _write_libero_config(config_file: Path, benchmark_root: Path) -> None:
    config = {
        "assets": benchmark_root / "assets",
        "bddl_files": benchmark_root / "bddl_files",
        "benchmark_root": benchmark_root,
        "datasets": _resolve_libero_dataset_root(benchmark_root),
        "init_states": benchmark_root / "init_files",
    }
    contents = "".join(f"{key}: {json.dumps(str(value))}\n" for key, value in config.items())

    config_file.parent.mkdir(parents=True, exist_ok=True)
    if config_file.exists() and config_file.read_text() == contents:
        return

    tmp_config = config_file.with_name(f"{config_file.name}.{os.getpid()}.tmp")
    tmp_config.write_text(contents)
    os.replace(tmp_config, config_file)


def _prepare_libero_config() -> None:
    if os.environ.get("SIMVLA_AUTO_LIBERO_CONFIG", "1").lower() in {"0", "false", "no"}:
        return

    benchmark_root = _resolve_libero_benchmark_root()
    repo_config_dir = Path(__file__).resolve().parent / ".libero_config"
    env_config_dir = os.environ.get("LIBERO_CONFIG_PATH")
    explicit_root = bool(os.environ.get("LIBERO_ROOT") or os.environ.get("SIMVLA_LIBERO_ROOT"))

    redirected_repo_config = False
    if env_config_dir:
        config_dir = Path(env_config_dir).expanduser()
        if _same_path(config_dir, repo_config_dir):
            config_dir = _default_libero_config_dir(benchmark_root)
            redirected_repo_config = True
    else:
        config_dir = _default_libero_config_dir(benchmark_root)

    config_dir = config_dir.resolve()
    config_file = config_dir / "config.yaml"
    should_write = explicit_root or redirected_repo_config or env_config_dir is None or not config_file.exists()

    if should_write:
        _write_libero_config(config_file, benchmark_root)

    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    print(f"LIBERO config: {config_file}")
    print(f"LIBERO benchmark root: {benchmark_root}")


_prepare_libero_config()

import imageio
import json_numpy
import numpy as np
import requests
from tqdm import tqdm

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client
    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Max steps per task suite (based on longest demo + buffer)
MAX_STEPS = {
    "libero_spatial": 800,   # longest demo: 193
    "libero_object": 800,    # longest demo: 254
    "libero_goal": 800,      # longest demo: 270
    "libero_10": 900,        # longest demo: 505
    "libero_90": 900,        # longest demo: 373
}

NUM_STEPS_WAIT = 10  # Wait for objects to stabilize

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to axis-angle representation.
    
    Uses the same convention as robosuite for consistency with training data.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# -----------------------------------------------------------------------------
# Client Policy Classes
# -----------------------------------------------------------------------------

class WebSocketClient:
    """
    WebSocket client for SimVLA server.
    
    Requires: pip install openpi-client
    """
    def __init__(
        self,
        host: str,
        port: int,
        replan_steps: int = 5,
        resize_size: int = 224,
        clip_actions: bool = True,
    ):
        if not HAS_WS_CLIENT:
            raise ImportError("openpi_client not installed. Run: pip install openpi-client")
        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.clip_actions = clip_actions
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            # Preprocess images
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], self.resize_size, self.resize_size)
            )
            
            # Build observation dict
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }
            
            # Query server
            result = self.client.infer(element)
            action_chunk = result["actions"]
            
            # Ensure numpy array
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            
            assert len(action_chunk) >= self.replan_steps, \
                f"Need {self.replan_steps} steps but got {len(action_chunk)}"
            
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        action = self.action_plan.popleft()
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        return action


class HTTPClient:
    """
    HTTP client for SimVLA server.
    """
    def __init__(self, host: str, port: int, replan_steps: int = 5, clip_actions: bool = True):
        self.url = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.clip_actions = clip_actions
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def infer(self, element: Dict) -> Dict:
        try:
            payload = {
                "image0": json_numpy.dumps(element["observation/image"]),
                "image1": json_numpy.dumps(element["observation/wrist_image"]),
                "proprio": json_numpy.dumps(element["observation/state"]),
                "language_instruction": element["prompt"],
                "steps": 10,
            }
            
            resp = requests.post(self.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            actions = np.array(data["action"])
            return {"actions": actions}
            
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            element = {
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state": obs["state"],
                "prompt": goal,
            }
            
            result = self.infer(element)
            action_chunk = result["actions"]
            
            for action in action_chunk[:self.replan_steps]:
                self.action_plan.append(action)

        action = self.action_plan.popleft()
        if self.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        return action


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int):
    """Initialize a LIBERO environment."""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(
    client,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
    task_ids: Optional[List[int]] = None,
    max_steps_override: Optional[int] = None,
) -> float:
    """
    Run LIBERO evaluation across all tasks in a suite.
    """
    np.random.seed(seed)
    
    # Initialize task suite
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = max_steps_override or MAX_STEPS.get(task_suite_name, 400)
    if task_ids is None:
        task_ids = list(range(num_tasks - 1, -1, -1))
    else:
        bad_task_ids = [task_id for task_id in task_ids if task_id < 0 or task_id >= num_tasks]
        if bad_task_ids:
            raise ValueError(f"Invalid task_ids for {task_suite_name}: {bad_task_ids}")
    
    Path(video_out_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Task suite: {task_suite_name}")
    print(f"   Tasks: {len(task_ids)}/{num_tasks}, Trials per task: {num_trials}")
    print(f"   Task IDs: {task_ids}")
    print(f"   Max steps: {max_steps}")
    
    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm(task_ids, desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        
        task_successes = 0
        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            # Reset
            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            
            replay_images = []
            t = 0
            done = False
            
            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    # Wait for objects to stabilize
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # Get images (rotated 180 degrees)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    if save_video:
                        replay_images.append(img)
                    
                    # Build state vector
                    # [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])
                    
                    # Pack observation
                    obs_dict = {
                        "image": img,
                        "wrist_image": wrist_img,
                        "state": state,
                    }
                    
                    # Get action (7D delta action)
                    action = client.step(obs_dict, task_description)
                    
                    # Execute (send delta action directly)
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    t += 1
                    
                except Exception as e:
                    print(f"Error in rollout: {e}")
                    break

            total_episodes += 1
            
            # Save video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = Path(video_out_path) / f"{task_segment}_ep{ep}_{suffix}.mp4"
            if replay_images and save_video:
                imageio.mimwrite(str(video_path), replay_images, fps=10)
            
            # Print episode result
            status_icon = "[OK]" if done else "[FAIL]"
            print(f"  {status_icon} Task {task_id} Ep {ep}: {suffix.upper()} (steps={t})")

        env.close()
        print(f"   Task {task_id}: {task_successes}/{num_trials} ({task_successes/num_trials*100:.1f}%)")
    
    success_rate = total_successes / max(total_episodes, 1)
    print(f"\nTotal success rate: {total_successes}/{total_episodes} ({success_rate*100:.1f}%)")
    
    return success_rate


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server connection info JSON")
    parser.add_argument("--client_type", type=str, default="websocket",
                        choices=["websocket", "http"],
                        help="Client type: websocket or http")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--video_out", type=str, default="./eval_results")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for faster evaluation")
    parser.add_argument("--task_ids", type=str, default=None,
                        help="Comma-separated task IDs to evaluate, e.g. '0' or '0,3,9'. Default: all tasks")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max rollout steps per episode; useful for smoke tests")
    parser.add_argument("--no_clip_actions", action="store_true",
                        help="Disable clipping policy actions to LIBERO's [-1, 1] action range")

    args = parser.parse_args()

    # Load connection info
    if args.connection_info:
        print(f"Loading connection info from: {args.connection_info}")
        while not Path(args.connection_info).exists():
            sys.stdout.write("\rWaiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
        print()
        with open(args.connection_info) as f:
            info = json.load(f)
            args.host = info["host"]
            args.port = info["port"]
    
    protocol = "ws" if args.client_type == "websocket" else "http"
    print(f"Starting LIBERO evaluation client")
    print(f"   Client type: {args.client_type}")
    print(f"   Server: {protocol}://{args.host}:{args.port}")
    print(f"   Task suite: {args.task_suite}")
    print(f"   Replan steps: {args.replan_steps}")
    print(f"   Clip actions: {not args.no_clip_actions}")
    print()
    
    # Initialize client
    if args.client_type == "websocket":
        client = WebSocketClient(
            args.host,
            args.port,
            replan_steps=args.replan_steps,
            clip_actions=not args.no_clip_actions,
        )
    else:
        client = HTTPClient(args.host, args.port, replan_steps=args.replan_steps, clip_actions=not args.no_clip_actions)

    task_ids = None
    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.replace(" ", "").split(",") if x]
    
    # Run evaluation
    video_path = Path(args.video_out) / args.task_suite
    eval_libero(
        client=client,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        video_out_path=str(video_path),
        save_video=not args.no_video,
        task_ids=task_ids,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()
