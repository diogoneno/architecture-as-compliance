"""env.json: interpreter and library versions plus the lockfile hash (an input file)."""
from __future__ import annotations

import os
import platform
from importlib import metadata
from pathlib import Path

from aac.run.hashing import dumps_sorted, sha256_file

VERSIONED = ("torch", "transformers", "numpy")
# Process-environment pins recorded only when set (scripts/gpu_run_remote.sh exports CUBLAS_WORKSPACE_CONFIG
# before Python starts; gpu-host-run jobs set CUDA_VISIBLE_DEVICES empty). Unset -> key absent, so env.json under

ENV_PINS = {"cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG", "cuda_visible_devices": "CUDA_VISIBLE_DEVICES"}


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "absent"


def env_block(lock_path: Path) -> dict:
    block = {"python": platform.python_version(), "lock_sha256": sha256_file(lock_path)}
    for dist in VERSIONED:
        block[dist] = _version(dist)
    return block


def write_env(run_dir: Path, lock_path: Path) -> dict:
    env = env_block(lock_path)
    env["platform"] = f"{platform.system()}-{platform.machine()}"
    for key, var in ENV_PINS.items():
        if var in os.environ:
            env[key] = os.environ[var]
    (run_dir / "env.json").write_text(dumps_sorted(env), encoding="ascii")
    return env
