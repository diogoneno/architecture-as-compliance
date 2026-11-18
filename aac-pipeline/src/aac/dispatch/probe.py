"""host probe -> runs/_probes/<host>.json.

scripts/probe_host.sh runs ON the probed host (locally for cpu-host, over `ssh host bash -s` for gpu-host)
and prints `key=value` lines; this module parses them, adds the host-change log from
configs/host-changes.yaml, picks the torch extra from the driver and writes the JSON through
tmp + os.replace (two concurrent probes write the same facts; the replace is atomic). A probe is an
INPUT, so `probed_utc` is allowed here; the file is excluded from the determinism checks."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

import yaml

from aac.dispatch import remote as _remote
from aac.dispatch.guard import torch_extra_for
from aac.dispatch.remote import Remote, Unreachable
from aac.run.paths import root as project_root

PROBE_KEYS = ("python", "uv", "uv_pythons", "driver", "gpu", "memory_total_mib", "memory_used_mib", "compute_apps", "disk_free_gb",
              "network", "reachable", "sentinel_present", "rsync", "tmux", "os", "nproc", "ram_gb", "aac_home_present",
              "torch_extra", "host_changes", "probed_utc")
INT_KEYS = ("memory_total_mib", "memory_used_mib", "compute_apps", "disk_free_gb", "nproc", "ram_gb")
BOOL_KEYS = ("sentinel_present", "aac_home_present")


def parse_probe_output(text: str) -> dict:
    """`key=value` lines -> typed dict. `-` means the tool/fact is absent (null)."""
    raw: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        raw[k.strip()] = v.strip()

    def val(key: str):
        v = raw.get(key, "-")
        return None if v in ("", "-") else v

    out: dict = {}
    for key in ("python", "uv", "uv_pythons", "driver", "gpu", "rsync", "tmux", "os"):
        out[key] = val(key)
    for key in INT_KEYS:
        v = val(key)
        out[key] = int(v) if v is not None and v.lstrip("-").isdigit() else None
    for key in BOOL_KEYS:
        v = val(key)
        out[key] = (v == "1") if v is not None else None
    out["network"] = {k[len("net_"):]: (int(v) if v.isdigit() else None) for k, v in sorted(raw.items()) if k.startswith("net_")}
    return out


def host_changes_for(host: str, path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # yaml.safe_load turns `date: 2026-10-15` into a datetime.date; keep the JSON plain ISO strings
    return [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in c.items()}
            for c in (data.get("changes") or []) if str(c.get("host")) == host]


def run_probe_script(host: str, remote: Remote, script: Path, local: bool) -> tuple[bool, str]:
    """(reachable, stdout). Local probes run bash on the script; remote ones stream it over ssh."""
    env = {"AAC_HOME": remote.aac_home, "AAC_SENTINEL": remote.sentinel, "AAC_PROBE_NET": " ".join(remote.probe_network)}
    if local:
        proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, env={**os.environ, **env}, timeout=300)
        return True, proc.stdout
    try:
        proc = _remote.ssh("bash -s", remote=remote, stdin=script.read_text(encoding="utf-8"), env=env, timeout=300)
    except Unreachable:
        return False, ""
    return proc.returncode == 0 or bool(proc.stdout.strip()), proc.stdout


def write_probe(host: str, *, root: Path | None = None, remote: Remote | None = None, local: bool | None = None) -> Path:
    root = Path(root) if root else project_root()
    rem = remote or _remote.load_remote()
    if local is None:
        local = host != rem.host
    reachable, out = run_probe_script(host, rem, root / "scripts" / "probe_host.sh", local)
    facts = parse_probe_output(out) if reachable else parse_probe_output("")
    record = {key: facts.get(key) for key in PROBE_KEYS}
    record["host"] = host
    record["reachable"] = bool(reachable)
    record["torch_extra"] = torch_extra_for(record["driver"], rem.torch_extra_by_driver)
    record["host_changes"] = host_changes_for(host, root / "configs" / "host-changes.yaml")
    record["probed_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dest = root / "runs" / "_probes" / f"{host}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)
    return dest
