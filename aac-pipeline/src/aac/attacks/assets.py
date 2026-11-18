"""Runtime-asset preflight : the OpenAttack data and LAMBO models that live under .hf/
(not in git) are checked against the sha256 values recorded in third_party/PINS.json
`runtime_assets_not_in_git` before any attack loads them (the OpenAttack assets are pickles:
a substituted file would be code execution in the attack environment).

 file entries "<relpath>": "<sha256>" -> sha256 of .hf/openattack/data/<relpath>
 tree entries "... listing = <sha256>" -> sha256 over the `sha256sum` listing of every
 file under the directory, paths relative to
 .hf/openattack/data, sorted by path (LC_ALL=C)
 other entries prose with no digest (e.g. WordNet) -> reported as not pinned, not checked
 LAMBO "<file>": "<sha256>" -> sha256 of .hf/lambo/<file>

`python -m aac.attacks.assets` prints one line per asset and exits 1 on any mismatch or missing file
(scripts/env_gate.sh step 8 and the start of step 5). Stdlib only."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

PINS = Path("third_party") / "PINS.json"
OPENATTACK_DATA = Path(".hf") / "openattack" / "data"
LAMBO_DIR = Path(".hf") / "lambo"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LISTING = re.compile(r"listing = ([0-9a-f]{64})")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(base: Path, rel_dir: str) -> str:
    """sha256 of `find <rel_dir> -type f | LC_ALL=C sort | xargs sha256sum` run inside base."""
    files = sorted((p for p in (Path(base) / rel_dir).rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(base).as_posix().encode("utf-8"))
    if not files:
        raise FileNotFoundError(f"{Path(base) / rel_dir}: no files")
    listing = "".join(f"{_sha256(p)}  {p.relative_to(base).as_posix()}\n" for p in files)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def verify_runtime_assets(root: Path) -> tuple[list[str], list[str]]:
    """(ok lines, problem lines) for every pinned runtime asset under root/.hf."""
    root = Path(root)
    pins = json.loads((root / PINS).read_text(encoding="utf-8"))["runtime_assets_not_in_git"]
    checks: list[tuple[str, Path | None, str | None, str]] = []  # (label, file, expected, kind)
    for rel, val in sorted(pins.get("openattack_assets_sha256", {}).items()):
        if _HEX64.match(val):
            checks.append((f"openattack/{rel}", root / OPENATTACK_DATA / rel, val, "file"))
        elif _LISTING.search(val):
            checks.append((f"openattack/{rel}", None, _LISTING.search(val).group(1), "tree"))
        else:
            checks.append((f"openattack/{rel}", None, None, "unpinned"))
    for rel, val in sorted(pins.get("lambo_models_sha256", {}).items()):
        checks.append((f"lambo/{rel}", root / LAMBO_DIR / rel, val, "file"))
    if not checks:
        return [], [f"{PINS}: runtime_assets_not_in_git pins no asset"]
    ok, bad = [], []
    for label, path, expected, kind in checks:
        if kind == "unpinned":
            ok.append(f"assets: {label} not pinned by digest in {PINS} (not checked)")
            continue
        try:
            if kind == "file":
                if path is None or expected is None:
                    bad.append(f"assets: {label} INVALID CONFIGURATION")
                    continue
                got = _sha256(path)
            else:
                if expected is None:
                    bad.append(f"assets: {label} INVALID CONFIGURATION")
                    continue
                got = tree_digest(root / OPENATTACK_DATA, label.split("/", 1)[1])
        except (FileNotFoundError, IsADirectoryError) as exc:
            bad.append(f"assets: {label} MISSING ({exc})")
            continue
        (ok if got == expected else bad).append(
            f"assets: {label} {'ok' if got == expected else 'MISMATCH'} {got[:12]}" + ("" if got == expected else f" != pinned {expected[:12]}"))
    return ok, bad


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path.cwd()
    ok, bad = verify_runtime_assets(root)
    for line in ok + bad:
        print(line)
    print(f"assets: {len(ok)} ok/unpinned, {len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
