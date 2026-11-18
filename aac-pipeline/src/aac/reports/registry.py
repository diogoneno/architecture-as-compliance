"""Table id -> producer(s). Every id in contracts/table-shells.json is registered here, so the
shell-header test binds every shell to a producer . Generators are named lazily
(module path + attribute) and imported on use; `External` names a producer outside
`aac reports build` (a script or the pack builder) so the shell test can still bind its header
once the file exists. The CLI derives its --table choices from the buildable ids."""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Generator:
    module: str
    attr: str = "build"

    def __str__(self) -> str:
        return f"{self.module}.{self.attr}"


@dataclass(frozen=True)
class External:
    producer: str      # path relative to the aac-pipeline root (script or module file)
    location: str      # directory the shell is written to: "reports" or "evidence-pack"
    note: str = ""


REPORTS = "reports"
PACK = "evidence-pack"

# Build order matters: 4.8 reads the long CSVs of 4.2-4.6; the figures read their table's rows.
REGISTRY: dict[str, list] = {
    "4.1": [Generator("aac.reports.table_4_1")],
    "4.2": [Generator("aac.reports.table_4_2")],
    "4.3": [Generator("aac.reports.table_4_3")],
    "4.4": [Generator("aac.reports.table_4_4")],
    "4.5": [Generator("aac.reports.table_4_5")],
    "4.6": [Generator("aac.reports.table_4_6"), Generator("aac.reports.fig_4_2")],  # the figure reads the long CSV the table writes
    "deferral_reasons": [Generator("aac.reports.table_4_6")],  # companion written by the Table 4.6 generator
    "subgroup_deferral": [Generator("aac.reports.table_subgroup")],  # appendix table (methodological note
    
    "calibration_intervals": [Generator("aac.reports.calibration_intervals")],
    "4.7": [External("scripts/regen_check.sh", REPORTS,
                     note="written by verification in a different checkout; verification table write")],
    "4.8": [Generator("aac.reports.table_4_8")],
    "D.1": [Generator("aac.reports.table_d1")],
    "D.2": [Generator("aac.reports.table_d2")],
    "D.3": [External("src/aac/pack/manifest.py", PACK, note="rendered by aac pack build from evidence-pack/manifest.json (D-21)")],
    "fig-4.1": [Generator("aac.reports.fig_4_1")],
    # : the deflection-controls appendix figure (gap review addition B); reads the Table 4.4 long CSV
    "fig-deflection-controls": [Generator("aac.reports.fig_deflection")],
}
# fig-4.1 and fig-deflection-controls are figures, not shells: buildable, no table-shells.json entry.
FIGURES = ("fig-4.1", "fig-deflection-controls")
# Real-only shells : the generator writes nothing for a namespace whose runs read the fixture corpus
# (the stub smoke chain), so the stub goldens gain no file; the shell test skips them there.
REAL_ONLY = ("subgroup_deferral", "fig-deflection-controls", "calibration_intervals")

# `aac reports build --all` order (externals and the deferral_reasons alias excluded)
BUILD_ORDER = ("4.1", "4.2", "4.3", "4.4", "fig-deflection-controls", "4.5", "4.6", "subgroup_deferral", "calibration_intervals", "fig-4.1", "D.1", "D.2", "4.8")


def is_external(table_id: str) -> bool:
    return all(isinstance(s, External) for s in REGISTRY[table_id])


def location(table_id: str) -> str:
    specs = REGISTRY[table_id]
    return specs[0].location if isinstance(specs[0], External) else REPORTS


def location_dir(table_id: str, root=None) -> Path:
    """The directory a table's shell lands in, inside the selected output namespace (AAC_OUT):
 reports_dir for generators and REPORTS externals, pack_dir for PACK externals."""
    from aac.run.paths import pack_dir, reports_dir

    return pack_dir(root) if location(table_id) == PACK else reports_dir(root)


def buildable() -> list[str]:
    return [tid for tid in REGISTRY if not is_external(tid)]


def shell_ids() -> list[str]:
    return [tid for tid in REGISTRY if tid not in FIGURES]


class AllowPendingRefused(ValueError):
    """--allow-pending asked of a generator whose build has no allow_pending parameter."""


def accepts_pending(fn) -> bool:
    return "allow_pending" in inspect.signature(fn).parameters


def resolve(spec: Generator):
    mod = importlib.import_module(spec.module)
    fn = getattr(mod, spec.attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def build_table(table_id: str, root, out_dir=None, allow_pending: bool = False) -> list[Path]:
    """Run the table's generator(s). allow_pending=True is passed only to generators whose
 build takes `allow_pending`; any other generator of the table refuses (AllowPendingRefused) before
 anything runs."""
    if table_id not in REGISTRY:
        raise KeyError(f"no producer registered for table {table_id!r}")
    if is_external(table_id):
        spec = REGISTRY[table_id][0]
        raise RuntimeError(f"table {table_id} is produced by {spec.producer}, not by `aac reports build`")
    fns = [(spec, resolve(spec)) for spec in REGISTRY[table_id]]
    if allow_pending:
        for spec, fn in fns:
            if not accepts_pending(fn):
                raise AllowPendingRefused(f"generator {spec.module} does not accept --allow-pending")
    outputs: list[Path] = []
    for _, fn in fns:
        outputs.extend(fn(root, out_dir, allow_pending=True) if allow_pending else fn(root, out_dir))
    return outputs


def build_all(root, out_dir=None, allow_pending: bool = False) -> list[Path]:
    """Every buildable table and figure in BUILD_ORDER; a generator that cannot find a required
 sealed run raises (no silent blank), so `make reports` fails loudly. (The CLI never passes
 allow_pending with --all.)"""
    outputs: list[Path] = []
    for tid in BUILD_ORDER:
        outputs.extend(build_table(tid, root, out_dir, allow_pending=allow_pending))
    return outputs
