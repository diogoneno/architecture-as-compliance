"""Deflection-controls figure (gap review addition B output "deflection vs control per recipe"; appendix
figure, the one added visual): per recipe, the registered deflection of victim B with its interval beside the
deflection references (a), (d) and (e) from the Table 4.4 long CSV (aac.reports.table_4_4, family R3-supplementary).

Two panels on one x axis of recipes (small multiples, never a dual axis): left, rates in percent (the registered
deflection rate and reference (d), conditional deflection); right, differences in percentage points (reference (a),
deflection minus clean deferral, and reference (e), the matched-confidence control) around a zero line. A value that
reads `n/a ...` in the long CSV is not drawn (no invented point); an interval is drawn only where the long row has one.

Real-only (registry.REAL_ONLY): a namespace whose Phase 4 profile is not `real` (the stub chain, the tiny world) gets
no file, so the stub goldens gain none. PNG only with `metadata={"Software": None, ...}` (deterministic on matplotlib
3.11.2, as fig_4_1/fig_4_2), fixed dpi and size; the source table's config_hash and data_version are tEXt chunks for
reports/INDEX.json. Output: reports/fig-deflection-controls.png."""
from __future__ import annotations

from pathlib import Path

from aac.reports import fmt, phase4_sources
from aac.run.paths import reports_dir

FIGURE = "fig-deflection-controls"
FILE = "fig-deflection-controls.png"
LONG = Path("long") / "table-4.4-r3.long.csv"
RECIPES = ("BERT-Attack", "TextFooler", "DeepWordBug", "PWWS", "BeamAttack")
COL_DEFL = "Deflection rate (percent)"
COL_A = "deflection reference (a): deflection minus clean deferral, same items (pp)"
COL_D = "deflection reference (d): conditional deflection, deflected / (deflected + strict success) (percent)"
COL_E = "deflection reference (e): matched-confidence control, observed minus matched clean deferral (pp)"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e7eb"
SERIES_1 = "#2a78d6"  # categorical slot 1 (validated pair with slot 2, light surface)
SERIES_2 = "#eb6834"  # categorical slot 2
PANELS = (
    ("Rates (percent)", ((COL_DEFL, "registered deflection", SERIES_1, "o"),
                          (COL_D, "(d) conditional deflection", SERIES_2, "s"))),
    ("Differences (percentage points)", ((COL_A, "(a) minus clean deferral", SERIES_1, "o"),
                                         (COL_E, "(e) minus matched-confidence control", SERIES_2, "s"))),
)


def _num(text: str) -> float | None:
    """The leading number of a long-CSV value (`12.5`, `+3.0`, `33.3 [10.0, 60.0]`), or None for an `n/a` value."""
    head = str(text).strip().split(" ")[0]
    try:
        return float(head)
    except ValueError:
        return None


def _read(path: Path) -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    comments, rows = fmt.read_table(path)
    if not comments or not comments[0].startswith("# config_hash="):
        raise ValueError(f"{path}: missing config_hash header comment")
    return {(r["row_label"], r["col_label"]): r for r in rows}, fmt.header_fields(comments[0])


def points(path: Path) -> dict[str, dict[str, tuple[float, float | None, float | None]]]:
    """{recipe: {column: (value, ci_lo, ci_hi)}} for the recipes whose victim-B row exists; n/a values are left out."""
    cells, _ = _read(path)
    out: dict[str, dict] = {}
    for recipe in RECIPES:
        row = f"{recipe} | Full pipeline"
        got = {}
        for col in (COL_DEFL, COL_A, COL_D, COL_E):
            r = cells.get((row, col))
            if r is None:
                continue
            v = _num(r["value"])
            if v is None:
                continue
            lo, hi = _num(r["ci_lo"]), _num(r["ci_hi"])
            got[col] = (v, lo, hi)
        if got:
            out[recipe] = got
    return out


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    if phase4_sources.select_profile(root) != "real":
        return []  # real-only (registry.REAL_ONLY): the stub chain and the tiny world write no figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    path = out_dir / LONG
    _, prov = _read(path)
    pts = points(path)
    if not pts:
        raise ValueError(f"{path}: no victim-B recipe row with a deflection value; nothing to plot")
    recipes = [r for r in RECIPES if r in pts]
    xs = list(range(len(recipes)))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=150, sharex=True)
    for ax, (title, series) in zip(axes, PANELS):
        for k, (col, label, color, marker) in enumerate(series):
            dx = -0.12 if k == 0 else 0.12
            sx, sy, lo_err, hi_err = [], [], [], []
            for x, recipe in zip(xs, recipes):
                if col not in pts[recipe]:
                    continue
                v, lo, hi = pts[recipe][col]
                sx.append(x + dx)
                sy.append(v)
                lo_err.append(v - lo if lo is not None else 0.0)
                hi_err.append(hi - v if hi is not None else 0.0)
            if not sx:
                continue
            ax.errorbar(sx, sy, yerr=[lo_err, hi_err], fmt=marker, color=color, ecolor=color, elinewidth=2.0,
                        capsize=0, markersize=8, markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
        if title.startswith("Differences"):
            ax.axhline(0.0, color=MUTED, linewidth=1.0, zorder=1)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(recipes, fontsize=8, color=INK)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7, frameon=False, labelcolor=INK)
    axes[0].set_ylabel("percent", color=INK)
    axes[1].set_ylabel("percentage points", color=INK)
    fig.suptitle("Victim B: registered deflection beside its references, per recipe (95 percent intervals)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    target = out_dir / FILE
    fig.savefig(target, format="png", dpi=150,
                metadata={"Software": None, "config_hash": prov["config_hash"], "data_version": prov["data_version"]})
    plt.close(fig)
    return [target]
