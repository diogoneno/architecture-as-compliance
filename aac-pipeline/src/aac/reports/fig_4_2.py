"""Figure 4.2: coverage versus selective risk from the Table 4.6 long rows.

PNG only (deterministic on matplotlib 3.11.2 with `metadata={"Software": None}`), fixed
dpi and figure size; one series, so the title names it and no legend box is drawn. The source
table's config_hash and data_version are written as tEXt chunks for reports/INDEX.json."""
from __future__ import annotations

from pathlib import Path

from aac.reports import fmt
from aac.run.paths import reports_dir

ROW_ORDER = ("100", "90", "80 (pre-specified)", "70")
INK = "#1f2933"
MUTED = "#6b7280"
SERIES = "#1d4ed8"


def _read_long(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Cells of the long CSV and the provenance fields of its header comment."""
    cells: dict[str, dict[str, str]] = {}
    comments, rows = fmt.read_table(path)
    if not comments or not comments[0].startswith("# config_hash="):
        raise ValueError(f"{path}: missing config_hash header comment")
    for row in rows:
        cells.setdefault(row["row_label"], {})[row["col_label"]] = row["value"]
    return cells, fmt.header_fields(comments[0])


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    cells, prov = _read_long(out_dir / "long" / "table-4.6-coverage-risk.long.csv")
    points = []
    for label in ROW_ORDER:
        cov = float(cells[label]["Coverage achieved (percent)"])
        risk = float(cells[label]["Selective risk"])
        points.append((label, cov, risk))
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    order = sorted(range(len(points)), key=lambda i: (xs[i], ys[i]))
    ax.plot([xs[i] for i in order], [ys[i] for i in order], color=SERIES, linewidth=2.0, zorder=2)
    ax.scatter(xs, ys, s=64, color=SERIES, edgecolor="white", linewidth=1.5, zorder=3)
    # one label per distinct point: coincident targets share a label instead of overprinting
    groups: dict[tuple[float, float], list[str]] = {}
    for label, cov, risk in points:
        groups.setdefault((round(cov, 3), round(risk, 3)), []).append(label.split(" ")[0])
    for (cov, risk), names in groups.items():
        text = f"target {names[0]}" if len(names) == 1 else "targets " + ", ".join(names)
        ax.annotate(text, (cov, risk), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK)
    ax.set_xlabel("Coverage achieved (percent)", color=INK)
    ax.set_ylabel("Selective risk (percent)", color=INK)
    ax.set_title("Full pipeline, chronological test set: coverage versus selective risk", fontsize=10, color=INK)
    ax.grid(True, color="#e5e7eb", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlim(min(xs) - 5, 105)
    fig.tight_layout()
    target = out_dir / "fig-4.2-coverage-risk.png"
    fig.savefig(target, format="png", dpi=150,
                metadata={"Software": None, "config_hash": prov["config_hash"], "data_version": prov["data_version"]})
    plt.close(fig)
    return [target]
