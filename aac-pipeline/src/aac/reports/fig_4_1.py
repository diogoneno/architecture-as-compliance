"""Figure 4.1: reliability diagram (15 equal-width bins) of the R1 chronological full-pipeline
seed-0 run before (raw_score) and after (calibrated_probability) C4, via
aac.metrics.calibration.reliability_bins. Every reliability marker has one fixed size (the
per-bin counts are carried by the histogram beneath, not by marker area). PNG only, fixed size and dpi, `Software` metadata
removed; the run's config_hash and data_version are written as tEXt chunks so reports/INDEX.json
can list the figure's provenance without a sidecar."""
from __future__ import annotations

from pathlib import Path

from aac.metrics.calibration import BINS, ece15, reliability_bins
from aac.reports import sources
from aac.run.paths import reports_dir

INK = "#1f2933"
MUTED = "#6b7280"
BEFORE = "#9ca3af"
AFTER = "#1d4ed8"
FILE = "fig-4.1-reliability.png"
MARKER_S = 36  # fixed area (pt^2) of every reliability marker, both series; the histogram beneath carries the counts
LEGEND_MARKER_S = MARKER_S  # legend handles use the same fixed area


def draw_legend(ax, host=None, anchor=(0.0, -0.22)):
    """Legend OUTSIDE the plotting axes: handles/labels come from `ax`, the legend is attached to `host`
 (default `ax`) with its upper-left corner at `anchor` in that host's axes coordinates (below it by
 default). Handles are set to the fixed LEGEND_MARKER_S so the legend always matches the data markers."""
    host = host if host is not None else ax
    handles, labels = ax.get_legend_handles_labels()
    leg = host.legend(handles, labels, loc="upper left", bbox_to_anchor=anchor, borderaxespad=0.0,
                      fontsize=8, frameon=False, scatterpoints=1)
    for handle in leg.legend_handles:
        handle.set_sizes([LEGEND_MARKER_S])
    return leg


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    state, entries = sources.require_runs(root, regime="R1", name="r1-chrono-full", table="fig-4.1", row_ident="seed 0", col_key="reliability")
    if state == "cut":
        raise sources.MissingRun("fig-4.1: the R1 chronological full run is never blanked by a cut")
    seed0 = [e for e in entries if int(e["seed"]) == 0]
    if not seed0:
        raise sources.MissingRun("fig-4.1 needs the seed-0 r1-chrono-full run")
    run = sources.load_run(root, seed0[0])
    recs = run.split("test")
    y = sources.positives_for(root, run, recs)
    raw = sources.arr(recs, "raw_score")
    cal = sources.arr(recs, "calibrated_probability")
    before = reliability_bins(raw, y, BINS)
    after = reliability_bins(cal, y, BINS)
    fig = plt.figure(figsize=(5.0, 7.0), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.08, left=0.14, right=0.96, top=0.94, bottom=0.20)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    for rows, colour, label, p in ((before, BEFORE, "before C4 (raw score)", raw), (after, AFTER, "after C4 (calibrated)", cal)):
        xs = [r["confidence"] for r in rows if r["n"]]
        ys = [r["accuracy"] for r in rows if r["n"]]
        ax.plot(xs, ys, color=colour, linewidth=1.5, zorder=2)
        ax.scatter(xs, ys, s=MARKER_S, color=colour, edgecolor="white", linewidth=1.0, zorder=3,
                   label=f"{label}; ECE {ece15(p, y):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("Observed share Refuted", color=INK)
    ax.set_title(f"Reliability, full pipeline, chronological test set, seed {run.seed} ({BINS} bins)", fontsize=9, color=INK)
    series = [(BEFORE, before), (AFTER, after)]
    counts = [r["n"] for _, rows in series for r in rows]
    nz = [c for c in counts if c > 0]
    use_log = bool(nz and max(nz) / min(nz) > 100)
    w = (1.0 / BINS) * 0.85 / len(series)
    for i, (colour, rows) in enumerate(series):
        centres = [(r["lo"] + r["hi"]) / 2 + (i - (len(series) - 1) / 2) * w for r in rows]
        axh.bar(centres, [r["n"] for r in rows], width=w, color=colour, zorder=2)
    if use_log:
        axh.set_yscale("log")
    axh.set_xlim(0, 1)
    axh.set_xlabel("Predicted P(false) (bin mean)", color=INK)
    axh.set_ylabel("Items per bin" + (" (log)" if use_log else ""), color=INK)
    axh.set_axisbelow(True)
    for a in (ax, axh):
        a.grid(True, color="#e5e7eb", linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED)
        a.tick_params(colors=MUTED, labelsize=8)
    draw_legend(ax, host=axh, anchor=(0.0, -0.45))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FILE
    fig.savefig(target, format="png", dpi=150,
                metadata={"Software": None, "config_hash": run.config_hash, "data_version": run.data_version})
    plt.close(fig)
    return [target]
