"""C3: fusion_inputs and raw_score = P(false); creates the RecordBuilder (the only point where
every upstream output is available). StubFusion uses a fixed logistic; ContentOnlyFusion passes
P(Refuted) through; LogisticGate is the logistic gate fitted on val_fuse."""
from __future__ import annotations

import math

import numpy as np

from aac.pipeline.record import RecordBuilder

FEATURES = ("encoder_probs", "support_agg", "refute_agg", "top_rerank_score", "n_evidence")


class StubFusion:
    name = "stub-logistic-v1"

    def __init__(self, cfg: dict) -> None:
        if tuple(cfg["features"]) != FEATURES:
            raise ValueError(f"c3.features must be {list(FEATURES)}")
        self.w = [float(x) for x in cfg["weights"]]
        if len(self.w) != 8:
            raise ValueError("c3.weights needs 8 entries: 4 encoder probs, support, refute, top, n_evidence/5")
        self.b = float(cfg["bias"])

    def fuse(self, ctx, req, c1_out: dict, c2_out: dict | None) -> RecordBuilder:
        probs = c1_out["encoder_distribution"]
        if c2_out is None or not c2_out["evidence_ids"]:
            support = refute = top = 0.0
            n = 0
        else:
            scores, stances = c2_out["evidence_scores"], c2_out["_stances"]
            sup = [s for s, st in zip(scores, stances) if st == 0]
            ref = [s for s, st in zip(scores, stances) if st == 1]
            support = sum(sup) / len(sup) if sup else 0.0
            refute = sum(ref) / len(ref) if ref else 0.0
            top = max(scores)
            n = len(scores)
        x = [*probs, support, refute, top, n / 5.0]
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        raw = round(1.0 / (1.0 + math.exp(-z)), 6)
        fuse_out = {
            "fusion_inputs": {
                "encoder_probs": [float(p) for p in probs],
                "support_agg": round(support, 6),
                "refute_agg": round(refute, 6),
                "top_rerank_score": round(top, 6),
                "n_evidence": n,
            },
            "raw_score": raw,
        }
        clean_c2 = None if c2_out is None else {k: v for k, v in c2_out.items() if not k.startswith("_")}
        return RecordBuilder(ctx, req, c1_out, clean_c2, fuse_out)


class ContentOnlyFusion:
    """C3 for `c3.gate: content-only-passthrough` (pre-registered specification) =
 encoder_distribution[1]; the evidence features are their neutral values. No fitted weights."""

    name = "content-only-passthrough"

    def __init__(self, cfg: dict) -> None:
        if tuple(cfg["features"]) != FEATURES:
            raise ValueError(f"c3.features must be {list(FEATURES)}")
        if cfg.get("weights"):
            raise ValueError("content-only-passthrough takes no weights (c3.weights must be [])")

    def fuse(self, ctx, req, c1_out: dict, c2_out: dict | None) -> RecordBuilder:
        probs = c1_out["encoder_distribution"]
        fuse_out = {
            "fusion_inputs": {"encoder_probs": [float(p) for p in probs], "support_agg": 0.0, "refute_agg": 0.0,
                              "top_rerank_score": 0.0, "n_evidence": 0},
            "raw_score": round(float(probs[1]), 6),
        }
        clean_c2 = None if c2_out is None else {k: v for k, v in c2_out.items() if not k.startswith("_")}
        return RecordBuilder(ctx, req, c1_out, clean_c2, fuse_out)


# --- : the real C3 gate -------------------------------------------------

REAL_FEATURES = ("encoder_probs", "support_agg", "refute_agg", "top_rerank_score", "n_evidence",
                 "evidence_coverage")
TRANSFORM = "logratio-encoder-v1"
DESIGN_COLUMNS = ("log_refuted_vs_supported", "log_nee_vs_supported", "log_conflicting_vs_supported",
                  "support_agg", "refute_agg", "top_rerank_score", "n_evidence_over_k_capped",
                  "evidence_coverage")
LOG_EPS = 1e-6
PARAM_DECIMALS = 6


def evidence_aggregates(c2_out: dict | None) -> tuple[float, float, float, int, bool]:
    """(support_agg, refute_agg, top_rerank_score, n_evidence, evidence_coverage) from the real C2
 output: max-aggregates (operational constraint), 6 d.p.; retrieval off (None) gives zeros and no coverage."""
    if c2_out is None:
        return 0.0, 0.0, 0.0, 0, False
    sup = [float(v) for v in c2_out.get("_support") or []]
    ref = [float(v) for v in c2_out.get("_refute") or []]
    scores = [float(v) for v in c2_out["evidence_scores"]]
    return (round(max(sup), 6) if sup else 0.0, round(max(ref), 6) if ref else 0.0,
            round(max(scores), 6) if scores else 0.0, len(c2_out["evidence_ids"]),
            bool(c2_out["evidence_coverage"]))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticGate:
    """C3 for `c3.gate: logistic` (pre-registered specification
 the rows C3 collects itself in fit mode (the logged val_fuse pass). The fitted fuse evaluates
 the logistic with the 6-d.p. parameters in numpy float64; it never calls sklearn predict, so a
 gate restored with load from the sealed fusion.json gives byte-identical raw scores."""

    name = "logistic-gate-v1"

    def __init__(self, cfg: dict) -> None:
        if tuple(cfg["features"]) != REAL_FEATURES:
            raise ValueError(f"c3.features must be {list(REAL_FEATURES)}")
        if cfg["transform"] != TRANSFORM:
            raise ValueError(f"c3.transform must be {TRANSFORM!r}")
        if cfg.get("standardise") is not True:
            raise ValueError("c3.standardise must be true (the scaler is part of the sealed report)")
        self.C = float(cfg["C"])
        self.solver = str(cfg["solver"])
        self.max_iter = int(cfg["max_iter"])
        self.k = int(cfg["k"])
        if self.k <= 0 or self.C <= 0 or self.max_iter <= 0:
            raise ValueError("c3.k, c3.C and c3.max_iter must be positive")
        self._fit_mode = False
        self._buffer: list[tuple[str, list[float]]] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self.report: dict = {}

    def design_row(self, c1_out: dict, c2_out: dict | None) -> list[float]:
        """8 floats: three encoder log-ratios against Supported (label order of LABELS), support_agg,
 refute_agg, top_rerank_score, min(n_evidence / k, 1), evidence_coverage as 0/1."""
        p_s, p_r, p_nee, p_c = (float(p) for p in c1_out["encoder_distribution"])
        support, refute, top, n, cov = evidence_aggregates(c2_out)
        base = math.log(p_s + LOG_EPS)
        return [math.log(p_r + LOG_EPS) - base, math.log(p_nee + LOG_EPS) - base,
                math.log(p_c + LOG_EPS) - base, support, refute, top, min(n / self.k, 1.0),
                1.0 if cov else 0.0]

    def start_fit(self) -> None:
        self._fit_mode = True
        self._buffer = []

    def fuse(self, ctx, req, c1_out: dict, c2_out: dict | None) -> RecordBuilder:
        probs = c1_out["encoder_distribution"]
        support, refute, top, n, _ = evidence_aggregates(c2_out)
        row = self.design_row(c1_out, c2_out)
        if self._fit_mode:
            self._buffer.append((req.claim_id, row))
            raw = round(float(probs[1]), 6)  # fit pass: C1 P(Refuted) passthrough
        elif self._coef is not None:
            x = np.asarray(row, dtype=np.float64)
            z = float(np.sum(((x - self._mean) / self._scale) * self._coef) + self._intercept)
            raw = round(_sigmoid(z), 6)
        else:
            raise RuntimeError("LogisticGate.fuse before start_fit, fit or load")
        fuse_out = {
            "fusion_inputs": {"encoder_probs": [float(p) for p in probs], "support_agg": support,
                              "refute_agg": refute, "top_rerank_score": top, "n_evidence": n},
            "raw_score": raw,
        }
        clean_c2 = None if c2_out is None else {k: v for k, v in c2_out.items() if not k.startswith("_")}
        return RecordBuilder(ctx, req, c1_out, clean_c2, fuse_out)

    def fit(self, y_by_claim: dict) -> dict:
        """Fit on the fit-mode buffer, in buffer order. y = 1 for Refuted (the positive class)."""
        if not self._fit_mode:
            raise RuntimeError("LogisticGate.fit without start_fit")
        if not self._buffer:
            raise ValueError("LogisticGate.fit: the fit-mode buffer is empty")
        missing = [cid for cid, _ in self._buffer if cid not in y_by_claim]
        if missing:
            raise ValueError(f"LogisticGate.fit: no label for {len(missing)} buffered claims, e.g. {missing[0]!r}")
        X = np.asarray([row for _, row in self._buffer], dtype=np.float64)
        y = np.asarray([int(y_by_claim[cid]) for cid, _ in self._buffer], dtype=int)
        if not np.all(np.isfinite(X)):
            raise ValueError("LogisticGate.fit: non-finite feature row")
        if not set(np.unique(y).tolist()) <= {0, 1}:
            raise ValueError("LogisticGate.fit: labels must be 0 or 1")
        if len(np.unique(y)) < 2:
            raise ValueError("LogisticGate.fit: val_fuse holds one class only")
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X)
        lr = LogisticRegression(C=self.C, solver=self.solver, max_iter=self.max_iter)
        lr.fit(scaler.transform(X), y)
        mean = np.round(np.asarray(scaler.mean_, dtype=np.float64), PARAM_DECIMALS)
        scale = np.round(np.asarray(scaler.scale_, dtype=np.float64), PARAM_DECIMALS)
        if np.any(scale == 0.0):
            raise ValueError("LogisticGate.fit: a feature scale rounds to 0 at 6 d.p.")
        coef = np.round(np.asarray(lr.coef_[0], dtype=np.float64), PARAM_DECIMALS)
        intercept = round(float(lr.intercept_[0]), PARAM_DECIMALS)
        report = {
            "gate": self.name, "features": list(REAL_FEATURES), "transform": TRANSFORM,
            "design_columns": list(DESIGN_COLUMNS), "scaler_mean": [float(v) for v in mean],
            "scaler_scale": [float(v) for v in scale], "coef": [float(v) for v in coef],
            "intercept": intercept, "C": self.C, "solver": self.solver, "max_iter": self.max_iter,
            "n_iter": int(np.max(lr.n_iter_)), "k": self.k, "n_val_fuse": int(len(y)),
            "n_positive": int(y.sum()), "sklearn_version": sklearn.__version__,
        }
        self._fit_mode = False
        self._buffer = []
        return self.load(report)

    def load(self, report: dict) -> dict:
        """Restore a fitted gate from a sealed fusion.json."""
        if tuple(report["features"]) != REAL_FEATURES or report["transform"] != TRANSFORM \
                or tuple(report["design_columns"]) != DESIGN_COLUMNS:
            raise ValueError("fusion report does not match the logratio-encoder-v1 feature set")
        arrays = [np.asarray(report[key], dtype=np.float64) for key in ("scaler_mean", "scaler_scale", "coef")]
        if any(a.shape != (len(DESIGN_COLUMNS),) for a in arrays):
            raise ValueError("fusion report: scaler_mean, scaler_scale and coef need 8 entries each")
        self._mean, self._scale, self._coef = arrays
        self._intercept = float(report["intercept"])
        self.k = int(report["k"])
        self._fit_mode = False
        self.report = dict(report)
        return self.report
