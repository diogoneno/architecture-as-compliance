"""C4: temperature or isotonic, chosen by lower out-of-fold validation ECE (15 equal-width bins).
StubCalibration uses a fixed T; FittedCalibration fits its own T by a bounded
1-D NLL search and refits both maps inside each fold."""
from __future__ import annotations

import numpy as np

from aac.metrics.selective import ece


def _temperature(scores: np.ndarray, t: float) -> np.ndarray:
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    logit = np.log(s / (1 - s)) / t
    return 1.0 / (1.0 + np.exp(-logit))


class StubCalibration:
    name = "stub-calibration-v1"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.t = float(cfg["temperature"])
        self.bins = int(cfg["bins"])
        self.method: str | None = None
        self._iso_x: np.ndarray | None = None
        self._iso_y: np.ndarray | None = None
        self.report: dict = {}

    def fit(self, raw_scores: list[float], y_false: list[int]) -> dict:
        """Select the method by out-of-fold validation ECE (2-fold by index parity for the
 fitted isotonic map; the fixed-T temperature map has no fitted parameter), then
 refit the chosen map on all of val_cal."""
        s = np.asarray(raw_scores, dtype=float)
        y = np.asarray(y_false, dtype=float)
        oof: dict[str, np.ndarray] = {}
        if "temperature" in self.cfg["methods"]:
            oof["temperature"] = _temperature(s, self.t)
        if "isotonic" in self.cfg["methods"]:
            from sklearn.isotonic import IsotonicRegression

            pred = np.empty_like(s)
            idx = np.arange(len(s))
            for fold in (0, 1):
                train, held = idx % 2 != fold, idx % 2 == fold
                if train.sum() < 2 or held.sum() == 0:
                    pred[held] = s[held]
                    continue
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
                iso.fit(s[train], y[train])
                pred[held] = iso.predict(s[held])
            oof["isotonic"] = pred
        val_ece = {m: round(float(ece(p, y, self.bins)), 6) for m, p in oof.items()}
        self.method = min(sorted(val_ece), key=lambda m: val_ece[m])
        if "isotonic" in self.cfg["methods"]:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
            iso.fit(s, y)
            # rounded to 6 d.p. so the in-memory map equals the sealed calibration.json a later
            # run (R3 victim) loads back
            self._iso_x = np.round(np.asarray(iso.X_thresholds_, dtype=float), 6)
            self._iso_y = np.round(np.asarray(iso.y_thresholds_, dtype=float), 6)
        self.report = {"methods": sorted(oof), "val_ece_out_of_fold": val_ece, "selected": self.method,
                       "temperature": self.t, "bins": self.bins, "n_val_cal": int(len(s)),
                       "selection_rule": "lower out-of-fold ECE on val_cal (2-fold by index parity); ties -> alphabetical"}
        if self._iso_x is not None:
            self.report["isotonic"] = {"x": [float(v) for v in self._iso_x], "y": [float(v) for v in self._iso_y]}
        return self.report

    def load(self, report: dict) -> dict:
        """Restore a fitted map from a sealed calibration.json (the R3 victim is the R1 run's C4)."""
        self.method = report["selected"]
        self.t = float(report["temperature"])
        self.bins = int(report["bins"])
        if "isotonic" in report:
            self._iso_x = np.asarray(report["isotonic"]["x"], dtype=float)
            self._iso_y = np.asarray(report["isotonic"]["y"], dtype=float)
        self.report = dict(report)
        return self.report

    def apply(self, raw: float) -> dict:
        if self.method is None:
            raise RuntimeError("StubCalibration.apply before fit")
        if self.method == "temperature":
            p = float(_temperature(np.array([raw]), self.t)[0])
        else:
            p = float(np.interp(raw, self._iso_x, self._iso_y))
        return {"calibrated_probability": round(p, 6), "calibration_method": self.method}


# --- : fitted temperature and the out-of-fold choice ----------------------

PARAM_DECIMALS = 6
METHODS = ("isotonic", "temperature")


def fit_temperature(p_false, y, lo: float = 0.05, hi: float = 20.0, xatol: float = 1e-8) -> float:
    """T minimising the mean NLL of sigmoid(logit(p) / T) over [lo, hi], rounded to 6 d.p.

 netcal's TemperatureScaling stores the inverse of this T as `.temperature` (operational constraint)."""
    from scipy.optimize import minimize_scalar

    p = np.clip(np.asarray(p_false, dtype=float), 1e-6, 1 - 1e-6)
    yy = np.asarray(y, dtype=float)
    if p.ndim != 1 or p.shape != yy.shape or p.size == 0:
        raise ValueError("fit_temperature needs two non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(yy))):
        raise ValueError("fit_temperature: non-finite input")
    if not 0.0 < lo < hi:
        raise ValueError(f"fit_temperature: bounds must satisfy 0 < lo < hi, got {lo}, {hi}")
    z = np.log(p / (1 - p))

    def nll(t: float) -> float:
        q = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-12, 1 - 1e-12)
        return float(-np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q)))

    res = minimize_scalar(nll, bounds=(float(lo), float(hi)), method="bounded", options={"xatol": float(xatol)})
    return round(float(res.x), PARAM_DECIMALS)


def _fit_isotonic(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(s, y)
    return (np.round(np.asarray(iso.X_thresholds_, dtype=float), PARAM_DECIMALS),
            np.round(np.asarray(iso.y_thresholds_, dtype=float), PARAM_DECIMALS))


class FittedCalibration:
    """C4 for `c4.method: fitted` (pre-registered specification
 temperature and its own isotonic map on the other fold; the method with the lower out-of-fold
 ECE wins (ties alphabetical); both maps are then refitted on all of val_cal and reported."""

    name = "fitted-calibration-v1"

    def __init__(self, cfg: dict) -> None:
        methods = sorted(str(m) for m in cfg["methods"])
        if not methods or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
            raise ValueError(f"c4.methods must be a non-empty subset of {list(METHODS)}")
        if cfg["select_by"] != "oof_ece":
            raise ValueError("c4.select_by must be 'oof_ece'")
        if int(cfg["folds"]) != 2:
            raise ValueError("c4.folds must be 2 (index parity)")
        lo, hi = (float(v) for v in cfg["temperature_bounds"])
        if not 0.0 < lo < hi:
            raise ValueError("c4.temperature_bounds must satisfy 0 < lo < hi")
        self.methods = methods
        self.bins = int(cfg["bins"])
        self.fit_on = str(cfg["fit_on"])
        self.bounds = (lo, hi)
        self.xatol = float(cfg["xatol"])
        self.method: str | None = None
        self.t: float | None = None
        self._iso_x: np.ndarray | None = None
        self._iso_y: np.ndarray | None = None
        self.report: dict = {}

    def _maps(self, s: np.ndarray, y: np.ndarray) -> dict:
        out: dict = {}
        if "temperature" in self.methods:
            out["temperature"] = fit_temperature(s, y, *self.bounds, xatol=self.xatol)
        if "isotonic" in self.methods:
            out["isotonic"] = _fit_isotonic(s, y)
        return out

    @staticmethod
    def _apply_map(method: str, fitted, s: np.ndarray) -> np.ndarray:
        if method == "temperature":
            return _temperature(s, fitted)
        x, yk = fitted
        return np.interp(s, x, yk)

    def fit(self, raw_scores: list[float], y_false: list[int]) -> dict:
        s = np.asarray(raw_scores, dtype=float)
        y = np.asarray(y_false, dtype=float)
        if s.ndim != 1 or s.shape != y.shape or s.size == 0:
            raise ValueError("FittedCalibration.fit needs two non-empty 1-D arrays of equal length")
        if not (np.all(np.isfinite(s)) and np.all(np.isin(y, (0.0, 1.0)))):
            raise ValueError("FittedCalibration.fit: raw scores must be finite and labels 0/1")
        idx = np.arange(len(s))
        oof = {m: np.empty_like(s) for m in self.methods}
        fold_t: list[float | None] = []
        for fold in (0, 1):
            train, held = idx % 2 != fold, idx % 2 == fold
            if held.sum() == 0:
                fold_t.append(None)
                continue
            if train.sum() < 2:  # too few items to fit a map: the held fold keeps its raw scores
                for m in self.methods:
                    oof[m][held] = s[held]
                fold_t.append(None)
                continue
            maps = self._maps(s[train], y[train])
            fold_t.append(maps.get("temperature"))
            for m in self.methods:
                oof[m][held] = self._apply_map(m, maps[m], s[held])
        val_ece = {m: round(float(ece(p, y, self.bins)), 6) for m, p in oof.items()}
        selected = min(sorted(val_ece), key=lambda m: val_ece[m])
        full = self._maps(s, y)
        in_sample = {m: round(float(ece(self._apply_map(m, full[m], s), y, self.bins)), 6) for m in self.methods}
        report = {
            "calibration": self.name, "methods": list(self.methods), "val_ece_out_of_fold": val_ece,
            "ece_in_sample": in_sample, "selected": selected, "temperature": None,
            "inverse_temperature": None, "fold_temperatures": fold_t,
            "temperature_bounds": [self.bounds[0], self.bounds[1]], "xatol": self.xatol, "bins": self.bins,
            "folds": 2, "fit_on": self.fit_on, "n_val_cal": int(len(s)),
            "selection_rule": ("lower out-of-fold ECE on val_cal (2-fold by index parity, each fold refits "
                               "its own map); ties -> alphabetical; chosen map refitted on all of val_cal"),
        }
        if "temperature" in full:
            report["temperature"] = full["temperature"]
            report["inverse_temperature"] = round(1.0 / full["temperature"], PARAM_DECIMALS)
        if "isotonic" in full:
            x, yk = full["isotonic"]
            report["isotonic"] = {"x": [float(v) for v in x], "y": [float(v) for v in yk]}
        return self.load(report)

    def load(self, report: dict) -> dict:
        """Restore the fitted maps from a sealed calibration.json."""
        self.method = report["selected"]
        if self.method not in METHODS:
            raise ValueError(f"calibration report: unknown method {self.method!r}")
        self.t = None if report.get("temperature") is None else float(report["temperature"])
        if "isotonic" in report:
            self._iso_x = np.asarray(report["isotonic"]["x"], dtype=float)
            self._iso_y = np.asarray(report["isotonic"]["y"], dtype=float)
        if (self.method == "temperature" and self.t is None) or (self.method == "isotonic" and self._iso_x is None):
            raise ValueError("calibration report lacks the selected map")
        self.bins = int(report["bins"])
        self.report = dict(report)
        return self.report

    def apply(self, raw: float) -> dict:
        if self.method is None:
            raise RuntimeError("FittedCalibration.apply before fit or load")
        if self.method == "temperature":
            if self.t is None:
                raise RuntimeError("temperature calibration active but parameter t is None")
            p = float(_temperature(np.array([raw], dtype=float), self.t)[0])
        else:
            p = float(np.interp(raw, self._iso_x, self._iso_y))
        return {"calibrated_probability": round(p, 6), "calibration_method": self.method}
