"""
Evidence-Driven Energy Forensics engine.

Components:
  FeatureBuilder  — normalized physics residuals + LightGBM PV surrogate model
  Diagnoser       — physics-structured Naive Bayes with active evidence acquisition
  SHAPExplainer   — LightGBM feature attributions via SHAP TreeExplainer
  persistent()    — noise filter: alert only after k consecutive anomalous steps

LightGBM is preferred over HistGradientBoosting for tabular weather data because
it natively handles missing values, integrates with SHAP TreeExplainer, and trains
faster due to leaf-wise growth. Comparison against a linear baseline is documented
in evaluate.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .faults import CLASSES

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _LGB_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


FEATURES = ["bal", "pvexp", "inv", "vpcc", "vinv"]

EXPLAIN = {
    "bal":   "power balance residual (grid + PV + BESS − load − losses)",
    "pvexp": "gap between measured PV and LightGBM-predicted PV (weather → power)",
    "inv":   "inverter AC output vs. independent PV meter disagreement",
    "vpcc":  "PCC voltage deviation from pandapower expected value",
    "vinv":  "inverter terminal voltage deviation from expected value",
}

HIDDEN_COST = {"inv": 1.0, "vinv": 0.5}


class _LGBWrapper:
    """LightGBM regressor with HistGBM fallback when LightGBM is unavailable."""

    def __init__(self):
        self._model = None
        self._backend = "lgb" if _LGB_AVAILABLE else "hist"

    def fit(self, X, y):
        if _LGB_AVAILABLE:
            params = {
                "objective": "regression",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "feature_fraction": 0.8,
                "subsample": 0.8,
                "subsample_freq": 5,
                "random_state": 42,
                "verbose": -1,
            }
            self._model = lgb.LGBMRegressor(**params)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, random_state=42)
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    @property
    def booster(self):
        """Return the LightGBM booster for SHAP (None if fallback backend)."""
        if _LGB_AVAILABLE and hasattr(self._model, "booster_"):
            return self._model.booster_
        return None


class SHAPExplainer:
    """
    SHAP TreeExplainer for the LightGBM PV surrogate.

    Usage:
        shap_vals = explainer.explain(X)
        explainer.summary_text(X, feature_names)
    """

    def __init__(self, wrapper: _LGBWrapper):
        self._exp = None
        if _SHAP_AVAILABLE and wrapper.booster is not None:
            try:
                self._exp = shap.TreeExplainer(wrapper.booster)
            except Exception:
                self._exp = None

    def explain(self, X: np.ndarray) -> np.ndarray | None:
        """Return SHAP values (n_samples × n_features) or None."""
        if self._exp is None:
            return None
        return self._exp.shap_values(X)

    def top_features(self, X_row: np.ndarray, feature_names: list[str],
                     top_k: int = 3) -> list[tuple[str, float]]:
        """Return the top_k SHAP feature contributions for a single time step."""
        if self._exp is None:
            return []
        vals = self._exp.shap_values(X_row.reshape(1, -1))[0]
        ranked = sorted(zip(feature_names, vals), key=lambda t: -abs(t[1]))
        return ranked[:top_k]

    @property
    def available(self) -> bool:
        return self._exp is not None


class FeatureBuilder:
    """
    Builds normalized physics residuals (robust MAD z-scores).

    Residuals computed:
      bal    — power balance (energy conservation)
      pvexp  — LightGBM model residual (measured PV minus weather-predicted PV)
      inv    — inverter vs. PV meter consistency
      vpcc   — PCC voltage residual
      vinv   — inverter terminal voltage residual
    """

    def __init__(self, window: int = 4):
        self.window = window
        self._pv_model = _LGBWrapper()
        self.shap_explainer: SHAPExplainer | None = None
        self.sig: dict[str, float] = {}
        self._feature_names = ["irradiance", "t_cell", "wind", "hour"]

    @staticmethod
    def _x(obs: pd.DataFrame) -> np.ndarray:
        h = obs.index.hour + obs.index.minute / 60
        return np.c_[obs.irradiance, obs.t_cell, obs.wind, h]

    def fit_pv_model(self, obs_normal: pd.DataFrame) -> "FeatureBuilder":
        """Train LightGBM on the healthy period (irradiance > 50 W/m²)."""
        m = obs_normal.irradiance > 50
        X = self._x(obs_normal[m])
        y = obs_normal.pv_meas[m].values
        self._pv_model.fit(X, y)
        self.shap_explainer = SHAPExplainer(self._pv_model)
        return self

    def fit_scales(self, obs_normal: pd.DataFrame) -> "FeatureBuilder":
        """Calibrate robust scale estimates (MAD) on normal-operating data."""
        r = self.raw(obs_normal)
        for f in FEATURES:
            x = r[f].dropna()
            mad = float((x - x.median()).abs().median())
            self.sig[f] = max(1.4826 * mad, 1e-6)
        return self

    def raw(self, obs: pd.DataFrame) -> pd.DataFrame:
        r = pd.DataFrame(index=obs.index)
        loss = 0.01 * obs.load_meas + 0.005 * obs.pv_meas
        r["bal"] = obs.grid_meas + obs.pv_meas + obs.bess_meas - obs.load_meas - loss
        exp = np.clip(self._pv_model.predict(self._x(obs)), 0, None)
        r["pvexp"] = ((obs.pv_meas - exp) / (0.05 * exp + 0.5)).where(obs.irradiance > 100)
        r["inv"] = obs.inv_ac - obs.pv_meas
        vexp = 400.0 * (1 - 0.012 * obs.load_meas / 60.0)
        r["vpcc"] = obs.v_pcc - vexp
        r["vinv"] = obs.v_inv - vexp
        return r

    def z(self, obs: pd.DataFrame, hidden: tuple | list = ()) -> pd.DataFrame:
        r = self.raw(obs)
        for f in FEATURES:
            r[f] = r[f] / self.sig.get(f, 1.0)
        for h in hidden:
            r[h] = np.nan
        return r.rolling(self.window, min_periods=1).median()

    def shap_pv(self, obs_row: pd.Series) -> list[tuple[str, float]]:
        """Return SHAP contributions for a single time step."""
        if self.shap_explainer is None or not self.shap_explainer.available:
            return []
        X_row = self._x(obs_row.to_frame().T)[0]
        return self.shap_explainer.top_features(X_row, self._feature_names)


class Diagnoser:
    """
    Probabilistic multi-hypothesis diagnoser.

    Implements physics-structured Naive Bayes with marginalization over missing
    measurements (NaN values). Missing sensors are excluded from the likelihood
    product — conditional independence is assumed and documented as a Phase 1
    simplification.

    Phase 2 roadmap: replace with a calibrated discriminative model or a
    Bayesian network with an explicit dependency structure (see PROPOSAL.md §4.2).
    """

    def __init__(self, prior_normal: float = 0.9):
        self.prior_normal = prior_normal
        self.prior: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, Z: pd.DataFrame, labels: pd.Series) -> "Diagnoser":
        n_cls, n_feat = len(CLASSES), len(FEATURES)
        self.mu = np.zeros((n_cls, n_feat))
        self.sd = np.ones((n_cls, n_feat))
        for i, c in enumerate(CLASSES):
            sub = Z[labels == c]
            for j, f in enumerate(FEATURES):
                x = sub[f].dropna()
                if len(x) >= 5:
                    self.mu[i, j] = float(x.mean())
                    self.sd[i, j] = max(float(x.std()), 1.0)
        k = n_cls - 1
        self.prior = np.array([self.prior_normal] + [(1 - self.prior_normal) / k] * k)
        return self

    def fit_v2(self, Z: pd.DataFrame, labels: pd.Series,
               classes: list) -> "Diagnoser":
        """Train on an arbitrary class list (V2: 8 classes). Same interface as fit()."""
        self._classes_v2 = list(classes)
        n_cls, n_feat = len(classes), len(FEATURES)
        self.mu = np.zeros((n_cls, n_feat))
        self.sd = np.ones((n_cls, n_feat))
        for i, c in enumerate(classes):
            sub = Z[labels == c]
            for j, f in enumerate(FEATURES):
                x = sub[f].dropna()
                if len(x) >= 5:
                    self.mu[i, j] = float(x.mean())
                    self.sd[i, j] = max(float(x.std()), 1.0)
        k = n_cls - 1
        self.prior = np.array([self.prior_normal] + [(1 - self.prior_normal) / k] * k)
        return self

    def loglik(self, Z: pd.DataFrame) -> np.ndarray:
        X = Z[FEATURES].values
        n_cls = self.mu.shape[0]
        ll = np.zeros((len(X), n_cls))
        for i in range(n_cls):
            lp = (-0.5 * ((X - self.mu[i]) / self.sd[i]) ** 2
                  - np.log(self.sd[i]))
            ll[:, i] = np.nansum(lp, axis=1)
        return ll

    def posterior(self, Z: pd.DataFrame) -> np.ndarray:
        ll = self.loglik(Z) + np.log(self.prior)
        ll -= ll.max(axis=1, keepdims=True)
        p = np.exp(ll)
        return p / p.sum(axis=1, keepdims=True)

    def contributions(self, z_row: pd.Series, cls: str) -> dict[str, float]:
        """Log-likelihood contribution of each measurement towards hypothesis 'cls' vs. 'normal'."""
        i = CLASSES.index(cls)
        out = {}
        for j, f in enumerate(FEATURES):
            x = z_row[f]
            if np.isnan(x):
                continue
            def ll(k):
                return -0.5 * ((x - self.mu[k, j]) / self.sd[k, j]) ** 2 - np.log(self.sd[k, j])
            out[f] = float(ll(i) - ll(0))
        return out

    def next_best_check(self, z_row: pd.Series, p: np.ndarray,
                        hidden: list[str], n_draws: int = 200,
                        seed: int = 0) -> list[tuple[str, float]]:
        """
        Active Evidence Acquisition.

        Selects the hidden measurement that maximizes expected information gain
        minus its verification cost.

        Criterion: U(m) = H(H|E) − E[H(H|E,y_m)] − λ·C(m)
        """
        rng = np.random.default_rng(seed)
        h0 = -np.sum(p * np.log(p + 1e-12))
        best = []
        for f in hidden:
            j = FEATURES.index(f)
            eg = 0.0
            for i, pc in enumerate(p):
                if pc < 1e-3:
                    continue
                xs = rng.normal(self.mu[i, j], self.sd[i, j], n_draws)
                lik = (np.exp(-0.5 * ((xs[:, None] - self.mu[:, j]) / self.sd[:, j]) ** 2)
                       / self.sd[:, j])
                q = lik * p
                q /= q.sum(axis=1, keepdims=True)
                eg += pc * np.mean(-np.sum(q * np.log(q + 1e-12), axis=1))
            gain = h0 - eg - 0.05 * HIDDEN_COST.get(f, 1.0)
            best.append((f, gain))
        best.sort(key=lambda t: -t[1])
        return best


def persistent(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Alert only when an anomaly persists for k consecutive steps.

    At 15-minute resolution, k=3 corresponds to a 45-minute confirmation
    window, reducing transient false positives.
    """
    s = pd.Series(mask.astype(int)).rolling(k, min_periods=k).sum().fillna(0).values
    return s >= k
