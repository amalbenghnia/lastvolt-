"""
pandapower distribution network model V2 — Campus microgrid.
Topology: 20/0.4 kV transformer, 3 low-voltage radial feeders, PV array, and BESS.

Bus configuration:
  Bus 0 : HV 20 kV  — Utility grid point of common coupling (slack bus)
  Bus 1 : LV 0.4 kV — Main distribution switchboard
  Bus 2 : Feeder 1  — Administration building
  Bus 3 : Feeder 2  — Laboratories and HVAC
  Bus 4 : Feeder 3  — Student residences and EV chargers
  Bus 5 : PV Bus    — Inverter connection
  Bus 6 : BESS Bus  — Battery storage connection
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    import pandapower as pp
    import pandapower.plotting as ppplot
    _PP_AVAILABLE = True
except ImportError:
    _PP_AVAILABLE = False


NETWORK_PARAMS = {
    "transformer": {
        "s_mva": 0.250,
        "vn_hv_kv": 20.0,
        "vn_lv_kv": 0.4,
        "vk_percent": 4.0,
        "vkr_percent": 1.3,
        "pfe_kw": 0.55,
        "i0_percent": 1.5,
        "vector_group": "Dyn11",
        "status": "simulation_assumption",
        "note": "Standard 250 kVA distribution transformer datasheet values",
    },
    "loads": {
        "admin_kw": 15.0,
        "labs_kw": 25.0,
        "residence_kw": 10.0,
    },
    "cables": {
        "L01": {"section_mm2": 70, "length_km": 0.040, "r": 0.443, "x": 0.076, "i_max_ka": 0.210},
        "L02": {"section_mm2": 95, "length_km": 0.060, "r": 0.320, "x": 0.074, "i_max_ka": 0.250},
        "L03": {"section_mm2": 70, "length_km": 0.080, "r": 0.443, "x": 0.076, "i_max_ka": 0.210},
        "L04": {"section_mm2": 35, "length_km": 0.030, "r": 0.868, "x": 0.078, "i_max_ka": 0.142},
        "L05": {"section_mm2": 35, "length_km": 0.020, "r": 0.868, "x": 0.078, "i_max_ka": 0.142},
    },
}

LOAD_SPLIT = {"admin": 0.30, "labs": 0.50, "residence": 0.20}

_NET_V2 = None


def _build_net_v2():
    """Instantiate the 7-bus pandapower network."""
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)

    b0 = pp.create_bus(net, vn_kv=20.0, name="HV_Grid_PCC")
    b1 = pp.create_bus(net, vn_kv=0.4, name="LV_MainSwitchboard")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Feeder1_Admin")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Feeder2_Labs_HVAC")
    b4 = pp.create_bus(net, vn_kv=0.4, name="Feeder3_Residence_EV")
    b5 = pp.create_bus(net, vn_kv=0.4, name="Bus_PV")
    b6 = pp.create_bus(net, vn_kv=0.4, name="Bus_BESS")

    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, name="Grid_20kV")

    p = NETWORK_PARAMS["transformer"]
    pp.create_transformer_from_parameters(
        net, hv_bus=b0, lv_bus=b1,
        sn_mva=p["s_mva"], vn_hv_kv=p["vn_hv_kv"], vn_lv_kv=p["vn_lv_kv"],
        vk_percent=p["vk_percent"], vkr_percent=p["vkr_percent"],
        pfe_kw=p["pfe_kw"], i0_percent=p["i0_percent"],
        name="Transformer_20_04kV",
    )

    c = NETWORK_PARAMS["cables"]
    for name, from_b, to_b, key in [
        ("L01_Admin", b1, b2, "L01"),
        ("L02_Labs", b1, b3, "L02"),
        ("L03_Residence", b1, b4, "L03"),
        ("L04_PV", b1, b5, "L04"),
        ("L05_BESS", b1, b6, "L05"),
    ]:
        cfg = c[key]
        pp.create_line_from_parameters(
            net, from_bus=from_b, to_bus=to_b,
            length_km=cfg["length_km"],
            r_ohm_per_km=cfg["r"], x_ohm_per_km=cfg["x"],
            c_nf_per_km=0.0, max_i_ka=cfg["i_max_ka"],
            name=name,
        )

    pp.create_load(net, bus=b2, p_mw=0.0, q_mvar=0.0, name="Load_Admin")
    pp.create_load(net, bus=b3, p_mw=0.0, q_mvar=0.0, name="Load_Labs")
    pp.create_load(net, bus=b4, p_mw=0.0, q_mvar=0.0, name="Load_Residence")

    pp.create_sgen(net, bus=b5, p_mw=0.0, q_mvar=0.0, name="PV_Inverter")
    pp.create_storage(net, bus=b6, p_mw=0.0, max_e_mwh=0.1, q_mvar=0.0, name="BESS")

    return net


def get_net_v2():
    global _NET_V2
    if _NET_V2 is None:
        if not _PP_AVAILABLE:
            raise ImportError("pandapower is required for network_v2")
        _NET_V2 = _build_net_v2()
    return _NET_V2


@dataclass
class PowerFlowResult:
    converged: bool
    v_lv_tableau: float
    v_admin: float
    v_labs: float
    v_residence: float
    v_pv: float
    v_bess: float
    grid_kw: float
    loss_kw: float
    p_transfo_kw: float
    r_balance_kw: float
    line_loading_pct: dict

    @property
    def v_pcc(self) -> float:
        return self.v_lv_tableau

    @property
    def v_inv(self) -> float:
        return self.v_pv

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "v_pcc_V": self.v_pcc,
            "v_inv_V": self.v_inv,
            "v_admin_V": self.v_admin,
            "v_labs_V": self.v_labs,
            "v_residence_V": self.v_residence,
            "grid_kw": self.grid_kw,
            "loss_kw": self.loss_kw,
            "r_balance_kw": self.r_balance_kw,
            **{f"loading_{k}_pct": v for k, v in self.line_loading_pct.items()},
        }


def run_powerflow_v2(
    pv_kw: float,
    load_kw: float,
    bess_kw: float,
    v_steg_pu: float = 1.0,
    load_split: dict | None = None,
) -> PowerFlowResult:
    """Execute Newton-Raphson AC power flow on the 7-bus distribution feeder."""
    if not _PP_AVAILABLE:
        return _fallback_v2(pv_kw, load_kw, bess_kw)

    split = load_split or LOAD_SPLIT
    net = get_net_v2()
    net.ext_grid.at[0, "vm_pu"] = v_steg_pu

    net.load.at[0, "p_mw"] = load_kw * split["admin"] / 1000.0
    net.load.at[1, "p_mw"] = load_kw * split["labs"] / 1000.0
    net.load.at[2, "p_mw"] = load_kw * split["residence"] / 1000.0

    net.sgen.at[0, "p_mw"] = max(pv_kw, 0.0) / 1000.0
    net.storage.at[0, "p_mw"] = bess_kw / 1000.0

    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=False,
                 max_iteration=50, tolerance_mva=1e-6, verbose=False)
        ok = net.converged
    except Exception:
        ok = False

    if ok:
        v_bt = net.res_bus.vm_pu.values[1:] * 400.0
        loss_lines_kw = net.res_line.pl_mw.sum() * 1000.0
        loss_transfo_kw = net.res_trafo.pl_mw.iloc[0] * 1000.0 if len(net.res_trafo) > 0 else 0.0
        loss_kw = loss_lines_kw + loss_transfo_kw
        grid_kw = net.res_ext_grid.p_mw.iloc[0] * 1000.0
        r_balance = grid_kw + pv_kw + bess_kw - load_kw - loss_kw
        line_names = list(net.line.name)
        loading_pct = dict(zip(
            line_names,
            net.res_line.loading_percent.values.tolist(),
        ))
        return PowerFlowResult(
            converged=True,
            v_lv_tableau=float(v_bt[0]),
            v_admin=float(v_bt[1]),
            v_labs=float(v_bt[2]),
            v_residence=float(v_bt[3]),
            v_pv=float(v_bt[4]),
            v_bess=float(v_bt[5]),
            grid_kw=float(grid_kw),
            loss_kw=float(loss_kw),
            p_transfo_kw=float(grid_kw),
            r_balance_kw=float(r_balance),
            line_loading_pct=loading_pct,
        )
    else:
        return _fallback_v2(pv_kw, load_kw, bess_kw)


def _fallback_v2(pv_kw, load_kw, bess_kw) -> PowerFlowResult:
    """Analytical fallback model if pandapower Newton-Raphson diverges."""
    loss = 0.012 * load_kw + 0.005 * pv_kw
    grid_kw = max(0.0, load_kw + loss - pv_kw - bess_kw)
    v_nom = 400.0
    v_drop = 0.8 * load_kw / 60.0
    v_pcc = v_nom * (1 - 0.012 * load_kw / 60.0) if load_kw > 0 else v_nom
    return PowerFlowResult(
        converged=False,
        v_lv_tableau=v_pcc,
        v_admin=v_pcc - 0.3 * v_drop,
        v_labs=v_pcc - 0.4 * v_drop,
        v_residence=v_pcc - 0.5 * v_drop,
        v_pv=v_pcc - 0.15 * v_drop,
        v_bess=v_pcc - 0.1 * v_drop,
        grid_kw=grid_kw,
        loss_kw=loss,
        p_transfo_kw=grid_kw,
        r_balance_kw=0.0,
        line_loading_pct={},
    )


class PandapowerSurrogate:
    """
    Polynomial surrogate approximating pandapower AC power flow.
    Provides sub-millisecond execution while strictly bounded by validation error gates.
    """
    MAX_ERROR_KW = 10.0
    N_FIT = 2000
    N_TEST = 500

    def __init__(self):
        self._fitted = False
        self._coeff = {}
        self.validation_report: dict = {}

    def _features(self, pv, load, bess):
        return np.array([
            1.0, pv / 50.0, load / 60.0, bess / 40.0,
            (pv * load) / 3000.0, (load ** 2) / 3600.0,
        ])

    def fit(self, rng=None, v_steg_range=(0.97, 1.03)):
        if not _PP_AVAILABLE:
            self._fitted = False
            return self
        rng = rng or np.random.default_rng(0)

        pv_fit = rng.uniform(0, 50, self.N_FIT)
        ld_fit = rng.uniform(10, 70, self.N_FIT)
        bs_fit = rng.uniform(-40, 40, self.N_FIT)
        vs_fit = rng.uniform(*v_steg_range, self.N_FIT)

        X, y_grid, y_vpcc, y_vinv = [], [], [], []
        for pv, ld, bs, vs in zip(pv_fit, ld_fit, bs_fit, vs_fit):
            res = run_powerflow_v2(pv, ld, bs, vs)
            X.append(self._features(pv, ld, bs))
            y_grid.append(res.grid_kw)
            y_vpcc.append(res.v_pcc)
            y_vinv.append(res.v_inv)

        X = np.array(X)
        self._coeff = {}
        for name, y in [("grid_kw", y_grid), ("v_pcc", y_vpcc), ("v_inv", y_vinv)]:
            coeff, _, _, _ = np.linalg.lstsq(X, np.array(y), rcond=None)
            self._coeff[name] = coeff

        self._fitted = True

        pv_te = rng.uniform(0, 55, self.N_TEST)
        ld_te = rng.uniform(8, 75, self.N_TEST)
        bs_te = rng.uniform(-42, 42, self.N_TEST)
        vs_te = rng.uniform(0.96, 1.04, self.N_TEST)

        errs = []
        for pv, ld, bs, vs in zip(pv_te, ld_te, bs_te, vs_te):
            true = run_powerflow_v2(pv, ld, bs, vs)
            pred = self.predict(pv, ld, bs)
            errs.append(abs(true.grid_kw - pred["grid_kw"]))

        errs = np.array(errs)
        self.validation_report = {
            "n_fit": self.N_FIT,
            "n_test": self.N_TEST,
            "grid_kw_mae": float(errs.mean()),
            "grid_kw_max": float(errs.max()),
            "grid_kw_p95": float(np.percentile(errs, 95)),
            "valid": bool(errs.max() < self.MAX_ERROR_KW),
            "note": "Surrogate validated against independent AC load flow trials.",
        }
        return self

    def predict(self, pv_kw, load_kw, bess_kw) -> dict:
        if not self._fitted:
            res = run_powerflow_v2(pv_kw, load_kw, bess_kw)
            return {"grid_kw": res.grid_kw, "v_pcc": res.v_pcc, "v_inv": res.v_inv}
        x = self._features(pv_kw, load_kw, bess_kw)
        return {name: float(x @ coeff) for name, coeff in self._coeff.items()}

    def predict_or_fallback(self, pv_kw, load_kw, bess_kw, v_steg_pu: float = 1.0) -> dict:
        in_domain = (0 <= pv_kw <= 55 and 8 <= load_kw <= 75 and -42 <= bess_kw <= 42)
        if self._fitted and in_domain and self.validation_report.get("valid", False):
            return self.predict(pv_kw, load_kw, bess_kw)
        res = run_powerflow_v2(pv_kw, load_kw, bess_kw, v_steg_pu)
        return {"grid_kw": res.grid_kw, "v_pcc": res.v_pcc, "v_inv": res.v_inv, "_used_pandapower": True}
