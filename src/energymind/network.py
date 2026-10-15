"""
Modèle réseau pandapower — Microgrid Campus Universitaire (Bus BT 400 V).
Hypothèse documentée : topologie simplifiée validée manuellement (Phase 1).
Remplacer par le schéma unifilaire réel en Phase 2.

Topologie :
  Bus 0 (slack) — Point de Connexion Commun Réseau Public (PCC)
  Bus 1         — Tableau général bâtiment A (charge principale)
  Bus 2         — Sous-tableau PV + onduleur
  Bus 3         — Rack BESS

Lignes :
  L01 : PCC → Tableau principal  (câble 70 mm², 50 m)
  L12 : Tableau principal → PV   (câble 35 mm², 30 m)
  L13 : Tableau principal → BESS (câble 35 mm², 20 m)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import pandapower as pp
    _PP_AVAILABLE = True
except ImportError:
    _PP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Construction du réseau (une seule fois, réutilisé)
# ---------------------------------------------------------------------------

def _build_net():
    """Retourne le réseau pandapower Campus BT."""
    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)

    # Buses (kV = 0.4 → 400 V BT)
    pp.create_bus(net, vn_kv=0.4, name="PCC_STEG")      # 0 — slack
    pp.create_bus(net, vn_kv=0.4, name="Tableau_A")     # 1
    pp.create_bus(net, vn_kv=0.4, name="Bus_PV")        # 2
    pp.create_bus(net, vn_kv=0.4, name="Bus_BESS")      # 3

    # Source externe STEG (slack)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="STEG")

    # Lignes BT aluminium câblé (r_ohm_per_km, x_ohm_per_km calculés pour câble 400 V)
    pp.create_line_from_parameters(
        net, from_bus=0, to_bus=1, length_km=0.050,
        r_ohm_per_km=0.443, x_ohm_per_km=0.076,
        c_nf_per_km=0.0, max_i_ka=0.21, name="L01_PCC_TabA")
    pp.create_line_from_parameters(
        net, from_bus=1, to_bus=2, length_km=0.030,
        r_ohm_per_km=0.868, x_ohm_per_km=0.078,
        c_nf_per_km=0.0, max_i_ka=0.142, name="L12_TabA_PV")
    pp.create_line_from_parameters(
        net, from_bus=1, to_bus=3, length_km=0.020,
        r_ohm_per_km=0.868, x_ohm_per_km=0.078,
        c_nf_per_km=0.0, max_i_ka=0.142, name="L13_TabA_BESS")

    # Éléments de charge/génération (valeurs injectées à chaque pas)
    pp.create_load(net, bus=1, p_mw=0.0, q_mvar=0.0, name="Load_campus")
    pp.create_sgen(net, bus=2, p_mw=0.0, q_mvar=0.0, name="PV_inverter")
    pp.create_storage(net, bus=3, p_mw=0.0, max_e_mwh=0.1, q_mvar=0.0, name="BESS")

    return net


_NET = None

def _get_net():
    global _NET
    if _NET is None:
        _NET = _build_net()
    return _NET


# ---------------------------------------------------------------------------
# Calcul du flux de puissance sur un pas de temps
# ---------------------------------------------------------------------------

def run_powerflow(pv_kw: float, load_kw: float, bess_kw: float,
                  v_steg_pu: float = 1.0) -> dict:
    """
    Lance un load-flow pandapower pour un pas de temps.

    Paramètres
    ----------
    pv_kw    : puissance PV injectée (kW, ≥ 0)
    load_kw  : consommation campus (kW, ≥ 0)
    bess_kw  : puissance BESS (kW > 0 = décharge, < 0 = charge)
    v_steg_pu: tension de la source STEG (p.u., typiquement 0.97–1.03)

    Retour
    ------
    dict avec tensions, courants, pertes, résidus physiques.
    """
    if not _PP_AVAILABLE:
        return _fallback_powerflow(pv_kw, load_kw, bess_kw)

    net = _get_net()
    net.ext_grid.at[0, "vm_pu"] = v_steg_pu

    # Unités MW (pandapower)
    net.load.at[0, "p_mw"] = load_kw / 1000.0
    net.sgen.at[0, "p_mw"] = max(pv_kw, 0.0) / 1000.0
    # BESS : décharge = génération, charge = consommation
    if bess_kw >= 0:
        net.storage.at[0, "p_mw"] = bess_kw / 1000.0   # décharge → sgen equiv
        net.load.at[0, "p_mw"] += 0.0
    else:
        net.storage.at[0, "p_mw"] = bess_kw / 1000.0   # charge (négatif)

    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=False,
                 max_iteration=30, tolerance_mva=1e-6, verbose=False)
        ok = True
    except Exception:
        ok = False

    if ok and net.converged:
        v = net.res_bus.vm_pu.values * 400.0            # tensions en V
        p_loss_kw = net.res_line.pl_mw.sum() * 1000.0   # pandapower 3.x → MW, converti en kW
        grid_kw = net.res_ext_grid.p_mw.iloc[0] * 1000.0
        v_pcc = float(v[0])
        v_inv = float(v[2])
        r_balance = grid_kw + pv_kw + bess_kw - load_kw - p_loss_kw
        return {
            "converged": True,
            "v_pcc_V": v_pcc,
            "v_inv_V": v_inv,
            "v_bess_V": float(v[3]),
            "grid_kw": grid_kw,
            "loss_kw": p_loss_kw,
            "r_balance_kw": r_balance,            # doit être ≈ 0
            "v_residual": v_inv - v_pcc,          # chute de tension PCC→onduleur
        }
    else:
        return _fallback_powerflow(pv_kw, load_kw, bess_kw)


def _fallback_powerflow(pv_kw, load_kw, bess_kw):
    """Calcul analytique BT si pandapower non disponible (documenté)."""
    v_nom = 400.0
    loss = 0.01 * load_kw + 0.005 * pv_kw
    grid_kw = load_kw + loss - pv_kw - bess_kw
    v_drop = 0.4 * load_kw / 60.0                       # chute linéaire hypothèse
    v_pcc = v_nom * (1.0 - 0.012 * load_kw / load_kw if load_kw > 0 else 1.0)
    return {
        "converged": False,
        "v_pcc_V": v_pcc,
        "v_inv_V": v_pcc - v_drop,
        "v_bess_V": v_pcc - v_drop * 0.5,
        "grid_kw": grid_kw,
        "loss_kw": loss,
        "r_balance_kw": 0.0,
        "v_residual": -v_drop,
    }


# ---------------------------------------------------------------------------
# Calcul vectoriel sur un DataFrame complet (évite la boucle Python)
# ---------------------------------------------------------------------------

def powerflow_residuals(df: pd.DataFrame, sample_rate: int = 4) -> pd.DataFrame:
    """
    Calcule les résidus pandapower sur un DataFrame de simulation.

    Pour des raisons de performance, un pas sur `sample_rate` est utilisé.
    Les valeurs intermédiaires sont interpolées.

    Paramètres
    ----------
    df         : DataFrame avec colonnes pv_kw, load_kw, bess_kw, grid_kw
    sample_rate: 1 = tous les pas, 4 = 1 h (recommandé pour Phase 1)

    Retour
    ------
    DataFrame avec r_balance_kw, v_pcc_V, v_inv_V, loss_kw (index = df.index)
    """
    rows = []
    idx_sampled = df.index[::sample_rate]
    df_s = df.loc[idx_sampled]

    for _, row in df_s.iterrows():
        res = run_powerflow(
            pv_kw=float(row.pv_kw),
            load_kw=float(row.load_kw),
            bess_kw=float(row.bess_kw),
        )
        rows.append(res)

    out = pd.DataFrame(rows, index=idx_sampled)
    # Interpolation sur l'index complet
    out = out.reindex(df.index).interpolate(method="time")
    return out
