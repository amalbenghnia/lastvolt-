"""
Détection de dérives temporelles — CUSUM (Page-Hinkley) adapté aux résidus normalisés.

Usage :
    from energymind.drift import CUSUMDetector
    cusum = CUSUMDetector(k=0.5, h=5.0)
    cusum.update(residual_z)   # retourne True si dérive détectée
    cusum.reset()

Référence : Page (1954), Hinkley (1971).
Paramètre k (sensibilité) et h (seuil d'alarme) calibrés sur période normale.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class CUSUMDetector:
    """
    CUSUM bidirectionnel sur un résidu normalisé.

    Paramètres
    ----------
    k : sensibilité (slack) — typiquement 0.5 × ampleur minimale à détecter
    h : seuil d'alarme — contrôle le taux de fausse alarme
    """
    k: float = 0.5
    h: float = 5.0
    S_pos: float = field(default=0.0, init=False)
    S_neg: float = field(default=0.0, init=False)
    _alarm: bool = field(default=False, init=False)

    def update(self, z: float) -> bool:
        """Ingère un résidu z, retourne True si une dérive est détectée."""
        self.S_pos = max(0.0, self.S_pos + z - self.k)
        self.S_neg = max(0.0, self.S_neg - z - self.k)
        self._alarm = (self.S_pos >= self.h) or (self.S_neg >= self.h)
        return self._alarm

    def reset(self):
        """Réinitialise après validation humaine ou correction."""
        self.S_pos = 0.0
        self.S_neg = 0.0
        self._alarm = False

    @property
    def alarm(self) -> bool:
        return self._alarm

    @property
    def state(self) -> dict:
        return {"S_pos": round(self.S_pos, 3), "S_neg": round(self.S_neg, 3),
                "alarm": self._alarm}


class MultiCUSUM:
    """
    Batterie de CUSUM sur plusieurs canaux de résidus.
    Chaque canal surveille une caractéristique physique différente.

    Paramètres
    ----------
    channels : liste des noms de résidus à surveiller
    k, h     : paramètres communs (à calibrer par canal en Phase 2)
    """
    def __init__(self, channels: list[str], k: float = 0.5, h: float = 5.0):
        self.channels = channels
        self.detectors = {c: CUSUMDetector(k=k, h=h) for c in channels}

    def update_row(self, z_row: dict | pd.Series) -> dict:
        """Met à jour tous les canaux avec une ligne de résidus normalisés."""
        return {c: self.detectors[c].update(float(z_row.get(c, 0.0) or 0.0))
                for c in self.channels}

    def any_alarm(self) -> bool:
        return any(d.alarm for d in self.detectors.values())

    def alarm_channels(self) -> list[str]:
        return [c for c, d in self.detectors.items() if d.alarm]

    def reset_all(self):
        for d in self.detectors.values():
            d.reset()

    def state(self) -> dict:
        return {c: d.state for c, d in self.detectors.items()}


def cusum_scan(Z: pd.DataFrame, channels: list[str],
               k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """
    Applique MultiCUSUM sur un DataFrame de résidus normalisés.

    Retour
    ------
    DataFrame booléen : True = alarme CUSUM sur ce canal à ce pas de temps.
    """
    mc = MultiCUSUM(channels, k=k, h=h)
    rows = []
    for _, row in Z[channels].iterrows():
        rows.append(mc.update_row(row))
    out = pd.DataFrame(rows, index=Z.index, columns=channels)
    return out


def calibrate_cusum(Z_normal: pd.DataFrame, channels: list[str],
                    target_fpr: float = 0.01) -> dict:
    """
    Calibre h pour atteindre un taux de fausse alarme cible.

    Stratégie : grid search sur h ∈ [1, 20], k = 0.5 fixé.
    Retourne {'k': k, 'h': h} avec le h optimal.
    """
    best_h, best_fpr = 5.0, 1.0
    for h_trial in np.arange(1.0, 20.5, 0.5):
        alarms = cusum_scan(Z_normal, channels, k=0.5, h=h_trial)
        fpr = float(alarms.any(axis=1).mean())
        if fpr <= target_fpr:
            best_h = h_trial
            best_fpr = fpr
            break
    return {"k": 0.5, "h": best_h, "achieved_fpr": best_fpr}
