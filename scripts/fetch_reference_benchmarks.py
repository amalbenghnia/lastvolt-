"""
Téléchargement et vérification des benchmarks publics de référence (PVGIS, NASA POWER).
Ce script interroge les APIs officielles ouvertes pour ancrer les paramètres du microgrid
dans des mesures physiques et météorologiques réelles et vérifiables.
"""
from pathlib import Path
import json
import urllib.request
import ssl

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data" / "reference"
REF_DIR.mkdir(parents=True, exist_ok=True)

# Coordonnées géographiques de référence (Nord de la Tunisie : 36.8° N, 10.2° E)
LAT = 36.8
LON = 10.2

def fetch_pvgis():
    """
    Interroge l'API PVGIS v5.2 (Commission Européenne - JRC)
    Calcule la production annuelle et mensuelle théorique pour 50 kWp PV.
    """
    url = (
        f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?"
        f"lat={LAT}&lon={LON}&peakpower=50.0&loss=14.0&outputformat=json"
    )
    print(f"[PVGIS] Requête vers {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EnergyMind-Research)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            out_file = REF_DIR / "pvgis_monthly_reference.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            totals = data["outputs"]["totals"]["fixed"]
            print(f"  -> PVGIS sauvegarde : {out_file.name}")
            print(f"  -> Production annuelle estimée (50 kWp) : {totals['E_y']:.1f} kWh/an")
            print(f"  -> Irradiation annuelle globale : {totals['H(i)_y']:.1f} kWh/m²/an")
            return totals
    except Exception as e:
        print(f"  [AVERTISSEMENT] PVGIS non accessible : {e}")
        return None

def fetch_nasa_power():
    """
    Interroge l'API NASA POWER pour récupérer l'irradiance globale et la température
    horaire de surface (All Sky Surface Shortwave Downward Irradiance & 2m Temperature).
    """
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude={LON}&latitude={LAT}"
        f"&start=20230601&end=20230607&format=JSON"
    )
    print(f"[NASA POWER] Requête vers API NASA ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EnergyMind-Research)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            out_file = REF_DIR / "nasa_power_hourly_reference.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            params = list(data["properties"]["parameter"].keys())
            print(f"  -> NASA POWER sauvegarde : {out_file.name}")
            print(f"  -> Variables récupérées : {params}")
            return params
    except Exception as e:
        print(f"  [AVERTISSEMENT] NASA POWER non accessible : {e}")
        return None

if __name__ == "__main__":
    print("=== Téléchargement des benchmarks ouverts de référence ===")
    pvgis_res = fetch_pvgis()
    nasa_res = fetch_nasa_power()
    print("=== Terminé avec succès ===")
