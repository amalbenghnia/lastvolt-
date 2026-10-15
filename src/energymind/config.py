from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None) -> dict:
    with open(path or ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)

def val(node):
    """Valeur d'un parametre ({value,status} ou scalaire)."""
    return node["value"] if isinstance(node, dict) and "value" in node else node
