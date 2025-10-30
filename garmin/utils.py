import json
import logging
from pathlib import Path

logging.getLogger("garminconnect").setLevel(logging.CRITICAL)

EXPORT_ROOT = Path("garmin_export").resolve()
EXPORT_ROOT.mkdir(exist_ok=True)


def save_json(data, path: Path):
    """Save data to JSON file (UTF-8, pretty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {path.relative_to(EXPORT_ROOT)}")

