"""Project data directories."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_MODELS = DATA_DIR / "models"
DATA_MODELS_HOUR = DATA_MODELS / "hour"
DATA_MODELS_DAY = DATA_MODELS / "day"
DATA_LIVE = DATA_DIR / "live"


def ensure_data_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_MODELS.mkdir(parents=True, exist_ok=True)
    DATA_MODELS_HOUR.mkdir(parents=True, exist_ok=True)
    DATA_MODELS_DAY.mkdir(parents=True, exist_ok=True)
    DATA_LIVE.mkdir(parents=True, exist_ok=True)
    (DATA_LIVE / "frozen").mkdir(parents=True, exist_ok=True)
