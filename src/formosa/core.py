from pathlib import Path
from platformdirs import user_data_dir

APP = "Formosa"
ROOT = Path(user_data_dir(APP))
DATA_DIR = ROOT / "data"

if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def set_data_dir(path: str | Path) -> None:
    global DATA_DIR
    DATA_DIR = Path(path)
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)