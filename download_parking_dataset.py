
from pathlib import Path
from roboflow import Roboflow
from api_key import load_api_key

# Dataset público de Roboflow
WORKSPACE = "muhammad-syihab-bdynf"
PROJECT = "parking-space-ipm1b"
VERSION = 4

# Carpeta donde guardaremos los datos
DATA_DIR = Path("data/parking-space-4").resolve()

def main():
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset ya descargado: {DATA_DIR}")
        return

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=load_api_key())

    dataset = (
        rf.workspace(WORKSPACE)
        .project(PROJECT)
        .version(VERSION)
        .download("yolov11", location=str(DATA_DIR))
    )

    print(f"Dataset descargado en: {dataset.location}")

if __name__ == "__main__":
    main()
