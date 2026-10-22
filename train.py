"""
Assignment 2: training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file. main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via uv run python main.py train, this file must, start to finish, with no
manual steps in between:

- Get your Roboflow API key via from api_key import load_api_key, so that it
  works whether the key is in your .env file or in the environment, and
  download your dataset version from Roboflow Universe.
- Finetune yolo11n/yolo11s (or the -cls variant) on that dataset, from the
  pretrained checkpoint.
- Evaluate on the test split against the two baselines named in the brief: the
  pretrained COCO model zero-shot, and a trivial predict-nothing / majority
  class baseline.
- Sweep the confidence threshold, compute the expected cost at each point from
  the cost matrix you defined in REPORT.md, and save the sweep.
- Save everything app.py needs to build its dashboard without retraining:
  the metrics, the per-class numbers, the cost sweep, and the chosen operating
  point.

This is your real training pipeline. It takes as long as it takes, and it runs
wherever training is practical for you: your own machine, a GPU you have access
to, or a hosted notebook. Nothing re-runs it to check your submission.

The weights it produces are what you commit to weights/ and what your app
serves. Say in REPORT.md where you trained and what settings you used.

Save the trained model into weights/. That is the file you commit and the file
app.py loads.

How you structure the code beyond that is your call to make and be able to
defend: file layout, function boundaries, naming.
"""
from pathlib import Path
import shutil
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "parking-space-4"
WEIGHTS_DIR = ROOT / "weights"
WORKSPACE = "muhammad-syihab-bdynf"
PROJECT = "parking-space-ipm1b"
VERSION = 4

def prepare_dataset():
    source_yaml = DATA_DIR / "data.yaml"

    if not source_yaml.exists():
        print("Downloading Roboflow dataset...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(
            VERSION).download("yolov11", location=str(DATA_DIR))

    if not source_yaml.exists():
        raise FileNotFoundError("Dataset download failed.")

    original = yaml.safe_load(source_yaml.read_text())

    config = {
        "path": str(DATA_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": original["names"],}

    runtime_yaml = DATA_DIR / "training_config.yaml"
    runtime_yaml.write_text(yaml.safe_dump(config))

    return runtime_yaml

def main():
    data_yaml = prepare_dataset()
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Training device: {device}")
    print(f"Dataset: {data_yaml}")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=16,
        patience=8,
        device=device,
        workers=2,
        seed=42,
        project=str(ROOT / "runs" / "parking"),
        name="yolo11n",
        exist_ok=True,
        plots=True,)

    WEIGHTS_DIR.mkdir(exist_ok=True)

    best = Path(model.trainer.best)
    destination = WEIGHTS_DIR / "best.pt"

    if not best.exists():
        raise FileNotFoundError(f"Missing weights: {best}")

    shutil.copy2(best, destination)
    print(f"Best model saved: {destination}")

if __name__ == "__main__":
    main()
