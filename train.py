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
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "parking-space-4"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "best.pt"
WORKSPACE = "muhammad-syihab-bdynf"
PROJECT = "parking-space-ipm1b"
VERSION = 4


def prepare_dataset():
    """Reuse or download the original public Roboflow dataset, version 4."""
    source_yaml = DATA_DIR / "data.yaml"
    if not source_yaml.is_file():
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        if DATA_DIR.exists():
            if any(DATA_DIR.iterdir()):
                raise RuntimeError(
                    f"Incomplete dataset at {DATA_DIR}; inspect it before retrying")
            DATA_DIR.rmdir()

        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR))

    if not source_yaml.is_file():
        raise FileNotFoundError(f"Missing Roboflow data.yaml: {source_yaml}")

    original = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    config = DATA_DIR / "training_config.yaml"
    config.write_text(
        yaml.safe_dump({
            "path": str(DATA_DIR.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": original["names"],}),
        encoding="utf-8",)
    return config


def train_and_save(data_yaml):
    """Train from COCO-pretrained YOLO11n and copy the best checkpoint."""
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}", flush=True)

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

    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"Training did not create best.pt: {best}")

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, WEIGHTS)
    print(f"Saved trained model: {WEIGHTS}", flush=True)


def run_script(script, *args):
    """Execute another project script and stop if it fails."""
    print(f"Running {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True,)


def verify_saved_results():
    """Check the main outputs needed by the dashboard and report."""
    required = [
        WEIGHTS,
        RESULTS / "parking_costs_valid.json",
        RESULTS / "parking_cost_curve_valid.png",]

    for split in ("valid", "test"):
        required += [
            RESULTS / f"parking_predictions_{split}.json",
            RESULTS / f"parking_metrics_{split}.json",]

    missing = [
        str(p.relative_to(ROOT))
        for p in required
        if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing pipeline outputs: " + ", ".join(missing))

    for split in ("valid", "test"):
        summary = json.loads((RESULTS / f"parking_metrics_{split}.json").read_text())
        predictions = json.loads((RESULTS / f"parking_predictions_{split}.json").read_text())

        if (
            summary.get("split") != split
            or summary.get("images_evaluated") != len(predictions)
            or not {"trained", "coco", "none"}.issubset(
                summary.get("results", {}))):
            raise ValueError(f"Incomplete or inconsistent {split} evaluation")

    print("Validation/test metrics, baselines and dashboard inputs verified.", flush=True,)


def save_test_costs():
    """Evaluate test using validation thresholds without re-optimizing."""
    from cost_analysis import (
        build_cases,
        hybrid_decisions,
        metrics,
        model_only_decisions,
        unmatched_above,)

    validation_costs = json.loads((RESULTS / "parking_costs_valid.json").read_text())
    test_samples = json.loads((RESULTS / "parking_predictions_test.json").read_text())

    truth, pred_class, confidence, extras = build_cases(test_samples)

    hybrid = validation_costs["hybrid_optimum"]
    model = validation_costs["model_only_optimum"]
    empty_t = float(hybrid["empty_conf_threshold"])
    occupied_t = float(hybrid["occupied_conf_threshold"])
    model_t = float(model["confidence_threshold_empty"])

    report = {
        "split": "test",
        "thresholds_selected_on": "validation (fixed before test evaluation)",
        "annotated_spaces": int(len(truth)),
        "thresholds": {
            "model_only_empty": model_t,
            "hybrid_empty": empty_t,
            "hybrid_occupied": occupied_t,},
        "model_only": metrics(truth, model_only_decisions(pred_class, confidence, model_t),),
        "hybrid": metrics(truth, hybrid_decisions(pred_class, confidence, empty_t, occupied_t),),
        "extra_detections_not_costed": unmatched_above(
            extras, empty_t, occupied_t),
        "limitations": ("Offline matching uses annotated parking-space locations. "
        "Human review is simulated, and unmatched detections are not costed."),}

    path = RESULTS / "parking_costs_test.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Test costs per 1,000: "
        f"model €{report['model_only']['cost_per_1000_eur']:.2f}, "
        f"hybrid €{report['hybrid']['cost_per_1000_eur']:.2f}",
        flush=True,)
    print(f"Saved {path.relative_to(ROOT)}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Use saved results; do not train or run YOLO",)
    args = parser.parse_args()

    if not args.cached_only:
        data_yaml = prepare_dataset()
        train_and_save(data_yaml)
        device = "0" if torch.cuda.is_available() else "cpu"
        run_script("evaluate_parking.py", "--split", "valid", "--device", device)
        run_script("cost_analysis.py")
        run_script("evaluate_parking.py", "--split", "test", "--device", device)

    verify_saved_results()
    save_test_costs()

    print("Pipeline outputs ready for Gradio and REPORT.md.", flush=True)

if __name__ == "__main__":
    main()
