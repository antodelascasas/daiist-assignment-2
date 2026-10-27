"""DAIIST Assignment 2 — Parking operations dashboard.

Loads committed YOLO11 weights and cached validation predictions; never trains.
Visualisation filters are separate from the offline business cost policy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from cost_analysis import GRID, SOURCE, build_cases, hybrid_decisions, metrics, unmatched_above

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "best.pt"


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Generate the cached evaluation first.")
    return json.loads(path.read_text(encoding="utf-8"))


SAMPLES = read_json(SOURCE)
TEST = read_json(RESULTS / "parking_metrics_test.json")
COSTS = read_json(RESULTS / "parking_costs_valid.json")
TRUTH, PRED_CLASS, CONF, EXTRAS = build_cases(SAMPLES)
if not WEIGHTS.is_file():
    raise FileNotFoundError(f"Missing trained weights: {WEIGHTS}")
MODEL = YOLO(str(WEIGHTS))

SELECTED_EMPTY = float(COSTS["hybrid_optimum"]["empty_conf_threshold"])
SELECTED_OCCUPIED = float(COSTS["hybrid_optimum"]["occupied_conf_threshold"])
MODEL_ONLY = float(COSTS["model_only_optimum"]["confidence_threshold_empty"])

BASELINES_DF = pd.DataFrame([
    {"Model": label,
     "mAP50 (%)": round(100 * TEST["results"][key]["map50"], 2),
     "mAP50–95 (%)": round(100 * TEST["results"][key]["map50_95"], 2)}
    for key, label in (("trained", "Fine-tuned YOLO11n"),
                       ("coco", "COCO zero-shot (vehicle proxy)"),
                       ("none", "Predict nothing"))
])
CLASS_DF = pd.DataFrame([
    {"Class": name,
     "Precision @0.25 (%)": round(100 * item["precision_at_0.25_iou50"], 2),
     "Recall @0.25 (%)": round(100 * item["recall_at_0.25_iou50"], 2),
     "AP50 (%)": round(100 * item["ap50"], 2),
     "mAP50–95 (%)": round(100 * item["map50_95"], 2),
     "Ground-truth objects": item["ground_truth_instances"]}
    for name, item in TEST["results"]["trained"]["per_class"].items()
])

CSS = """
.gradio-container { max-width: 1320px !important; margin: auto !important; }
#hero { padding: 12px 0 4px; }
#hero h1 { font-size: 30px !important; margin: 0 0 6px !important; }
#hero p { color: var(--body-text-color-subdued); margin: 0 !important; }
#statusline { margin: 12px 0 0; }
#imagePanel img { object-fit: contain !important; }
#smallhelp {font-size: 12px; color: var(--body-text-color-subdued);}
"""

COLORS = {
    "PASS": (27, 148, 91),
    "REVIEW": (212, 143, 37),
    "FAIL": (218, 66, 72),
}


def font_for(image_width: int):
    px = max(13, min(24, int(image_width / 50)))
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", px)
    except OSError:
        return ImageFont.load_default()


def update_analysis(empty_t: float, occupied_t: float):
    decisions = hybrid_decisions(PRED_CLASS, CONF, float(empty_t), float(occupied_t))
    values = metrics(TRUTH, decisions)
    extra = unmatched_above(EXTRAS, float(empty_t), float(occupied_t))
    cm = pd.DataFrame(
        values["cm_actual_empty_occupied_vs_pass_review_fail"],
        index=["Actually empty", "Actually occupied"],
        columns=["PASS · Free", "REVIEW", "FAIL · Occupied"],
    ).reset_index(names="Actual class")
    daily = float(COSTS["assumptions"]["daily_spots"])
    reviews_per_day = daily * values["review_share"]
    extras = sum(extra.values())
    summary = (
        f"### Current policy · validation\n"
        f"**€{values['cost_per_1000_eur']:.2f}** estimated per 1,000 annotated spaces  ·  "
        f"**{values['review_share']*100:.2f}%** human review  ·  "
        f"**{reviews_per_day:.1f}** remote checks/day (at {daily:.0f} spaces/day)\n\n"
        f"Wrong automatic decisions: **{values['fp']} FP** · **{values['fn']} FN**  "
        f"|  Unmatched extra detections: **{extras:,}**, not included in cost."
    )
    y = []
    for t in GRID:
        d = hybrid_decisions(PRED_CLASS, CONF, float(t), float(occupied_t))
        y.append(metrics(TRUTH, d)["cost_per_1000_eur"])
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(GRID, y, lw=2.3, label="Hybrid, occupied threshold fixed")
    model_curve = COSTS["model_only_confidence_sweep"]
    ax.plot([p["confidence_threshold_empty"] for p in model_curve],
            [p["cost_per_1000_eur"] for p in model_curve],
            lw=1.5, alpha=.75, linestyle="--", label="Model only")
    ax.axvline(float(empty_t), ls=":", lw=1.8, label=f"Selected {empty_t:.2f}")
    ax.axvline(MODEL_ONLY, ls="-.", lw=1, alpha=.55,
               label=f"Model-only minimum {MODEL_ONLY:.2f}")
    ax.set(xlabel="Min confidence to approve an empty prediction",
           ylabel="Cost per 1,000 annotated spaces (€)")
    ax.set_xlim(0.01, 0.99)
    ax.grid(alpha=.17)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return summary, cm, fig


def classify(box, empty_t, occupied_t):
    if box["cls"] == 0:
        return "PASS" if box["conf"] >= float(empty_t) else "REVIEW"
    if box["cls"] == 1:
        return "FAIL" if box["conf"] >= float(occupied_t) else "REVIEW"
    return "REVIEW"


def infer_image(image):
    """Run YOLO only on image upload/click, not every slider movement."""
    if image is None:
        return [], None
    raw = np.asarray(image, dtype=np.uint8)
    result = MODEL.predict(source=raw, conf=0.01, imgsz=640, iou=0.45,
                           agnostic_nms=True, max_det=200, device="cpu", verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for cls, score, xyxy in zip(result.boxes.cls.int().cpu().tolist(),
                                    result.boxes.conf.cpu().tolist(),
                                    result.boxes.xyxy.cpu().tolist()):
            if cls not in (0, 1):
                continue
            detections.append({"cls": int(cls), "conf": float(score),
                               "xyxy": [float(v) for v in xyxy]})
    return detections, raw


def draw_predictions(image, detections, empty_t, occupied_t, min_display_conf,
                     show_review, max_visible):
    if image is None:
        return None, "Upload a photo to see predictions."
    canvas = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = font_for(canvas.width)
    # Presentation filter only. Does NOT change offline costs or the YOLO threshold policy.
    qualifying = [p for p in detections if p["conf"] >= float(min_display_conf)]
    display = [(p, classify(p, empty_t, occupied_t)) for p in qualifying]
    if not show_review:
        display = [(p, action) for p, action in display if action != "REVIEW"]
    display.sort(key=lambda pair: pair[0]["conf"], reverse=True)
    display = display[:int(max_visible)]
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    used_labels = []
    thickness = max(2, min(4, int(canvas.width / 450)))
    for p, action in display:
        counts[action] += 1
        x1, y1, x2, y2 = [round(v) for v in p["xyxy"]]
        x1, x2 = max(0, x1), min(canvas.width-1, x2)
        y1, y2 = max(0, y1), min(canvas.height-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        color = COLORS[action]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
        name = "FREE" if p["cls"] == 0 else "OCCUPIED"
        label = f"{name}  {p['conf']:.2f}  |  {action}"
        text_box = draw.textbbox((0, 0), label, font=font)
        label_w = text_box[2] + 14
        label_h = text_box[3] - text_box[1] + 10
        lx = max(0, min(x1, canvas.width - label_w))
        # Prefer above; if crowded, try below. Limit label collisions.
        candidates = [y1-label_h-2, y2+3, y1+3]
        candidates = [max(0, min(y, canvas.height-label_h)) for y in candidates]
        ly = candidates[0]
        for candidate in candidates:
            area = (lx, candidate, lx+label_w, candidate+label_h)
            if all(area[2] < old[0] or area[0] > old[2] or
                   area[3] < old[1] or area[1] > old[3]
                   for old in used_labels):
                ly = candidate
                break
        draw.rounded_rectangle((lx, ly, lx+label_w, ly+label_h),
                               radius=4, fill=color)
        draw.text((lx+7, ly+3-text_box[1]), label, font=font, fill=(255, 255, 255))
        used_labels.append((lx, ly, lx+label_w, ly+label_h))

    hidden = len(detections) - len(display)
    message = (
        f"**Displayed:** {len(display)} boxes — "
        f"{counts['PASS']} pass · {counts['REVIEW']} review · {counts['FAIL']} fail.  "
        f"**Not displayed:** {hidden} low-confidence, review, or excess boxes.\n\n"
        f"Display filtering starts at **{float(min_display_conf):.2f}** and shows "
        f"up to **{int(max_visible)}** boxes. **It affects this image only;** "
        f"the cost calculations still use the full cached validation data. "
        f"These boxes are unverified detections, **not identified physical parking bays**."
    )
    return np.asarray(canvas), message


def run_prediction(image, empty_t, occupied_t, min_display_conf,
                   show_review, max_visible):
    detections, raw = infer_image(image)
    annotated, note = draw_predictions(raw, detections, empty_t, occupied_t,
                                       min_display_conf, show_review, max_visible)
    return detections, annotated, note


def redraw(image, detections, empty_t, occupied_t, min_display_conf,
           show_review, max_visible):
    return draw_predictions(image, detections or [], empty_t, occupied_t,
                            min_display_conf, show_review, max_visible)


GRADIO_MAJOR = int(gr.__version__.split(".")[0])
BLOCKS_STYLE = {"theme": gr.themes.Soft(), "css": CSS} if GRADIO_MAJOR < 6 else {}

with gr.Blocks(title="Parking Operations · YOLO11n", **BLOCKS_STYLE) as demo:
    gr.Markdown(
        "# Parking Operations | AI Decision Dashboard\n"
        "Fine-tuned **YOLO11n** · Parking occupancy · Validation-based decision support",
        elem_id="hero",
    )
    gr.Markdown(
        "**Business rules:** Free → PASS · Occupied → FAIL · Uncertain → Human review. "
        "**Assumed costs:** €1 FP, €5 FN, €0.50 per remote review.", elem_id="statusline"
    )

    with gr.Row():
        empty_slider = gr.Slider(
            0.01, 0.99, value=SELECTED_EMPTY, step=0.01,
            label="Approve FREE when confidence ≥", info="Business threshold")
        occupied_slider = gr.Slider(
            0.01, 0.99, value=SELECTED_OCCUPIED, step=0.01,
            label="Reject OCCUPIED when confidence ≥", info="Business threshold")

    with gr.Tabs():
        with gr.Tab("Inspect a photo"):
            gr.Markdown(
                "Upload a parking photo. **Only clear predictions are drawn by default**; "
                "adjust the display filter for more detail. The business thresholds remain separate."
            )
            with gr.Row():
                uploaded = gr.Image(type="numpy", label="Original camera image", height=440)
                annotated = gr.Image(type="numpy", label="Clean decision overlay", height=440,
                                     interactive=False, elem_id="imagePanel")
            inference_note = gr.Markdown("Upload a photo to see predictions.")
            with gr.Accordion("Display controls — do not affect cost calculations", open=False):
                with gr.Row():
                    display_conf = gr.Slider(0.01, 0.95, value=0.30, step=0.01,
                                             label="Minimum confidence to draw")
                    max_visible = gr.Slider(1, 40, value=10, step=1,
                                            label="Maximum displayed boxes")
                    show_review = gr.Checkbox(value=False, label="Also draw REVIEW boxes")
            run_button = gr.Button("Analyze image", variant="primary")
            cached_detections = gr.State([])
            run_inputs = [uploaded, empty_slider, occupied_slider, display_conf,
                          show_review, max_visible]
            run_outputs = [cached_detections, annotated, inference_note]
            run_button.click(run_prediction, inputs=run_inputs, outputs=run_outputs)
            uploaded.change(run_prediction, inputs=run_inputs, outputs=run_outputs)
            draw_inputs = [uploaded, cached_detections, empty_slider, occupied_slider,
                           display_conf, show_review, max_visible]
            for control in (empty_slider, occupied_slider, display_conf, show_review, max_visible):
                control.change(redraw, inputs=draw_inputs,
                               outputs=[annotated, inference_note])
            gr.Markdown(
                "*This is a detector demo. Unknown or undetected bays cannot be marked free "
                "without a fixed site-specific bay map.*", elem_id="smallhelp")

        with gr.Tab("Operating costs & human review"):
            summary_view = gr.Markdown()
            confusion_view = gr.Dataframe(label="Validation: actual parking status vs. operational decision",
                                           interactive=False)
            cost_view = gr.Plot(label="Cost versus model confidence threshold")
            c = COSTS["comparison_eur_per_1000"]
            gr.Markdown(
                f"**Comparison (€/1,000 annotated bays, original selected optimum):** "
                f"assume all free **€{c['blind_assume_all_empty']:.2f}** · "
                f"assume all occupied **€{c['blind_assume_all_occupied']:.2f}** · "
                f"human checks all **€{c['human_review_all']:.2f}** · "
                f"model only **€{c['model_only']:.2f}** · "
                f"hybrid **€{c['hybrid']:.2f}**."
            )

        with gr.Tab("Model evaluation"):
            gr.Markdown("### Model comparison · 208 usable test images")
            gr.Dataframe(BASELINES_DF, interactive=False)
            gr.Markdown("### Results by class · fine-tuned YOLO11n")
            gr.Dataframe(CLASS_DF, interactive=False)
            gr.Markdown("*COCO zero-shot detects vehicles, not parking bays; comparisons are approximate.*")

        with gr.Tab("Methodology & limitations"):
            gr.Markdown(
                "**Validation uses annotated parking-space boxes to associate detections "
                "with a particular space.** This information would not be available "
                "for arbitrary uploaded photos without a fixed bay map. "
                "Additional unmatched predictions are reported but excluded from costs; "
                "thus the financial numbers are **optimistic offline estimates, not "
                "verified production savings**. Several video sources occur across "
                "train/validation/test, so measured performance can be inflated. "
                "The class confidence scores are not calibrated probabilities. "
                "Photo display filters never change the stored validation metrics."
            )

    for slider in (empty_slider, occupied_slider):
        slider.change(update_analysis, inputs=[empty_slider, occupied_slider],
                      outputs=[summary_view, confusion_view, cost_view])
    demo.load(update_analysis, inputs=[empty_slider, occupied_slider],
              outputs=[summary_view, confusion_view, cost_view])


if __name__ == "__main__":
    launch_style = {"theme": gr.themes.Soft(), "css": CSS} if GRADIO_MAJOR >= 6 else {}
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("APP_PORT", "7860")),
                **launch_style)
