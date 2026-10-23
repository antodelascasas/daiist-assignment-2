"""Evaluate parking-space detectors without training.

Run in the assignment repo, after downloading Roboflow dataset version 4:

    uv run python evaluate_parking.py --split valid

The evaluation compares fine-tuned YOLO11n against COCO zero-shot and
predict-nothing baselines, using the SAME images and IoU-based matching.
COCO car/bus/truck predictions map to occupied; empty has no COCO class.
The output is a reproducible evaluation aid, not a replacement for a
production deployment evaluation on held-out cameras.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'data' / 'parking-space-4'
OUTPUT_DIR = ROOT / 'results'
CLASSES = ('empty', 'occupied')
IOU_LEVELS = np.arange(0.5, 1.0, 0.05)
# Official pretrained COCO category IDs: car=2, bus=5, truck=7.
COCO_TO_OCCUPIED = {2, 5, 7}


def bounds(row):
    vals = list(map(float, row[1:]))
    if len(row) == 5:  # normalized YOLO detection: xc,yc,w,h
        x, y, w, h = vals
        b = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    else:  # normalized YOLO segmentation polygon: x1,y1,...
        xs, ys = vals[::2], vals[1::2]
        b = [min(xs), min(ys), max(xs), max(ys)]
    return [float(np.clip(z, 0, 1)) for z in b]


def read_gt(label):
    if not label.exists():
        return []  # a background image may legitimately have no label file
    lines = [s.split() for s in label.read_text().splitlines() if s.strip()]
    forms = set()
    for row in lines:
        if len(row) == 5:
            forms.add('box')
        elif len(row) >= 7 and len(row) % 2 == 1:
            forms.add('polygon')
        else:
            return None  # ignored: invalid annotation format
        if row[0] not in ('0', '1'):
            return None
    if len(forms) > 1:
        return None  # same image ignored by Ultralytics during training
    try:
        return [{'cls': int(row[0]), 'box': bounds(row)} for row in lines]
    except ValueError:
        return None


def load_samples(split):
    samples = []
    excluded = 0
    image_dir = DATASET / split / 'images'
    if not image_dir.exists():
        raise FileNotFoundError(f'Dataset missing: {image_dir}. Download it first.')
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        gt = read_gt(DATASET / split / 'labels' / (path.stem + '.txt'))
        if gt is None:
            excluded += 1
            continue
        samples.append({'image': path.name, 'gt': gt, '_path': str(path)})
    print(f'{split}: {len(samples)} usable images, {excluded} incompatible label files.')
    print('Ground truth instances:', dict(Counter(CLASSES[g['cls']] for s in samples for g in s['gt'])))
    return samples


def predict(samples, model_path, key, device):
    model = YOLO(str(model_path))
    is_coco = key == 'coco'
    for i, sample in enumerate(samples):
        result = model.predict(
            source=sample['_path'], device=device, imgsz=640,
            conf=0.001, iou=0.7, max_det=300, verbose=False,
        )[0]
        boxes = result.boxes
        predictions = []
        if boxes is not None:
            for cls, confidence, coords in zip(
                boxes.cls.int().cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.xyxyn.cpu().tolist(),
            ):
                if is_coco:
                    if cls not in COCO_TO_OCCUPIED:
                        continue
                    cls = 1
                predictions.append({
                    'cls': int(cls), 'conf': float(confidence),
                    'box': [float(x) for x in coords],
                })
        sample[key] = predictions
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f'{key}: {i+1}/{len(samples)} images evaluated', flush=True)


def iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-12)


def class_detections(samples, key, class_id, confidence=0.001):
    preds = [(p['conf'], i, p['box']) for i, s in enumerate(samples)
             for p in s.get(key, [])
             if p['cls'] == class_id and p['conf'] >= confidence]
    return sorted(preds, key=lambda p: p[0], reverse=True)


def match_class(samples, key, cls, threshold=0.5, confidence=0.001):
    gt = {i: [g['box'] for g in s['gt'] if g['cls'] == cls]
          for i, s in enumerate(samples)}
    total_gt = sum(len(v) for v in gt.values())
    matched = {i: set() for i in gt}
    predictions = class_detections(samples, key, cls, confidence)
    tp = np.zeros(len(predictions), dtype=float)
    fp = np.ones(len(predictions), dtype=float)
    for j, (_, img_idx, box) in enumerate(predictions):
        best_iou, best = threshold, -1
        for k, true_box in enumerate(gt[img_idx]):
            if k in matched[img_idx]:
                continue
            value = iou(box, true_box)
            if value >= best_iou:
                best_iou, best = value, k
        if best >= 0:
            matched[img_idx].add(best)
            tp[j], fp[j] = 1., 0.
    return tp, fp, total_gt


def precision_recall_ap(tp, fp, n_gt):
    if len(tp) == 0 or n_gt == 0:
        return 0., 0., 0.
    cum_tp = tp.cumsum()
    cum_fp = fp.cumsum()
    rec = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    # COCO-style 101-point interpolated AP from the confidence-ranked curve.
    ap = float(np.mean([np.max(prec[rec >= r], initial=0.)
                        for r in np.linspace(0, 1, 101)]))
    return float(prec[-1]), float(rec[-1]), ap


def confusion(samples, key, confidence=0.25, threshold=0.5):
    # Rows = actual [empty,occupied,background], columns = predicted same.
    cm = np.zeros((3, 3), dtype=int)
    for s in samples:
        used_gt = set()
        pred = sorted([p for p in s.get(key, []) if p['conf'] >= confidence],
                      key=lambda p: p['conf'], reverse=True)
        for p in pred:
            best_iou, best = threshold, -1
            for j, g in enumerate(s['gt']):
                if j in used_gt:
                    continue
                val = iou(p['box'], g['box'])
                if val >= best_iou:
                    best_iou, best = val, j
            if best >= 0:
                used_gt.add(best)
                cm[s['gt'][best]['cls'], p['cls']] += 1
            else:
                cm[2, p['cls']] += 1
        for j, g in enumerate(s['gt']):
            if j not in used_gt:
                cm[g['cls'], 2] += 1
    return cm.tolist()


def summarize(samples, key):
    per_class = {}
    for c, name in enumerate(CLASSES):
        tp, fp, n = match_class(samples, key, c, confidence=0.25)
        p, r, _ = precision_recall_ap(tp, fp, n)
        aps = []
        for iou_threshold in IOU_LEVELS:
            t, f, count = match_class(samples, key, c, float(iou_threshold))
            aps.append(precision_recall_ap(t, f, count)[2])
        per_class[name] = {
            'precision_at_0.25_iou50': round(p, 6),
            'recall_at_0.25_iou50': round(r, 6),
            'ap50': round(aps[0], 6),
            'map50_95': round(float(np.mean(aps)), 6),
            'ground_truth_instances': n,
        }
    cm = confusion(samples, key)
    return {
        'map50': round(float(np.mean([v['ap50'] for v in per_class.values()])), 6),
        'map50_95': round(float(np.mean([v['map50_95'] for v in per_class.values()])), 6),
        'per_class': per_class,
        'confusion_at_0.25_iou50': cm,
        'confusion_labels': ['empty', 'occupied', 'background'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    samples = load_samples(args.split)
    if not samples:
        raise RuntimeError('No usable images')
    weights = ROOT / 'weights' / 'best.pt'
    if not weights.exists():
        raise FileNotFoundError(f'Missing model: {weights}')
    predict(samples, weights, 'trained', args.device)
    predict(samples, 'yolo11n.pt', 'coco', args.device)
    for s in samples:
        s['none'] = []
        del s['_path']
    summary = {
        'split': args.split,
        'images_evaluated': len(samples),
        'iou_match': 0.5,
        'score_cutoff_for_confusion': 0.25,
        'mapping_coco_to_parking': {'car': 'occupied', 'bus': 'occupied',
                                    'truck': 'occupied', 'empty': 'not available'},
        'results': {key: summarize(samples, key)
                    for key in ('trained', 'coco', 'none')},
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_path = OUTPUT_DIR / f'parking_predictions_{args.split}.json'
    metrics_path = OUTPUT_DIR / f'parking_metrics_{args.split}.json'
    raw_path.write_text(json.dumps(samples, separators=(',', ':')))
    metrics_path.write_text(json.dumps(summary, indent=2))
    for key, result in summary['results'].items():
        print(f'\n{key}: mAP50={result["map50"]:.4f} '
              f'mAP50-95={result["map50_95"]:.4f}')
        for cls, m in result['per_class'].items():
            print(f'  {cls}: P@0.25={m["precision_at_0.25_iou50"]:.3f} '
                  f'R@0.25={m["recall_at_0.25_iou50"]:.3f} '
                  f'AP50={m["ap50"]:.3f}')
        print('  confusion [actual rows, predicted columns]')
        for row in result['confusion_at_0.25_iou50']:
            print(' ', row)
    print(f'\nSaved: {raw_path}\nSaved: {metrics_path}')
    print('Note: AP uses COCO-style 101-point interpolation and IoU greedy matching;')
    print('numbers may differ slightly from Ultralytics built-in validation.')


if __name__ == '__main__':
    main()
