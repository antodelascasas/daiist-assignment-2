"""Operating-cost analysis for parking detection (validation only).

Uses cached YOLO detections, never retrains or runs inference.
Evaluates EACH annotated parking space with ground-truth geometry as an offline
anchor, not ground-truth class for choosing a prediction. Real deployment needs
mapped physical spaces; extra unmatched detections are reported but uncosted.

Run from project root:
    uv run python cost_analysis.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results' / 'parking_predictions_valid.json'
OUT = ROOT / 'results'

# Hypothetical costs, not measured company financials.
COST_FP = 1.0          # Empty spot wrongly rejected as occupied.
COST_FN = 5.0          # Occupied spot wrongly recommended as free.
COST_REVIEW = 0.50     # Remote human check, assumed accurate.
DAILY_SPOTS = 500
MINUTES_PER_REVIEW = 1.0
PAID_HOURS_PER_SHIFT = 8.0
PRODUCTIVE_SHARE = 0.70
MATCH_IOU = 0.50
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return intersection / (area_a + area_b - intersection + 1e-12)


def build_cases(samples):
    """Match predicted boxes to known GT locations, WITHOUT using GT class.

    Each GT box is a virtual space anchor for *offline* evaluation only.
    Unmatched predictions cannot be reliably priced as real parking spaces.
    """
    truth, predicted_class, confidence = [], [], []
    extra = []
    for sample in samples:
        gt = sample['gt']
        matched = [None] * len(gt)
        for p in sorted(sample['trained'], key=lambda item: item['conf'], reverse=True):
            best_idx, best_iou = -1, MATCH_IOU
            for j, g in enumerate(gt):
                if matched[j] is not None:
                    continue
                overlap = iou(p['box'], g['box'])
                if overlap >= best_iou:
                    best_idx, best_iou = j, overlap
            if best_idx >= 0:
                matched[best_idx] = p
            else:
                extra.append((int(p['cls']), float(p['conf'])))
        for g, p in zip(gt, matched):
            truth.append(int(g['cls']))
            predicted_class.append(-1 if p is None else int(p['cls']))
            confidence.append(0.0 if p is None else float(p['conf']))
    return (
        np.array(truth, dtype=np.int8),
        np.array(predicted_class, dtype=np.int8),
        np.array(confidence, dtype=np.float64),
        np.array(extra, dtype=float).reshape(-1, 2),
    )


def weights_for_rate(truth, assumed_rate=None):
    observed = float(np.mean(truth == 1))
    if not 0.0 < observed < 1.0:
        raise ValueError('Need both GT classes')
    p = observed if assumed_rate is None else float(assumed_rate)
    if not 0 <= p <= 1:
        raise ValueError('Base rate must be between 0 and 1')
    return np.where(truth == 1, p / observed, (1-p)/(1-observed))


def hybrid_decisions(pred_class, conf, empty_threshold, occupied_threshold):
    # 0=PASS (available), 1=REVIEW, 2=FAIL (occupied).
    decision = np.ones(len(pred_class), dtype=np.int8)
    decision[(pred_class == 0) & (conf >= empty_threshold)] = 0
    decision[(pred_class == 1) & (conf >= occupied_threshold)] = 2
    return decision


def model_only_decisions(pred_class, conf, empty_threshold):
    # Conservative no-review default: unknown/low-conf -> OCCUPIED.
    decision = np.full(len(pred_class), 2, dtype=np.int8)
    decision[(pred_class == 0) & (conf >= empty_threshold)] = 0
    return decision


def metrics(truth, decision, base_rate=None, fn_cost=COST_FN):
    w = weights_for_rate(truth, base_rate)
    fp = (truth == 0) & (decision == 2)
    fn = (truth == 1) & (decision == 0)
    review = decision == 1
    cost = COST_FP * fp + fn_cost * fn + COST_REVIEW * review
    cm = [[int(np.sum((truth == actual) & (decision == category)))
           for category in (0, 1, 2)] for actual in (0, 1)]
    return {
        'cost_per_1000_eur': round(float(np.sum(cost * w) * 1000 / len(truth)), 4),
        'review_share': round(float(np.sum(review * w) / len(truth)), 6),
        'fp': int(fp.sum()), 'fn': int(fn.sum()),
        'review_count': int(review.sum()),
        'cm_actual_empty_occupied_vs_pass_review_fail': cm,
    }


def optimize_hybrid(truth, pred_class, confidence, assumed_rate=None, fn_cost=COST_FN):
    """Optimize both actual *predicted-class confidence* cutoffs independently.

    Pass decisions only arise from an 'empty' prediction; fail decisions only
    from an 'occupied' prediction. This makes the two cutoffs separable.
    """
    w = weights_for_rate(truth, assumed_rate)
    n = len(truth)
    results = []
    for predicted in (0, 1):
        choices = []
        for t in GRID:
            action = (pred_class == predicted) & (confidence >= t)
            # Starting from REVIEW ALL, replacing reviews by action.
            if predicted == 0:  # predict empty -> PASS; only FN can cost.
                delta = np.where(truth == 1, fn_cost - COST_REVIEW, -COST_REVIEW)
            else:  # predict occupied -> FAIL; only FP can cost.
                delta = np.where(truth == 0, COST_FP - COST_REVIEW, -COST_REVIEW)
            incremental = float(np.sum(w[action] * delta[action]))
            choices.append((incremental, -int(action.sum()), float(t)))
        results.append(min(choices)[2])
    t_empty, t_occ = results
    m = metrics(truth, hybrid_decisions(pred_class, confidence, t_empty, t_occ), assumed_rate, fn_cost)
    return {'empty_conf_threshold': t_empty, 'occupied_conf_threshold': t_occ,
            'signed_conf_low': round(-t_empty, 2), 'signed_conf_high': t_occ, **m}


def unmatched_above(extra, empty_threshold, occupied_threshold):
    if not len(extra):
        return {'unmatched_empty_predictions': 0, 'unmatched_occupied_predictions': 0}
    return {
        'unmatched_empty_predictions': int(np.sum((extra[:, 0] == 0) & (extra[:, 1] >= empty_threshold))),
        'unmatched_occupied_predictions': int(np.sum((extra[:, 0] == 1) & (extra[:, 1] >= occupied_threshold))),
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing validation predictions: {SOURCE}')
    samples = json.loads(SOURCE.read_text())
    truth, pcls, confidence, extra = build_cases(samples)
    if not len(truth):
        raise ValueError('Empty ground-truth dataset')
    base_rate = float(np.mean(truth == 1))

    model_curve = []
    for t in GRID:
        m = metrics(truth, model_only_decisions(pcls, confidence, t))
        model_curve.append({'confidence_threshold_empty': float(t), **m})
    best_model = min(model_curve, key=lambda item: (item['cost_per_1000_eur'], -item['confidence_threshold_empty']))
    hybrid = optimize_hybrid(truth, pcls, confidence)
    extra_hybrid = unmatched_above(extra, hybrid['empty_conf_threshold'], hybrid['occupied_conf_threshold'])

    alternative_costs = {
        'blind_assume_all_empty': round(COST_FN * base_rate * 1000, 4),
        'blind_assume_all_occupied': round(COST_FP * (1 - base_rate) * 1000, 4),
        'human_review_all': round(COST_REVIEW * 1000, 4),
        'model_only': best_model['cost_per_1000_eur'],
        'hybrid': hybrid['cost_per_1000_eur'],
    }

    sensitivities = []
    for rate in (0.50, base_rate, 0.90):
        for fn in (2.0, 5.0, 10.0):
            m = optimize_hybrid(truth, pcls, confidence, assumed_rate=rate, fn_cost=fn)
            sensitivities.append({'base_rate': round(rate, 6), 'fn_eur': fn,
                                  'empty_conf_threshold': m['empty_conf_threshold'],
                                  'occupied_conf_threshold': m['occupied_conf_threshold'],
                                  'cost_per_1000_eur': m['cost_per_1000_eur'],
                                  'review_share': m['review_share']})

    review_cases = DAILY_SPOTS * hybrid['review_share']
    labor_hours = review_cases * MINUTES_PER_REVIEW / 60
    fte = labor_hours / (PAID_HOURS_PER_SHIFT * PRODUCTIVE_SHARE)
    report = {
        'assumptions': {
            'fp_eur': COST_FP, 'fn_eur': COST_FN, 'review_eur': COST_REVIEW,
            'positive_class': 'occupied', 'daily_spots': DAILY_SPOTS,
            'review_minutes': MINUTES_PER_REVIEW,
            'review_assumed_accurate': True,
            'matching_iou_threshold': MATCH_IOU,
            'observed_base_rate': round(base_rate, 6),
            'note': 'Offline GT-box anchors; GT class never used for prediction. Deployment requires fixed space coordinates.',
            'model_only_policy': 'empty prediction >= threshold => PASS, everything else => FAIL (conservative)',
            'hybrid_policy': 'empty confidence >= empty threshold => PASS; occupied confidence >= occupied threshold => FAIL; everything else => REVIEW',
            'extra_unmatched_predictions': 'Reported separately, uncosted; may include hallucinated spaces or incomplete labels',
        },
        'gt_space_count': len(truth),
        'no_detection_at_conf_0_001': int(np.sum(pcls == -1)),
        'model_only_optimum': best_model,
        'hybrid_optimum': hybrid,
        'unmatched_hybrid': extra_hybrid,
        'comparison_eur_per_1000': alternative_costs,
        'staffing': {'daily_spots': DAILY_SPOTS,
                     'reviews_per_day': round(review_cases, 3),
                     'labor_hours_per_day': round(labor_hours, 4),
                     'fte': round(fte, 5),
                     'review_minutes_per_case': MINUTES_PER_REVIEW},
        'sensitivity': sensitivities,
        'model_only_confidence_sweep': model_curve,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / 'parking_costs_valid.json').write_text(json.dumps(report, indent=2))

    plt.figure(figsize=(8, 4.5))
    plt.plot([r['confidence_threshold_empty'] for r in model_curve],
             [r['cost_per_1000_eur'] for r in model_curve], label='Model only (unknown -> occupied)')
    plt.axvline(best_model['confidence_threshold_empty'], color='black', linestyle='--',
                label=f"Selected = {best_model['confidence_threshold_empty']:.2f}")
    plt.xlabel('Minimum confidence for an EMPTY prediction to PASS')
    plt.ylabel('Expected cost (€ / 1,000 annotated spaces)')
    plt.title('Validation: operating cost versus YOLO confidence cutoff')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'parking_cost_curve_valid.png', dpi=160)
    plt.close()

    print(f'Annotated spaces: {len(truth)}; occupied base rate: {base_rate:.2%}')
    print('Optimal MODEL ONLY (confidence threshold on EMPTY):',
          best_model['confidence_threshold_empty'],
          '| €/1,000:', best_model['cost_per_1000_eur'])
    print('Optimal HYBRID (empty confidence, occupied confidence):',
          hybrid['empty_conf_threshold'], hybrid['occupied_conf_threshold'],
          '| €/1,000:', hybrid['cost_per_1000_eur'],
          '| review share:', f"{hybrid['review_share']:.2%}")
    print('Hybrid confusion [actual empty,occupied] x [pass,review,fail]:',
          hybrid['cm_actual_empty_occupied_vs_pass_review_fail'])
    print('Unmatched EXTRA predictions at hybrid thresholds (not costed):', extra_hybrid)
    print('Scenario costs €/1,000:', alternative_costs)
    print('Human review cases/day:', round(review_cases, 2),
          '| hours/day:', round(labor_hours, 3), '| FTE:', round(fte, 5))
    print('Sensitivity results saved:', len(sensitivities), 'scenarios')
    print('Saved results/parking_costs_valid.json and results/parking_cost_curve_valid.png')


if __name__ == '__main__':
    main()
