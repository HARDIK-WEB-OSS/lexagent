"""
Computes and prints the full evaluation report.
Metrics: Accuracy, Precision, Recall, F1 (macro + per-class) for both
label classification and risk level prediction.
"""
import json
from collections import defaultdict, Counter
import math

def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)

def evaluate(results: list, task: str = "label"):
    true_key = "true_label" if task == "label" else "true_risk"
    pred_key = "pred_label" if task == "label" else "pred_risk"

    classes = sorted(set(r[true_key] for r in results))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0

    for r in results:
        t, p = r[true_key], r[pred_key]
        if t == p:
            correct += 1
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    accuracy = correct / len(results)
    per_class = {}
    for cls in classes:
        p, r, f1 = precision_recall_f1(tp[cls], fp[cls], fn[cls])
        support = sum(1 for x in results if x[true_key] == cls)
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": support}

    # Macro average (unweighted)
    macro_p = sum(v["precision"] for v in per_class.values()) / len(per_class)
    macro_r = sum(v["recall"]    for v in per_class.values()) / len(per_class)
    macro_f1= sum(v["f1"]        for v in per_class.values()) / len(per_class)

    return {
        "accuracy": round(accuracy, 3),
        "macro_precision": round(macro_p, 3),
        "macro_recall": round(macro_r, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class": per_class,
        "n": len(results),
    }

def print_report(name: str, label_eval: dict, risk_eval: dict):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {name}")
    print(bar)

    for task_name, ev in [("CLAUSE CLASSIFICATION", label_eval), ("RISK LEVEL PREDICTION", risk_eval)]:
        print(f"\n  ── {task_name} (n={ev['n']}) ──")
        print(f"  Accuracy:         {ev['accuracy']:.1%}")
        print(f"  Macro Precision:  {ev['macro_precision']:.1%}")
        print(f"  Macro Recall:     {ev['macro_recall']:.1%}")
        print(f"  Macro F1:         {ev['macro_f1']:.1%}")
        print(f"\n  {'Class':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
        print(f"  {'-'*53}")
        for cls, m in sorted(ev['per_class'].items()):
            print(f"  {cls:<30} {m['precision']:>6.1%} {m['recall']:>6.1%} {m['f1']:>6.1%} {m['support']:>5}")

def print_delta(baseline_label, lexagent_label, baseline_risk, lexagent_risk):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  DELTA: LexAgent vs Regex Baseline")
    print(bar)
    for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        bl = baseline_label[metric]
        la = lexagent_label[metric]
        delta = la - bl
        symbol = "▲" if delta > 0 else "▼"
        print(f"  Classification {metric:<20} Baseline: {bl:.1%}  LexAgent: {la:.1%}  {symbol} {abs(delta):.1%}")
    print()
    for metric in ["accuracy", "macro_f1"]:
        bl = baseline_risk[metric]
        la = lexagent_risk[metric]
        delta = la - bl
        symbol = "▲" if delta > 0 else "▼"
        print(f"  Risk scoring   {metric:<20} Baseline: {bl:.1%}  LexAgent: {la:.1%}  {symbol} {abs(delta):.1%}")

    print(f"\n  ── HEADLINE METRIC ──")
    clf_delta = lexagent_label["macro_f1"] - baseline_label["macro_f1"]
    risk_delta = lexagent_risk["macro_f1"] - baseline_risk["macro_f1"]
    print(f"  Classification F1 improvement: {clf_delta:+.1%}")
    print(f"  Risk scoring   F1 improvement: {risk_delta:+.1%}")


if __name__ == "__main__":
    with open("evaluation/baseline_results.json") as f:
        baseline = json.load(f)
    with open("evaluation/lexagent_results.json") as f:
        lexagent = json.load(f)

    print_report("REGEX BASELINE", evaluate(baseline, "label"), evaluate(baseline, "risk"))
    print_report("LEXAGENT",       evaluate(lexagent, "label"), evaluate(lexagent, "risk"))
    print_delta(
        evaluate(baseline, "label"), evaluate(lexagent, "label"),
        evaluate(baseline, "risk"),  evaluate(lexagent, "risk"),
    )

    # Save full report
    report = {
        "baseline": {"label": evaluate(baseline, "label"), "risk": evaluate(baseline, "risk")},
        "lexagent": {"label": evaluate(lexagent, "label"), "risk": evaluate(lexagent, "risk")},
    }
    with open("evaluation/full_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved → evaluation/full_report.json")
