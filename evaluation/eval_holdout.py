"""
Evaluate fine-tuned model on held-out test set only.
These 18 clauses were never seen during training.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.risk_agent import RiskAgent

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = Path("models/legal-bert-final")
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    model      = AutoModelForSequenceClassification.from_pretrained(model_path)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label   = {int(k): v for k, v in model.config.id2label.items()}
    risk_agent = RiskAgent()

    with open("evaluation/test_holdout.json") as f:
        test_data = json.load(f)
    with open("evaluation/test_dataset.json") as f:
        all_data = json.load(f)

    # Baseline on same holdout set
    with open("evaluation/baseline_results.json") as f:
        baseline_all = json.load(f)
    holdout_ids   = {d['id'] for d in test_data}
    baseline_hold = [r for r in baseline_all if r['id'] in holdout_ids]

    print(f"Evaluating on {len(test_data)} held-out clauses (never seen during training)\n")

    lexagent_results = []
    for item in test_data:
        inputs  = tokenizer(item["clause_text"], truncation=True,
                            padding=True, max_length=256, return_tensors="pt")
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id    = logits.argmax(-1).item()
        pred_label = id2label[pred_id]
        pred_score = torch.softmax(logits, dim=-1)[0][pred_id].item()

        clause = {"clause_id": item["id"], "clause_text": item["clause_text"],
                  "label": pred_label, "label_score": pred_score,
                  "references": [], "definitions": {}, "page_hint": 1}
        scored, _ = risk_agent.score([clause])

        lexagent_results.append({
            "id": item["id"], "true_label": item["true_label"],
            "true_risk": item["true_risk"], "pred_label": pred_label,
            "pred_risk": scored[0]["risk_level"],
        })

    # Compute metrics
    from collections import defaultdict
    def metrics(results, true_key, pred_key):
        classes = sorted(set(r[true_key] for r in results))
        tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
        correct = 0
        for r in results:
            t, p = r[true_key], r[pred_key]
            if t == p: correct += 1; tp[t] += 1
            else: fp[p] += 1; fn[t] += 1
        per = {}
        for c in classes:
            p = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0
            r = tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0
            f = 2*p*r/(p+r) if p+r else 0
            per[c] = (round(p,3), round(r,3), round(f,3))
        mf1 = sum(v[2] for v in per.values())/len(per)
        return round(correct/len(results),3), round(mf1,3), per

    la_clf_acc,  la_clf_f1,  la_clf_per  = metrics(lexagent_results, "true_label", "pred_label")
    la_risk_acc, la_risk_f1, la_risk_per = metrics(lexagent_results, "true_risk",  "pred_risk")
    bl_clf_acc,  bl_clf_f1,  bl_clf_per  = metrics(baseline_hold,    "true_label", "pred_label")
    bl_risk_acc, bl_risk_f1, bl_risk_per = metrics(baseline_hold,    "true_risk",  "pred_risk")

    bar = "═"*62
    print(f"{bar}")
    print(f"  HELD-OUT TEST SET RESULTS  (n={len(test_data)}, never seen in training)")
    print(f"{bar}")
    print(f"\n  {'Metric':<35} {'Baseline':>10} {'LexAgent':>10} {'Delta':>8}")
    print(f"  {'-'*63}")
    for name, bl, la in [
        ("Classification Accuracy",  bl_clf_acc,  la_clf_acc),
        ("Classification Macro F1",  bl_clf_f1,   la_clf_f1),
        ("Risk Scoring Accuracy",    bl_risk_acc, la_risk_acc),
        ("Risk Scoring Macro F1",    bl_risk_f1,  la_risk_f1),
    ]:
        delta = la - bl
        sym   = "▲" if delta >= 0 else "▼"
        print(f"  {name:<35} {bl:>9.1%} {la:>9.1%} {sym}{abs(delta):>6.1%}")

    print(f"\n  Per-class classification (holdout):")
    print(f"  {'Class':<30} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*48}")
    for cls, (p,r,f) in sorted(la_clf_per.items()):
        print(f"  {cls:<30} {p:>6.1%} {r:>6.1%} {f:>6.1%}")

    # Save
    report = {
        "holdout_n": len(test_data),
        "lexagent":  {"clf_acc": la_clf_acc, "clf_f1": la_clf_f1,
                      "risk_acc": la_risk_acc, "risk_f1": la_risk_f1},
        "baseline":  {"clf_acc": bl_clf_acc, "clf_f1": bl_clf_f1,
                      "risk_acc": bl_risk_acc, "risk_f1": bl_risk_f1},
    }
    with open("evaluation/holdout_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved → evaluation/holdout_report.json")

if __name__ == "__main__":
    main()
