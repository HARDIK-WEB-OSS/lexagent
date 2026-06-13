"""
Evaluate LexAgent using the fine-tuned classification model directly
+ the production risk scoring engine.
"""
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.risk_agent import RiskAgent

def run_lexagent(dataset: list) -> list:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = Path("models/legal-bert-local")
    print(f"Loading fine-tuned model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Read id2label from config
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"Labels: {len(id2label)} — {list(id2label.values())[:5]}...")

    risk_agent = RiskAgent()
    print("Agents loaded. Running evaluation...")

    results = []
    for i, item in enumerate(dataset):
        # ── Classify with fine-tuned model ──────────────────────────
        inputs  = tokenizer(
            item["clause_text"],
            truncation=True, padding=True,
            max_length=256, return_tensors="pt"
        )
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id    = logits.argmax(-1).item()
        pred_label = id2label[pred_id]
        pred_score = torch.softmax(logits, dim=-1)[0][pred_id].item()

        # ── Risk score ───────────────────────────────────────────────
        clause = {
            "clause_id":   item["id"],
            "clause_text": item["clause_text"],
            "label":       pred_label,
            "label_score": pred_score,
            "references":  [],
            "definitions": {},
            "page_hint":   1,
        }
        scored, _ = risk_agent.score([clause])
        scored     = scored[0]

        results.append({
            "id":             item["id"],
            "true_label":     item["true_label"],
            "true_risk":      item["true_risk"],
            "pred_label":     pred_label,
            "pred_label_score": round(pred_score, 4),
            "pred_risk":      scored["risk_level"],
            "pred_risk_score": scored["risk_score"],
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(dataset)}")

    return results


if __name__ == "__main__":
    with open("evaluation/test_dataset.json") as f:
        dataset = json.load(f)

    results = run_lexagent(dataset)

    with open("evaluation/lexagent_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"LexAgent predictions complete: {len(results)} clauses")

    # Quick sanity check
    from collections import Counter
    print("Pred label distribution:", Counter(r["pred_label"] for r in results).most_common(5))
    print("Pred risk distribution: ", Counter(r["pred_risk"]  for r in results).most_common())
