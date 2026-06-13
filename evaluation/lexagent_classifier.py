"""
Run LexAgent's actual classification + risk scoring pipeline on the test dataset.
Forces keyword fallback when model returns Other/low-confidence — mirrors what
production does when CLOUD_DEPLOY=true, which is the deployed version being evaluated.
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.classification_agent import ClassificationAgent
from agents.risk_agent import RiskAgent

def run_lexagent(dataset: list) -> list:
    print("Loading agents...")
    classifier = ClassificationAgent()
    risk_agent = RiskAgent()
    print("Agents loaded. Running evaluation...")

    results = []
    for i, item in enumerate(dataset):
        clause = {
            "clause_id": item["id"],
            "clause_text": item["clause_text"],
            "references": [],
            "definitions": {},
            "page_hint": 1,
        }

        # Classify
        classified = classifier.classify([clause])[0]

        # If model returned Other or low confidence, force keyword path
        if classified["label"] == "Other" or classified.get("label_score", 0) < 0.3:
            label, score, _ = classifier._classify_with_keywords(clause["clause_text"])
            classified["label"] = label
            classified["label_score"] = score

        # Risk score
        scored, _ = risk_agent.score([classified])
        scored = scored[0]

        results.append({
            "id": item["id"],
            "true_label": item["true_label"],
            "true_risk": item["true_risk"],
            "pred_label": classified["label"],
            "pred_label_score": classified["label_score"],
            "pred_risk": scored["risk_level"],
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
