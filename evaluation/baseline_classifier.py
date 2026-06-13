"""
Simple regex/keyword baseline classifier.
This is what LexAgent is being compared AGAINST.
It uses only keyword matching — zero ML, zero context awareness.
"""
import re
import json
from typing import Tuple

# ── Keyword maps for label classification ──
LABEL_KEYWORDS = {
    "Non-Compete": ["non-compete", "not compete", "competitive activity", "competing business", "restraint of trade", "non compete"],
    "IP Assignment": ["work product", "inventions", "intellectual property", "assign", "copyright", "patent", "ip assignment", "all inventions"],
    "Termination": ["terminate", "termination", "notice of termination", "resignation"],
    "Confidentiality/NDA": ["confidential", "confidentiality", "non-disclosure", "nda", "proprietary information", "trade secret"],
    "Liability Limitation": ["limitation of liability", "not be liable", "no liability", "cap on liability", "exclude liability", "liable"],
    "Indemnification": ["indemnif", "hold harmless", "indemnity"],
    "Force Majeure": ["force majeure", "act of god", "beyond reasonable control", "natural disaster", "pandemic"],
    "Dispute Resolution": ["arbitration", "mediation", "dispute", "settlement"],
    "Governing Law": ["governing law", "governed by the laws", "jurisdiction of"],
    "Payment Terms": ["salary", "payment", "compensation", "fees", "remuneration", "invoice"],
    "Non-Solicitation": ["non-solicitation", "not solicit", "poaching", "recruit"],
    "Warranty": ["warrant", "warranty", "representation", "as is", "no warranty"],
    "Data Privacy": ["personal data", "data protection", "gdpr", "privacy"],
    "Assignment": ["assign this agreement", "novation", "transfer of rights"],
    "Renewal/Expiration": ["auto-renew", "automatically renew", "renewal", "expire", "expiration"],
    "Severance": ["severance", "severance pay", "separation pay"],
    "Audit Rights": ["audit", "audit rights", "inspect records"],
    "Uncapped Liability": ["unlimited liability", "uncapped", "no cap", "without limitation"],
}

# ── Keyword maps for risk scoring ──
RISK_HIGH_KEYWORDS = [
    "worldwide", "global", "perpetual", "indefinite", "forever", "in perpetuity",
    "no liability", "zero liability", "not be liable", "immediate termination",
    "without notice", "all work product", "personal time", "personal equipment",
    "unlimited liability", "uncapped", "no severance", "no warranty",
    "sole discretion", "without consent", "without notice",
]
RISK_CRITICAL_KEYWORDS = [
    "all work product.*personal time", "personal equipment.*outside working hours",
    "unlimited liability", "uncapped.*liability",
]
RISK_MEDIUM_KEYWORDS = [
    "assign", "auto-renew", "automatically renew", "without prior notice",
    "sole discretion", "may modify", "reserve the right",
]
RISK_LOW_KEYWORDS = [
    "mutual", "both parties", "prior written consent", "30 days notice",
    "publicly available", "carveout", "schedule a", "excluded",
    "limited to", "only for", "within scope",
]

DURATION_PATTERN = re.compile(r'(\d+)\s*(year|month|week|day)s?', re.IGNORECASE)


def classify_label(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for label, keywords in LABEL_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[label] = score
    if not scores:
        return "Other"
    return max(scores, key=scores.get)


def classify_risk(text: str, label: str) -> str:
    text_lower = text.lower()

    # CRITICAL: regex patterns
    for pattern in RISK_CRITICAL_KEYWORDS:
        if re.search(pattern, text_lower):
            return "CRITICAL"

    # HIGH: keyword presence
    if sum(1 for kw in RISK_HIGH_KEYWORDS if kw in text_lower) >= 1:
        # Duration check for non-compete
        if label == "Non-Compete":
            durations = DURATION_PATTERN.findall(text)
            for num, unit in durations:
                months = int(num) * 12 if "year" in unit else int(num)
                if months > 12:
                    return "HIGH"
        return "HIGH"

    # MEDIUM
    if sum(1 for kw in RISK_MEDIUM_KEYWORDS if kw in text_lower) >= 1:
        return "MEDIUM"

    # LOW
    if sum(1 for kw in RISK_LOW_KEYWORDS if kw in text_lower) >= 1:
        return "LOW"

    return "NONE"


def run_baseline(dataset: list) -> list:
    results = []
    for item in dataset:
        pred_label = classify_label(item["clause_text"])
        pred_risk = classify_risk(item["clause_text"], pred_label)
        results.append({
            "id": item["id"],
            "true_label": item["true_label"],
            "true_risk": item["true_risk"],
            "pred_label": pred_label,
            "pred_risk": pred_risk,
        })
    return results


if __name__ == "__main__":
    with open("evaluation/test_dataset.json") as f:
        dataset = json.load(f)

    results = run_baseline(dataset)

    with open("evaluation/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline predictions complete: {len(results)} clauses")
