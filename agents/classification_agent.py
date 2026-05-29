"""
Agent 3: Classification Agent
Zero-shot clause classification using nlpaueb/legal-bert-base-uncased.
Falls back to keyword heuristics if model unavailable.
"""
import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CUAD_LABELS = [
    "Non-Compete",
    "IP Assignment",
    "Termination",
    "Governing Law",
    "Payment Terms",
    "Confidentiality/NDA",
    "Liability Limitation",
    "Indemnification",
    "Force Majeure",
    "Dispute Resolution",
    "Notice Period",
    "Non-Solicitation",
    "Warranty",
    "Audit Rights",
    "Assignment",
    "Change of Control",
    "Data Privacy",
    "Exclusivity",
    "Liquidated Damages",
    "Most Favored Nation",
    "Renewal/Expiration",
    "Severance",
    "Source Code Escrow",
    "Uncapped Liability",
    "Anti-Assignment",
    "Insurance",
    "Minimum Commitment",
    "Revenue Share",
    "Price Restrictions",
    "Cap on Liability",
    "Affiliate IP",
    "Covenant Not to Sue",
    "EBITDA",
    "Effective Date",
    "Expiration Date",
    "Jurisdiction",
    "License Grant",
    "Post-Termination Services",
    "Price Adjustment",
    "Unlimited License",
    "Volume Restriction",
]

MIN_WORDS = 20
MIN_SCORE_THRESHOLD = 0.15

# Keyword-based fallback classifier
KEYWORD_MAP = {
    "Non-Compete": [
        "non-compete", "non compete", "not compete", "competitive activity",
        "competing business", "restraint of trade",
    ],
    "IP Assignment": [
        "intellectual property", "work product", "inventions", "assign",
        "copyright", "patent", "ownership of ip",
    ],
    "Termination": [
        "terminate", "termination", "notice of termination", "end of agreement",
        "cancel", "cancellation",
    ],
    "Governing Law": [
        "governing law", "governed by", "laws of", "applicable law",
    ],
    "Payment Terms": [
        "payment", "invoice", "fee", "compensation", "salary", "remuneration",
        "due date", "net 30", "net 60",
    ],
    "Confidentiality/NDA": [
        "confidential", "confidentiality", "non-disclosure", "nda",
        "proprietary information", "trade secret",
    ],
    "Liability Limitation": [
        "limit of liability", "limitation of liability", "liable", "cap on liability",
        "not be liable", "exclude liability",
    ],
    "Indemnification": [
        "indemnif", "hold harmless", "defend", "indemnity",
    ],
    "Force Majeure": [
        "force majeure", "act of god", "beyond reasonable control",
        "natural disaster", "pandemic",
    ],
    "Dispute Resolution": [
        "dispute", "arbitration", "mediation", "resolve", "settlement",
        "court", "litigation",
    ],
    "Notice Period": [
        "notice period", "days notice", "written notice", "notify",
    ],
    "Warranty": [
        "warrant", "warranty", "representation", "guarantee",
    ],
    "Data Privacy": [
        "personal data", "data protection", "gdpr", "privacy", "data subject",
        "processing of data",
    ],
    "Governing Law": ["governing law", "governed by the laws"],
    "Jurisdiction": ["jurisdiction", "courts of", "submit to jurisdiction"],
    "Severance": ["severance", "severance pay", "separation pay"],
    "Non-Solicitation": ["non-solicitation", "not solicit", "poaching"],
    "Assignment": ["assign", "assignment", "transfer of rights", "novation"],
    "Renewal/Expiration": ["renew", "renewal", "expire", "expiration", "term of agreement"],
}


class ClassificationAgent:
    """Classifies legal clauses using zero-shot NLP or keyword fallback."""

    def __init__(self):
        self._pipeline = None
        self._pipeline_loaded = False
        self._load_model()

    def _load_model(self):
        import os
        if os.getenv("CLOUD_DEPLOY"):
            logger.info("CLOUD_DEPLOY=true — using keyword classification (no torch needed).")
            self._pipeline_loaded = False
            return
        try:
            from transformers import pipeline

            logger.info("Loading legal-bert zero-shot classification pipeline...")
            self._pipeline = pipeline(
                "zero-shot-classification",
                model="nlpaueb/legal-bert-base-uncased",
                device=-1,  # CPU default; upgraded to CUDA below if available
            )
            # Try to move to GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    self._pipeline = pipeline(
                        "zero-shot-classification",
                        model="nlpaueb/legal-bert-base-uncased",
                        device=0,
                    )
                    logger.info("Using CUDA GPU for classification.")
                else:
                    logger.info("Using CPU for classification.")
            except Exception:
                pass

            self._pipeline_loaded = True
            logger.info("Classification model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load legal-bert model: {e}. "
                "Falling back to keyword-based classification."
            )
            self._pipeline_loaded = False

    def classify(self, clauses: List[Dict]) -> List[Dict]:
        enriched = []
        for clause in clauses:
            text = clause.get("clause_text", "")
            word_count = len(text.split())

            if word_count < MIN_WORDS:
                clause["label"] = "Other"
                clause["label_score"] = 0.0
                clause["secondary_label"] = None
                enriched.append(clause)
                continue

            if self._pipeline_loaded:
                label, score, secondary_label = self._classify_with_model(text)
            else:
                label, score, secondary_label = self._classify_with_keywords(text)

            clause["label"] = label
            clause["label_score"] = round(score, 4)
            clause["secondary_label"] = secondary_label
            enriched.append(clause)

        logger.info(
            f"Classified {len(enriched)} clauses. "
            f"Method: {'model' if self._pipeline_loaded else 'keywords'}"
        )
        return enriched

    def _classify_with_model(self, text: str):
        # Truncate to first 512 chars for model token limit
        truncated = text[:512]
        try:
            result = self._pipeline(
                truncated,
                candidate_labels=CUAD_LABELS,
                multi_label=False,
            )
            labels = result["labels"]
            scores = result["scores"]

            top_label = labels[0]
            top_score = scores[0]

            secondary_label = None
            if len(scores) > 1 and scores[1] >= MIN_SCORE_THRESHOLD:
                secondary_label = labels[1]

            return top_label, top_score, secondary_label
        except Exception as e:
            logger.warning(f"Model inference failed: {e}. Using keywords.")
            return self._classify_with_keywords(text)

    def _classify_with_keywords(self, text: str):
        text_lower = text.lower()
        best_label = "Other"
        best_score = 0.0
        second_label = None
        second_score = 0.0

        for label, keywords in KEYWORD_MAP.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            score = min(hits / max(len(keywords), 1), 1.0)
            if score > best_score:
                second_label = best_label if best_score >= MIN_SCORE_THRESHOLD else None
                second_score = best_score
                best_label = label
                best_score = score
            elif score > second_score and score >= MIN_SCORE_THRESHOLD:
                second_label = label
                second_score = score

        # Normalize keyword score to 0-1 range and make it plausible
        # (keyword hits give integers, cap at realistic confidence)
        normalized = min(best_score * 0.6 + 0.2, 0.85) if best_score > 0 else 0.1
        return best_label, normalized, second_label
