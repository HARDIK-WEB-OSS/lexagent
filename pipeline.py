"""
Pipeline Orchestrator
Runs all 6 agents in sequence and assembles the final risk report.
"""
import uuid
import logging
from datetime import datetime, timezone
from collections import Counter
from typing import List, Dict

from agents.extraction_agent import ExtractionAgent
from agents.segmentation_agent import SegmentationAgent
from agents.classification_agent import ClassificationAgent
from agents.risk_agent import RiskAgent
from agents.contradiction_agent import ContradictionAgent
from agents.summarizer_agent import SummarizerAgent

logger = logging.getLogger(__name__)

HIGH_RISK_LEVELS = {"CRITICAL", "HIGH"}


class Pipeline:
    """Orchestrates the 6-agent LexAgent analysis pipeline."""

    def __init__(self):
        logger.info("Initializing pipeline agents...")
        self.extraction_agent = ExtractionAgent()
        self.segmentation_agent = SegmentationAgent()
        self.classification_agent = ClassificationAgent()
        self.risk_agent = RiskAgent()
        self.contradiction_agent = ContradictionAgent()
        self.summarizer_agent = SummarizerAgent()
        logger.info("Pipeline ready.")

    def run(self, file_path: str, settings: dict = None) -> dict:
        settings = settings or {}
        jurisdiction = settings.get("jurisdiction", "IN")
        comparative = settings.get("comparative", True)
        negotiate = settings.get("negotiate", True)

        contract_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        logger.info(f"[{contract_id}] Starting analysis: {file_path}")

        # ── Agent 1: Extraction ──────────────────────────────────────────
        logger.info(f"[{contract_id}] Agent 1: Extraction")
        extraction = self.extraction_agent.extract(file_path)
        raw_text = extraction["raw_text"]

        # ── Agent 2: Segmentation ────────────────────────────────────────
        logger.info(f"[{contract_id}] Agent 2: Segmentation")
        clauses = self.segmentation_agent.segment(raw_text)

        # Build reference map for contradiction agent
        reference_map = {c["clause_id"]: c.get("references", []) for c in clauses}

        # ── Agent 3: Classification ──────────────────────────────────────
        logger.info(f"[{contract_id}] Agent 3: Classification ({len(clauses)} clauses)")
        classified_clauses = self.classification_agent.classify(clauses)

        # ── Agent 4: Risk Scoring ────────────────────────────────────────
        logger.info(f"[{contract_id}] Agent 4: Risk Scoring")
        scored_clauses, missing_clauses = self.risk_agent.score(classified_clauses, jurisdiction=jurisdiction)

        # Compute overall risk score
        if scored_clauses:
            overall_risk_score = sum(c["risk_score"] for c in scored_clauses) / len(scored_clauses)
        else:
            overall_risk_score = 0.0

        # ── Agent 5: Contradiction Detection ────────────────────────────
        logger.info(f"[{contract_id}] Agent 5: Contradiction Detection")
        contradictions = self.contradiction_agent.detect(scored_clauses, reference_map)

        # ── Assemble partial report for summarizer ───────────────────────
        risk_distribution = Counter(c["risk_level"] for c in scored_clauses)
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"):
            risk_distribution.setdefault(level, 0)

        critical_clauses = [
            c for c in scored_clauses if c.get("risk_level") in HIGH_RISK_LEVELS
        ]
        critical_clauses.sort(key=lambda c: c["risk_score"], reverse=True)

        negotiation_points = [
            c["negotiation_point"]
            for c in scored_clauses
            if c.get("negotiation_point") and c.get("risk_level") in HIGH_RISK_LEVELS
        ]

        document_type = self._infer_document_type(scored_clauses)

        partial_report = {
            "contract_id": contract_id,
            "file_name": extraction["file_name"],
            "analysis_timestamp": start_time.isoformat(),
            "document_type": document_type,
            "extraction_method": extraction["extraction_method"],
            "pages": extraction["pages"],
            "total_clauses": len(scored_clauses),
            "overall_risk_score": round(overall_risk_score, 4),
            "risk_distribution": dict(risk_distribution),
            "critical_clauses": critical_clauses,
            "contradictions": contradictions,
            "missing_clauses": missing_clauses,
            "plain_english_summary": "",
            "negotiation_points": negotiation_points,
            "comparative_scores": {},
            "all_clauses": scored_clauses,
        }

        # ── Agent 6: Summarization + Memory ─────────────────────────────
        logger.info(f"[{contract_id}] Agent 6: Summarization + Memory")
        final_report = self.summarizer_agent.summarize(partial_report)

        # Store all clauses in ChromaDB memory
        self.summarizer_agent.store_clauses(scored_clauses, contract_id)

        # Comparative aggressiveness for HIGH/CRITICAL clauses
        comparative_scores = {}
        for clause in critical_clauses[:10] if comparative else []:  # Skipped if comparative=False
            label = clause.get("label", "Other")
            text = clause.get("clause_text", "")
            cid = clause.get("clause_id")
            comp = self.summarizer_agent.compare_aggressiveness(text, label)
            comparative_scores[cid] = comp

        final_report["comparative_scores"] = comparative_scores

        # Strip negotiation points if user toggled off
        if not negotiate:
            for c in final_report.get("all_clauses", []):
                c["negotiation_point"] = ""
            for c in final_report.get("critical_clauses", []):
                c["negotiation_point"] = ""
            final_report["negotiation_points"] = []

        # Attach settings to report for UI reference
        final_report["analysis_settings"] = settings

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"[{contract_id}] Analysis complete in {elapsed:.1f}s. "
            f"Overall risk: {overall_risk_score:.3f}"
        )

        return final_report

    @staticmethod
    def _infer_document_type(clauses: List[Dict]) -> str:
        """Infer the document type from the most common non-generic clause label."""
        label_counts = Counter(
            c.get("label", "Other")
            for c in clauses
            if c.get("label") not in ("Other", None)
        )
        if not label_counts:
            return "General Contract"

        top_label = label_counts.most_common(1)[0][0]

        label_to_doc_type = {
            "Non-Compete": "Employment Agreement",
            "IP Assignment": "Employment / IP Agreement",
            "Confidentiality/NDA": "Non-Disclosure Agreement",
            "Payment Terms": "Service Agreement",
            "License Grant": "License Agreement",
            "Revenue Share": "Partnership Agreement",
            "Data Privacy": "Data Processing Agreement",
            "Indemnification": "Commercial Agreement",
        }
        return label_to_doc_type.get(top_label, "Commercial Contract")
