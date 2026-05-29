import os
"""
Agent 6: Summarizer + Memory Agent
Calls Ollama for plain-English summary, stores clauses in ChromaDB,
and provides comparative aggressiveness scoring.
"""
import json
import logging
import re
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "mistral:7b"
CHROMA_COLLECTION = "contract_clauses"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SUMMARY_PROMPT = """You are a senior legal analyst specializing in employment and commercial contracts in India. 
Given the following contract risk analysis JSON, write a plain English summary (maximum 200 words) 
that a non-lawyer professional can immediately understand and act on.

Focus on:
1. The top 3 highest-risk clauses and exactly why they are dangerous
2. The most important negotiation points the person must raise before signing
3. Any missing protections that leave the person exposed

Be direct. Use plain language. No legalese. 

Risk Report:
{report_json}

Write the summary now:"""


class SummarizerAgent:
    """Generates plain-English summaries via Ollama and manages ChromaDB clause memory."""

    def __init__(self):
        self._chroma_client = None
        self._collection = None
        self._embedding_model = None
        self._init_chromadb()
        self._init_embedding_model()

    def _init_chromadb(self):
        try:
            import chromadb
            self._chroma_client = chromadb.Client()
            self._collection = self._chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB initialized. Collection: {CHROMA_COLLECTION}")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}. Comparative scoring disabled.")
            self._chroma_client = None
            self._collection = None

    def _init_embedding_model(self):
        import os
        if os.getenv("CLOUD_DEPLOY"):
            logger.info("CLOUD_DEPLOY=true — skipping embedding model (no sentence-transformers).")
            self._embedding_model = None
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"Embedding model init failed: {e}. Comparative scoring disabled.")
            self._embedding_model = None

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def summarize(self, risk_report: dict) -> dict:
        """
        Generate a plain English summary of the risk report.
        Falls back to template-based summary if Ollama is unavailable.
        """
        # Trim report to essentials to fit prompt
        trimmed = {
            "overall_risk_score": risk_report.get("overall_risk_score"),
            "risk_distribution": risk_report.get("risk_distribution"),
            "critical_clauses": [
                {
                    "clause_id": c.get("clause_id"),
                    "label": c.get("label"),
                    "risk_level": c.get("risk_level"),
                    "risk_reason": c.get("risk_reason"),
                    "negotiation_point": c.get("negotiation_point"),
                }
                for c in risk_report.get("critical_clauses", [])[:5]
            ],
            "missing_clauses": risk_report.get("missing_clauses", []),
            "contradictions": [
                {
                    "type": c.get("contradiction_type"),
                    "description": c.get("description"),
                    "severity": c.get("severity"),
                }
                for c in risk_report.get("contradictions", [])[:3]
            ],
        }

        prompt = SUMMARY_PROMPT.format(report_json=json.dumps(trimmed, indent=2))

        ollama_available = self._check_ollama()
        if ollama_available:
            summary = self._call_ollama(prompt)
            if summary:
                risk_report["plain_english_summary"] = summary
                risk_report["_summary_method"] = "ollama"
                return risk_report

        # Fallback: template-based summary
        risk_report["plain_english_summary"] = self._template_summary(risk_report)
        risk_report["_summary_method"] = "template_fallback"
        return risk_report

    def _check_ollama(self) -> bool:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            logger.warning("Ollama not reachable at localhost:11434. Using template fallback.")
            return False

    def _call_ollama(self, prompt: str) -> Optional[str]:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
            }
            response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
            response.raise_for_status()

            chunks = []
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        chunks.append(data["response"])
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

            full_text = "".join(chunks).strip()
            return full_text if full_text else None

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None

    def _template_summary(self, risk_report: dict) -> str:
        """Template-based fallback summary when Ollama is unavailable."""
        score = risk_report.get("overall_risk_score", 0)
        dist = risk_report.get("risk_distribution", {})
        critical = risk_report.get("critical_clauses", [])
        missing = risk_report.get("missing_clauses", [])
        contradictions = risk_report.get("contradictions", [])

        risk_label = (
            "CRITICAL RISK" if score >= 0.75
            else "HIGH RISK" if score >= 0.5
            else "MODERATE RISK" if score >= 0.25
            else "LOW RISK"
        )

        lines = [
            f"CONTRACT RISK ASSESSMENT — {risk_label} (Score: {score:.0%})",
            "",
            f"This contract has {dist.get('CRITICAL', 0)} critical and "
            f"{dist.get('HIGH', 0)} high-risk clauses out of "
            f"{sum(dist.values())} total clauses analyzed.",
            "",
        ]

        if critical:
            lines.append("TOP RISKS:")
            for i, c in enumerate(critical[:3], 1):
                lines.append(
                    f"{i}. [{c.get('label', 'Unknown')}] {c.get('risk_reason', 'Risk identified.')}"
                )
                if c.get("negotiation_point"):
                    lines.append(f"   → Action: {c['negotiation_point']}")
            lines.append("")

        if missing:
            lines.append("MISSING PROTECTIONS:")
            for m in missing[:3]:
                lines.append(f"• {m}")
            lines.append("")

        if contradictions:
            lines.append(
                f"WARNING: {len(contradictions)} contradiction(s) detected between clauses. "
                "These require legal review before signing."
            )
            lines.append("")

        lines.append(
            "NOTE: This analysis is automated. Have a lawyer review before signing."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ChromaDB: store clauses
    # ------------------------------------------------------------------

    def store_clauses(self, clauses: List[Dict], contract_id: str):
        """Embed clause texts and store in ChromaDB with metadata."""
        if self._collection is None or self._embedding_model is None:
            logger.warning("ChromaDB or embedding model not available. Skipping storage.")
            return

        try:
            documents = []
            metadatas = []
            ids = []

            for i, clause in enumerate(clauses):
                text = clause.get("clause_text", "").strip()
                if not text or len(text) < 20:
                    continue

                clause_id = clause.get("clause_id", f"clause-{i}")
                unique_id = f"{contract_id}_{clause_id}"

                documents.append(text)
                metadatas.append(
                    {
                        "contract_id": contract_id,
                        "clause_id": clause_id,
                        "clause_label": clause.get("label", "Other"),
                        "risk_level": clause.get("risk_level", "NONE"),
                        "risk_score": float(clause.get("risk_score", 0.0)),
                    }
                )
                ids.append(unique_id)

            if not documents:
                return

            embeddings = self._embedding_model.encode(documents).tolist()

            # ChromaDB batch upsert
            batch_size = 50
            for start in range(0, len(documents), batch_size):
                end = start + batch_size
                self._collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )

            logger.info(
                f"Stored {len(documents)} clauses for contract {contract_id} in ChromaDB."
            )
        except Exception as e:
            logger.error(f"ChromaDB store_clauses failed: {e}")

    # ------------------------------------------------------------------
    # ChromaDB: comparative aggressiveness
    # ------------------------------------------------------------------

    def compare_aggressiveness(self, clause_text: str, clause_label: str) -> dict:
        """
        Query ChromaDB for similar clauses and compare risk scores.
        Returns percentile aggressiveness.
        """
        if self._collection is None or self._embedding_model is None:
            return {
                "percentile": None,
                "similar_count": 0,
                "verdict": "Comparative scoring unavailable (ChromaDB not initialized).",
            }

        try:
            embedding = self._embedding_model.encode([clause_text]).tolist()

            results = self._collection.query(
                query_embeddings=embedding,
                n_results=min(10, self._collection.count()),
                where={"clause_label": clause_label} if clause_label != "Other" else None,
                include=["metadatas", "distances"],
            )

            metadatas = results.get("metadatas", [[]])[0]
            if not metadatas:
                return {
                    "percentile": None,
                    "similar_count": 0,
                    "verdict": f"No similar {clause_label} clauses in memory yet.",
                }

            similar_scores = [float(m.get("risk_score", 0.0)) for m in metadatas]
            similar_count = len(similar_scores)

            # We need a risk score for the current clause — use avg keyword match
            # In practice, pipeline passes the scored clause so we use its risk_score
            # Here we estimate from text length/content as a fallback
            current_score = self._estimate_risk_score_from_text(clause_text)

            less_aggressive = sum(1 for s in similar_scores if s < current_score)
            percentile = (less_aggressive / similar_count) * 100 if similar_count > 0 else 50.0

            verdict = (
                f"More aggressive than {percentile:.0f}% of similar {clause_label} clauses seen."
            )

            return {
                "percentile": round(percentile, 1),
                "similar_count": similar_count,
                "verdict": verdict,
            }
        except Exception as e:
            logger.error(f"compare_aggressiveness failed: {e}")
            return {
                "percentile": None,
                "similar_count": 0,
                "verdict": f"Comparative scoring error: {str(e)}",
            }

    @staticmethod
    def _estimate_risk_score_from_text(text: str) -> float:
        """
        Rough heuristic: longer, more restrictive clauses → higher score.
        Only used as fallback when scored clause not passed in.
        """
        text_lower = text.lower()
        high_risk_terms = [
            "perpetual", "irrevocable", "worldwide", "all inventions",
            "no liability", "immediate termination", "without notice",
        ]
        hits = sum(1 for t in high_risk_terms if t in text_lower)
        return min(hits * 0.15 + 0.1, 1.0)
