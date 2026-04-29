"""
Agent 5: Contradiction Detector
Uses NetworkX dependency graph to find cross-clause conflicts.
"""
import re
import uuid
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    logger.warning("NetworkX not installed. Contradiction detection will be limited.")
    NX_AVAILABLE = False

# IP ownership keywords
EMPLOYER_OWNS_TERMS = [
    "employer owns", "company owns", "belongs to employer", "belongs to company",
    "vests in the company", "assigned to employer", "property of employer",
    "property of the company", "owned by the company",
]
EMPLOYEE_OWNS_TERMS = [
    "employee owns", "employee retains", "employee's property",
    "owned by employee", "retained by employee", "belong to employee",
]
ALL_IP_TERMS = ["all ip", "all intellectual property", "all inventions", "all work product"]
CARVEOUT_TERMS = [
    "carveout", "carve-out", "except", "excluding", "prior inventions",
    "pre-existing", "personal inventions",
]

# Termination trigger patterns
NOTICE_REQUIRED_TERMS = ["days notice", "written notice", "notice period", "upon notice"]
IMMEDIATE_TERMS = ["immediate", "immediately", "without notice", "forthwith"]


class ContradictionAgent:
    """Detects logical contradictions between clauses using a dependency graph."""

    def detect(
        self, classified_clauses: List[Dict], reference_map: Optional[Dict] = None
    ) -> List[Dict]:
        if reference_map is None:
            reference_map = {c["clause_id"]: c.get("references", []) for c in classified_clauses}

        contradictions = []

        # Build the dependency graph
        graph = self._build_graph(classified_clauses, reference_map)

        # Run all contradiction checks
        contradictions.extend(self._check_ip_ownership_conflict(classified_clauses))
        contradictions.extend(self._check_notice_period_conflict(classified_clauses))
        contradictions.extend(self._check_termination_trigger_conflict(classified_clauses))
        contradictions.extend(self._check_definition_conflict(classified_clauses))
        contradictions.extend(self._check_obligation_without_right(classified_clauses, graph))

        logger.info(f"Contradiction detection found {len(contradictions)} issues.")
        return contradictions

    # ------------------------------------------------------------------
    # Graph builder
    # ------------------------------------------------------------------

    def _build_graph(self, clauses: List[Dict], reference_map: Dict):
        if not NX_AVAILABLE:
            return None

        G = nx.DiGraph()
        for clause in clauses:
            cid = clause["clause_id"]
            G.add_node(cid, **{k: v for k, v in clause.items() if isinstance(v, (str, int, float, bool))})

        for clause_id, refs in reference_map.items():
            for ref in refs:
                if G.has_node(ref):
                    G.add_edge(clause_id, ref, relationship="references")

        return G

    # ------------------------------------------------------------------
    # Check 1: IP ownership conflict
    # ------------------------------------------------------------------

    def _check_ip_ownership_conflict(self, clauses: List[Dict]) -> List[Dict]:
        results = []
        ip_clauses = [
            c for c in clauses
            if c.get("label") in ("IP Assignment", "License Grant", "Affiliate IP")
        ]

        employer_owns_clauses = []
        employee_owns_clauses = []
        all_ip_clauses = []
        carveout_clauses = []

        for c in ip_clauses:
            text_lower = c["clause_text"].lower()
            if any(t in text_lower for t in EMPLOYER_OWNS_TERMS):
                employer_owns_clauses.append(c)
            if any(t in text_lower for t in EMPLOYEE_OWNS_TERMS):
                employee_owns_clauses.append(c)
            if any(t in text_lower for t in ALL_IP_TERMS):
                all_ip_clauses.append(c)
            if any(t in text_lower for t in CARVEOUT_TERMS):
                carveout_clauses.append(c)

        # Conflict: employer owns AND employee owns
        if employer_owns_clauses and employee_owns_clauses:
            for ea in employer_owns_clauses:
                for ee in employee_owns_clauses:
                    if ea["clause_id"] != ee["clause_id"]:
                        results.append(self._make_contradiction(
                            ea, ee,
                            "IP Ownership Conflict",
                            "One clause assigns IP to employer while another states employee retains ownership.",
                            "HIGH",
                        ))

        # Conflict: "all IP" assigned to employer but another clause has carveout
        if all_ip_clauses and carveout_clauses:
            for ai in all_ip_clauses:
                for co in carveout_clauses:
                    if ai["clause_id"] != co["clause_id"]:
                        results.append(self._make_contradiction(
                            ai, co,
                            "IP Scope Conflict",
                            "'All IP' assignment conflicts with a carveout clause elsewhere in the contract.",
                            "MEDIUM",
                        ))

        return results

    # ------------------------------------------------------------------
    # Check 2: Notice period conflict
    # ------------------------------------------------------------------

    def _check_notice_period_conflict(self, clauses: List[Dict]) -> List[Dict]:
        results = []
        notice_clauses = [
            c for c in clauses
            if c.get("label") in ("Termination", "Notice Period")
            and any(t in c["clause_text"].lower() for t in NOTICE_REQUIRED_TERMS)
        ]

        if len(notice_clauses) < 2:
            return results

        period_pattern = re.compile(r'(\d+)\s*(day|week|month)s?', re.IGNORECASE)

        # Extract all notice periods per clause
        clause_periods = []
        for c in notice_clauses:
            matches = period_pattern.findall(c["clause_text"])
            if matches:
                days = [self._to_days(int(m[0]), m[1]) for m in matches]
                clause_periods.append((c, days))

        # Compare all pairs — if periods differ for same party context, flag conflict
        for i in range(len(clause_periods)):
            for j in range(i + 1, len(clause_periods)):
                ca, days_a = clause_periods[i]
                cb, days_b = clause_periods[j]
                # If the sets of days are completely non-overlapping, likely conflict
                if set(days_a).isdisjoint(set(days_b)) and ca["clause_id"] != cb["clause_id"]:
                    results.append(self._make_contradiction(
                        ca, cb,
                        "Notice Period Conflict",
                        f"Clause {ca['clause_id']} specifies {days_a} days but "
                        f"clause {cb['clause_id']} specifies different notice periods.",
                        "MEDIUM",
                    ))

        return results

    # ------------------------------------------------------------------
    # Check 3: Termination trigger conflict
    # ------------------------------------------------------------------

    def _check_termination_trigger_conflict(self, clauses: List[Dict]) -> List[Dict]:
        results = []
        term_clauses = [c for c in clauses if c.get("label") == "Termination"]

        notice_required = [
            c for c in term_clauses
            if any(t in c["clause_text"].lower() for t in NOTICE_REQUIRED_TERMS)
        ]
        immediate_term = [
            c for c in term_clauses
            if any(t in c["clause_text"].lower() for t in IMMEDIATE_TERMS)
        ]

        if notice_required and immediate_term:
            for nr in notice_required:
                for it in immediate_term:
                    if nr["clause_id"] != it["clause_id"]:
                        results.append(self._make_contradiction(
                            nr, it,
                            "Termination Trigger Conflict",
                            "One clause requires notice before termination; another permits "
                            "immediate termination — conditions not clearly distinguished.",
                            "HIGH",
                        ))

        return results

    # ------------------------------------------------------------------
    # Check 4: Definition conflict
    # ------------------------------------------------------------------

    def _check_definition_conflict(self, clauses: List[Dict]) -> List[Dict]:
        results = []
        # Aggregate all definitions across clauses
        term_definitions: Dict[str, List[Dict]] = {}

        for c in clauses:
            for term, definition in c.get("definitions", {}).items():
                term_normalized = term.strip().lower()
                if term_normalized not in term_definitions:
                    term_definitions[term_normalized] = []
                term_definitions[term_normalized].append(
                    {"clause": c, "definition": definition}
                )

        for term, entries in term_definitions.items():
            if len(entries) < 2:
                continue
            # Check if any two definitions differ substantially
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    def_a = entries[i]["definition"].strip().lower()
                    def_b = entries[j]["definition"].strip().lower()
                    # Similarity heuristic: if they share < 30% of words, conflict
                    similarity = self._word_similarity(def_a, def_b)
                    if similarity < 0.3:
                        ca = entries[i]["clause"]
                        cb = entries[j]["clause"]
                        results.append(self._make_contradiction(
                            ca, cb,
                            "Definition Conflict",
                            f'The term "{term}" is defined differently in two clauses.',
                            "MEDIUM",
                        ))

        return results

    # ------------------------------------------------------------------
    # Check 5: Obligation without right
    # ------------------------------------------------------------------

    def _check_obligation_without_right(
        self, clauses: List[Dict], graph
    ) -> List[Dict]:
        results = []
        shall_provide_pattern = re.compile(
            r'(?:party\s+[AB]|employer|company|employee|vendor|client)\s+shall\s+provide',
            re.IGNORECASE,
        )
        shall_receive_pattern = re.compile(
            r'(?:party\s+[AB]|employer|company|employee|vendor|client)\s+shall\s+(?:receive|accept|acknowledge)',
            re.IGNORECASE,
        )

        for c in clauses:
            text = c["clause_text"]
            has_shall_provide = shall_provide_pattern.search(text)
            if not has_shall_provide:
                continue

            # Check if connected clauses (via reference graph) have a shall_receive
            connected_clause_ids = []
            if graph and NX_AVAILABLE:
                try:
                    successors = list(graph.successors(c["clause_id"]))
                    predecessors = list(graph.predecessors(c["clause_id"]))
                    connected_clause_ids = successors + predecessors
                except Exception:
                    pass

            connected_clauses = [
                other for other in clauses
                if other["clause_id"] in connected_clause_ids
            ]
            # Also check globally referenced clauses
            for ref_id in c.get("references", []):
                connected_clauses += [oc for oc in clauses if oc["clause_id"] == ref_id]

            has_corresponding_receive = any(
                shall_receive_pattern.search(oc["clause_text"])
                for oc in connected_clauses
            )

            if not has_corresponding_receive and connected_clauses:
                # Only flag if there are connected clauses but none reciprocate
                results.append(
                    {
                        "contradiction_id": f"CONTR-{uuid.uuid4().hex[:8].upper()}",
                        "clause_a_id": c["clause_id"],
                        "clause_b_id": "N/A",
                        "clause_a_text": c["clause_text"][:200],
                        "clause_b_text": "No corresponding receive/acknowledgment clause found.",
                        "contradiction_type": "Obligation Without Corresponding Right",
                        "description": "This clause imposes an obligation to provide/deliver but no "
                        "connected clause imposes a corresponding obligation to accept or "
                        "acknowledge, leaving enforcement ambiguous.",
                        "severity": "LOW",
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_contradiction(
        clause_a: Dict,
        clause_b: Dict,
        contradiction_type: str,
        description: str,
        severity: str,
    ) -> Dict:
        return {
            "contradiction_id": f"CONTR-{uuid.uuid4().hex[:8].upper()}",
            "clause_a_id": clause_a["clause_id"],
            "clause_b_id": clause_b["clause_id"],
            "clause_a_text": clause_a["clause_text"][:200],
            "clause_b_text": clause_b["clause_text"][:200],
            "contradiction_type": contradiction_type,
            "description": description,
            "severity": severity,
        }

    @staticmethod
    def _word_similarity(text_a: str, text_b: str) -> float:
        """Jaccard similarity on word sets."""
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        if not words_a and not words_b:
            return 1.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _to_days(num: int, unit: str) -> int:
        unit_lower = unit.lower()
        if "year" in unit_lower:
            return num * 365
        if "month" in unit_lower:
            return num * 30
        if "week" in unit_lower:
            return num * 7
        return num
