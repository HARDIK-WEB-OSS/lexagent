"""
Agent 4: Risk Scoring Agent
Pure rule-based, deterministic risk engine. No ML. Fully auditable.
"""
import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

RISK_SCORES = {
    "CRITICAL": 1.0,
    "HIGH": 0.75,
    "MEDIUM": 0.5,
    "LOW": 0.25,
    "NONE": 0.0,
}

# Duration extraction pattern: captures number + unit
DURATION_PATTERN = re.compile(
    r'(\d+)\s*(year|month|week|day)s?',
    re.IGNORECASE,
)

# Geographic scope keywords
GLOBAL_SCOPE_TERMS = [
    "worldwide", "global", "international", "across the world",
    "all jurisdictions", "any country", "all countries",
]

INDIA_SCOPE_TERMS = ["india", "indian territory", "republic of india"]

LOCAL_SCOPE_TERMS = [
    "city", "state", "district", "province", "county", "region",
    "local", "municipal",
]

PERPETUAL_TERMS = [
    "perpetual", "indefinite", "forever", "in perpetuity",
    "without expiration", "permanent", "never expires",
]

NO_LIABILITY_TERMS = [
    "no liability", "zero liability", "shall not be liable",
    "not be responsible", "disclaims all liability",
    "excluded from liability",
]

MUTUAL_CAP_TERMS = ["mutual", "both parties", "either party"]

FOREIGN_JURISDICTIONS = [
    "england", "wales", "united kingdom", "uk", "usa", "united states",
    "new york", "delaware", "california", "singapore", "dubai",
    "hong kong", "cayman islands", "british virgin islands",
    "mauritius", "netherlands", "luxembourg",
]


class RiskAgent:
    """Deterministic rule-based risk scoring for legal clauses."""

    def score(self, classified_clauses: List[Dict], jurisdiction: str = "IN") -> Tuple[List[Dict], List[str]]:
        """
        Returns (scored_clauses, missing_clauses_list).
        """
        scored = []
        found_labels = set()

        for clause in classified_clauses:
            label = clause.get("label", "Other")
            text = clause.get("clause_text", "")
            found_labels.add(label)

            risk_level, risk_score, risk_reason, negotiation_point = self._score_clause(
                label, text
            )

            clause["risk_level"] = risk_level
            clause["risk_score"] = risk_score
            clause["risk_reason"] = risk_reason
            clause["negotiation_point"] = negotiation_point
            scored.append(clause)

        missing = self._check_missing_clauses(found_labels, scored)

        # Compute overall document risk score (weighted average)
        if scored:
            overall = sum(c["risk_score"] for c in scored) / len(scored)
        else:
            overall = 0.0

        # Attach overall score to each clause for pipeline use
        for c in scored:
            c["_overall_risk_score"] = round(overall, 4)

        logger.info(
            f"Scored {len(scored)} clauses. Overall risk: {overall:.3f}. "
            f"Missing clauses: {missing}"
        )
        return scored, missing

    # ------------------------------------------------------------------
    # Master dispatcher
    # ------------------------------------------------------------------

    def _score_clause(
        self, label: str, text: str
    ) -> Tuple[str, float, str, str]:
        """Dispatch to specific rule set based on clause label."""
        self._jurisdiction = jurisdiction
        dispatch = {
            "Non-Compete": self._score_non_compete,
            "IP Assignment": self._score_ip_assignment,
            "Termination": self._score_termination,
            "Confidentiality/NDA": self._score_confidentiality,
            "Liability Limitation": self._score_liability,
            "Cap on Liability": self._score_liability,
            "Uncapped Liability": self._score_uncapped_liability,
            "Governing Law": self._score_governing_law,
            "Jurisdiction": self._score_governing_law,
        }
        scorer = dispatch.get(label)
        if scorer:
            return scorer(text)
        return "NONE", 0.0, "No specific risk rules for this clause type.", ""

    # ------------------------------------------------------------------
    # Non-Compete
    # ------------------------------------------------------------------

    def _score_non_compete(self, text: str) -> Tuple[str, float, str, str]:
        text_lower = text.lower()
        duration_months = self._extract_max_duration_months(text)
        is_global = any(t in text_lower for t in GLOBAL_SCOPE_TERMS)
        is_india = any(t in text_lower for t in INDIA_SCOPE_TERMS)
        is_local = any(t in text_lower for t in LOCAL_SCOPE_TERMS)

        if duration_months > 12 or (is_global or is_india):
            reason = (
                f"Non-compete duration is {duration_months} months"
                if duration_months
                else "Non-compete has broad geographic scope"
            )
            if is_global:
                reason += " with worldwide/global scope"
            elif is_india:
                reason += " covering all of India"
            return (
                "HIGH",
                RISK_SCORES["HIGH"],
                reason,
                "Negotiate scope down to your city/state and cap duration at 6 months or less.",
            )

        if 6 <= duration_months <= 12:
            return (
                "MEDIUM",
                RISK_SCORES["MEDIUM"],
                f"Non-compete duration is {duration_months} months — acceptable but watch scope.",
                "Push for city-level geographic restriction and confirm exact activities excluded.",
            )

        if duration_months > 0 and duration_months < 6 and is_local:
            return (
                "LOW",
                RISK_SCORES["LOW"],
                f"Non-compete is {duration_months} months, locally scoped.",
                "Confirm that 'local' is defined precisely (e.g., specific city name).",
            )

        return (
            "MEDIUM",
            RISK_SCORES["MEDIUM"],
            "Non-compete clause present but duration/scope is unclear.",
            "Demand explicit duration and geographic scope before signing.",
        )

    # ------------------------------------------------------------------
    # IP Assignment
    # ------------------------------------------------------------------

    def _score_ip_assignment(self, text: str) -> Tuple[str, float, str, str]:
        text_lower = text.lower()
        has_all_work_product = "all work product" in text_lower or "all work-product" in text_lower
        has_personal_time = any(
            t in text_lower
            for t in ["personal time", "outside working hours", "personal equipment",
                      "on my own time", "outside of work", "outside business hours"]
        )
        has_all_inventions = "all inventions" in text_lower
        has_preexisting_carveout = any(
            t in text_lower
            for t in ["pre-existing", "preexisting", "prior inventions",
                      "prior ip", "schedule a", "exhibit a", "carve-out", "carveout"]
        )
        is_work_related = any(
            t in text_lower
            for t in ["related to the business", "within scope of employment",
                      "in the course of employment", "job duties"]
        )

        if has_all_work_product and has_personal_time:
            return (
                "CRITICAL",
                RISK_SCORES["CRITICAL"],
                "IP assignment claims ALL work product INCLUDING work done on personal time "
                "with personal equipment. This is an extreme overreach.",
                "REJECT this clause outright. Demand carveout for inventions made on personal "
                "time unrelated to employer's business. Add a Schedule A listing your prior IP.",
            )

        if has_all_inventions and not has_preexisting_carveout:
            return (
                "HIGH",
                RISK_SCORES["HIGH"],
                "Assignment covers 'all inventions' with no carveout for pre-existing IP.",
                "Insist on a Schedule A listing your prior inventions and IP. Ensure "
                "assignment is limited to inventions created using company resources or "
                "related to company business.",
            )

        if is_work_related or has_preexisting_carveout:
            return (
                "MEDIUM",
                RISK_SCORES["MEDIUM"],
                "IP assignment limited to work-related inventions or includes prior-IP carveout.",
                "Verify the carveout is specific enough. 'Related to business' can be interpreted "
                "broadly. Get specific list of excluded inventions appended.",
            )

        return (
            "MEDIUM",
            RISK_SCORES["MEDIUM"],
            "IP assignment clause present with unclear scope.",
            "Clarify scope boundaries and add explicit carveout for personal/pre-existing IP.",
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _score_termination(self, text: str) -> Tuple[str, float, str, str]:
        text_lower = text.lower()
        employer_days, employee_days = self._extract_notice_periods(text)

        if employer_days is not None and employee_days is not None:
            ratio = employee_days / employer_days if employer_days > 0 else float("inf")
            if employer_days < 15 and employee_days > 60:
                return (
                    "HIGH",
                    RISK_SCORES["HIGH"],
                    f"Asymmetric notice: employer needs only {employer_days} days but employee "
                    f"must give {employee_days} days notice.",
                    "Demand notice parity: both parties should have equal notice periods. "
                    "Push for minimum 30 days mutual notice.",
                )
            if ratio >= 3 and employer_days != employee_days:
                # Employee notice is 3x+ employer notice — still HIGH asymmetry
                return (
                    "HIGH",
                    RISK_SCORES["HIGH"],
                    f"Severe asymmetry: employer notice is {employer_days} days, "
                    f"employee must give {employee_days} days — a {ratio:.0f}x disparity.",
                    "Demand equal notice periods. This asymmetry is one-sided and unfair.",
                )
            if employer_days != employee_days:
                return (
                    "MEDIUM",
                    RISK_SCORES["MEDIUM"],
                    f"Notice periods differ: employer {employer_days} days vs employee "
                    f"{employee_days} days.",
                    "Negotiate equal notice periods for both parties.",
                )

        if "immediate termination" in text_lower or "without notice" in text_lower:
            return (
                "HIGH",
                RISK_SCORES["HIGH"],
                "Clause allows immediate termination without notice.",
                "Insist on minimum notice period even for termination for cause. "
                "Ensure termination triggers are explicitly and narrowly defined.",
            )

        if employer_days is not None and employer_days < 15:
            return (
                "MEDIUM",
                RISK_SCORES["MEDIUM"],
                f"Employer notice period is very short ({employer_days} days).",
                "Negotiate for at least 30 days employer notice.",
            )

        return (
            "LOW",
            RISK_SCORES["LOW"],
            "Termination clause with notice period — appears standard.",
            "Verify that termination triggers are explicitly listed and notice periods are mutual.",
        )

    # ------------------------------------------------------------------
    # Confidentiality
    # ------------------------------------------------------------------

    def _score_confidentiality(self, text: str) -> Tuple[str, float, str, str]:
        text_lower = text.lower()
        is_perpetual = any(t in text_lower for t in PERPETUAL_TERMS)
        has_public_carveout = any(
            t in text_lower
            for t in ["publicly available", "public domain", "publicly known",
                      "general knowledge", "in the public"]
        )
        duration_months = self._extract_max_duration_months(text)

        if is_perpetual:
            return (
                "HIGH",
                RISK_SCORES["HIGH"],
                "Confidentiality obligation is perpetual/indefinite — no expiration.",
                "Negotiate a fixed term (2–3 years post-termination). Perpetual obligations "
                "are unenforceable in many jurisdictions and place unreasonable burden on you.",
            )

        if not has_public_carveout:
            return (
                "HIGH",
                RISK_SCORES["HIGH"],
                "No carveout for publicly available information — you could be liable for "
                "disclosing information that is already public.",
                "Demand standard carveouts: (i) publicly available information, "
                "(ii) independently developed, (iii) received from third party, "
                "(iv) required to be disclosed by law.",
            )

        if duration_months > 60:
            return (
                "MEDIUM",
                RISK_SCORES["MEDIUM"],
                f"Confidentiality period is {duration_months} months — unusually long.",
                f"Negotiate down to 24–36 months post-termination.",
            )

        return (
            "LOW",
            RISK_SCORES["LOW"],
            "Confidentiality clause with reasonable duration and carveouts.",
            "Ensure carveout for information required to be disclosed by court order.",
        )

    # ------------------------------------------------------------------
    # Liability
    # ------------------------------------------------------------------

    def _score_liability(self, text: str) -> Tuple[str, float, str, str]:
        text_lower = text.lower()
        has_no_liability = any(t in text_lower for t in NO_LIABILITY_TERMS)
        is_mutual = any(t in text_lower for t in MUTUAL_CAP_TERMS)
        cap_months = self._extract_cap_in_months(text)

        if has_no_liability or ("0" in text and "liability" in text_lower):
            return (
                "HIGH",
                RISK_SCORES["HIGH"],
                "Liability cap may be zero or employer disclaims all liability.",
                "Reject zero-cap clauses. Negotiate minimum cap of 12 months of fees/salary "
                "for direct damages. Ensure carveout for wilful misconduct and fraud.",
            )

        if cap_months is not None and cap_months < 3:
            return (
                "MEDIUM",
                RISK_SCORES["MEDIUM"],
                f"Liability cap is less than 3 months of fees — very low.",
                "Negotiate cap to at least 12 months of fees. Ensure gross negligence and "
                "intentional misconduct are uncapped.",
            )

        if is_mutual:
            return (
                "LOW",
                RISK_SCORES["LOW"],
                "Mutual liability limitation — balanced between parties.",
                "Verify that exceptions to cap (IP infringement, data breach, fraud) are clearly listed.",
            )

        return (
            "MEDIUM",
            RISK_SCORES["MEDIUM"],
            "Liability limitation clause present — cap amount unclear.",
            "Ensure cap amount is explicitly stated and not one-sided.",
        )

    def _score_uncapped_liability(self, text: str) -> Tuple[str, float, str, str]:
        return (
            "CRITICAL",
            RISK_SCORES["CRITICAL"],
            "Clause explicitly states uncapped/unlimited liability.",
            "REJECT uncapped liability for ordinary breach. Negotiate hard cap. Only "
            "accept uncapped exposure for IP infringement, fraud, or death/personal injury.",
        )

    # ------------------------------------------------------------------
    # Governing Law
    # ------------------------------------------------------------------

    def _score_governing_law(self, text: str) -> Tuple[str, float, str, str]:
        text_lower = text.lower()
        jurisdiction = getattr(self, '_jurisdiction', 'IN')

        # Build list of "foreign" jurisdictions relative to user's selected jurisdiction
        home_terms = {
            "IN": ["india", "indian"],
            "US": ["united states", "usa", "u.s.a", "new york", "delaware", "california"],
            "UK": ["england", "wales", "united kingdom", "uk", "u.k"],
            "SG": ["singapore"],
        }
        home = home_terms.get(jurisdiction, home_terms["IN"])
        is_home = any(t in text_lower for t in home)
        is_foreign = any(j in text_lower for j in FOREIGN_JURISDICTIONS) and not is_home
        is_arbitration_abroad = "arbitration" in text_lower and is_foreign

        if is_arbitration_abroad:
            foreign_j = next((j for j in FOREIGN_JURISDICTIONS if j in text_lower and j not in home), "foreign jurisdiction")
            return (
                "HIGH",
                RISK_SCORES["HIGH"],
                f"Arbitration seat is in a foreign jurisdiction ({foreign_j}), making dispute resolution expensive and impractical.",
                f"Negotiate arbitration seat to {jurisdiction} home jurisdiction.",
            )

        if is_foreign:
            foreign_j = next((j for j in FOREIGN_JURISDICTIONS if j in text_lower and j not in home), "foreign jurisdiction")
            return (
                "MEDIUM",
                RISK_SCORES["MEDIUM"],
                f"Governing law is a foreign jurisdiction ({foreign_j}).",
                f"Negotiate governing law to {jurisdiction} jurisdiction.",
            )

        return ("NONE", RISK_SCORES["NONE"], "Governing law matches selected jurisdiction.", "")

    # ------------------------------------------------------------------
    # Missing clause detection
    # ------------------------------------------------------------------

    def _check_missing_clauses(
        self, found_labels: set, clauses: List[Dict]
    ) -> List[str]:
        missing = []

        if "Severance" not in found_labels:
            missing.append(
                "MISSING: Severance clause — no protection for compensation on termination without cause."
            )

        if "Dispute Resolution" not in found_labels:
            missing.append(
                "MISSING: Dispute Resolution clause — no agreed mechanism for resolving disagreements."
            )

        has_ip_assignment = "IP Assignment" in found_labels
        has_ip_reversion = any(
            "revert" in c.get("clause_text", "").lower()
            or "return" in c.get("clause_text", "").lower()
            for c in clauses
            if c.get("label") in ["IP Assignment", "Termination"]
        )
        if has_ip_assignment and not has_ip_reversion:
            missing.append(
                "MISSING: IP reversion on termination — no provision for IP created "
                "by you to revert to you upon termination."
            )

        if "Warranty" not in found_labels:
            missing.append(
                "MISSING: Warranty clause — no representations or warranties from either party."
            )

        return missing

    # ------------------------------------------------------------------
    # Utility: Duration extraction
    # ------------------------------------------------------------------

    def _extract_max_duration_months(self, text: str) -> int:
        """Extract the maximum duration mentioned in a clause, converted to months."""
        matches = DURATION_PATTERN.findall(text)
        max_months = 0
        for num_str, unit in matches:
            num = int(num_str)
            unit_lower = unit.lower()
            if "year" in unit_lower:
                months = num * 12
            elif "month" in unit_lower:
                months = num
            elif "week" in unit_lower:
                months = num // 4
            elif "day" in unit_lower:
                months = num // 30
            else:
                months = 0
            max_months = max(max_months, months)
        return max_months

    def _extract_notice_periods(self, text: str):
        """
        Extract employer and employee notice periods by parsing sentence-by-sentence.
        Returns (employer_days, employee_days) — either can be None.
        """
        employer_days = None
        employee_days = None

        employer_re = re.compile(r'\b(employer|company|organization|firm)\b', re.IGNORECASE)
        employee_re = re.compile(r'\b(employee|staff|worker|consultant|individual)\b', re.IGNORECASE)

        sentences = re.split(r'[.;\n]', text)
        for sentence in sentences:
            durations = DURATION_PATTERN.findall(sentence)
            if not durations:
                continue
            days = max(self._to_days(int(n), u) for n, u in durations)
            has_em = bool(employer_re.search(sentence))
            has_ee = bool(employee_re.search(sentence))

            if has_ee and not has_em:
                if employee_days is None:
                    employee_days = days
            elif has_em and not has_ee:
                if employer_days is None:
                    employer_days = days
            elif has_em and has_ee:
                em_pos = (employer_re.search(sentence) or re.search('$', sentence)).start()
                ee_pos = (employee_re.search(sentence) or re.search('$', sentence)).start()
                if em_pos < ee_pos:
                    if employer_days is None:
                        employer_days = days
                else:
                    if employee_days is None:
                        employee_days = days

        # Fallback: use raw duration order
        if employer_days is None and employee_days is None:
            all_d = DURATION_PATTERN.findall(text)
            if len(all_d) >= 2:
                all_days = [self._to_days(int(n), u) for n, u in all_d]
                employer_days = min(all_days)
                employee_days = max(all_days)
            elif len(all_d) == 1:
                d = self._to_days(int(all_d[0][0]), all_d[0][1])
                employer_days = d
                employee_days = d

        return employer_days, employee_days

    def _extract_cap_in_months(self, text: str) -> Optional[int]:
        """Extract liability cap expressed as N months of fees."""
        pattern = re.compile(
            r'(\d+)\s*(?:month|months)\s*(?:of\s+)?(?:fees|compensation|salary)',
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if m:
            return int(m.group(1))
        return None

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
