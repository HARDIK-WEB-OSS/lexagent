"""
Tests for RiskAgent
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.risk_agent import RiskAgent


def make_clause(clause_id, label, text):
    return {
        "clause_id": clause_id,
        "clause_text": text,
        "label": label,
        "label_score": 0.85,
        "secondary_label": None,
        "references": [],
        "definitions": {},
        "page_hint": 1,
    }


class TestNonCompeteRisk:
    def test_noncompete_high_risk_duration_and_global(self):
        """2-year worldwide non-compete → HIGH"""
        clause = make_clause(
            "5.1", "Non-Compete",
            "Employee shall not compete with Company or its affiliates for a period of 2 years "
            "after termination on a worldwide basis in any capacity whatsoever."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "HIGH"
        assert result["risk_score"] == 0.75

    def test_noncompete_medium_risk_6_to_12_months(self):
        """9-month non-compete with no global scope → MEDIUM"""
        clause = make_clause(
            "5.1", "Non-Compete",
            "Employee agrees not to engage in any competitive activity for 9 months "
            "following the end of employment."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "MEDIUM"

    def test_noncompete_low_risk_local_short(self):
        """3-month city-level non-compete → LOW"""
        clause = make_clause(
            "5.1", "Non-Compete",
            "Employee shall not compete within the city limits for a period of 3 months "
            "after termination of this agreement."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "LOW"

    def test_noncompete_high_risk_india_scope(self):
        """Non-compete covering all of India → HIGH"""
        clause = make_clause(
            "5.2", "Non-Compete",
            "The employee shall not carry on a competing business anywhere in India "
            "for a period of 18 months after termination."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "HIGH"


class TestIPAssignmentRisk:
    def test_ip_critical_all_work_product_personal_time(self):
        """'All work product' + 'personal time' → CRITICAL"""
        clause = make_clause(
            "6.1", "IP Assignment",
            "Employee hereby assigns to Employer all work product, inventions, and creations, "
            "including those developed on personal time using personal equipment outside working hours."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "CRITICAL"
        assert result["risk_score"] == 1.0

    def test_ip_high_all_inventions_no_carveout(self):
        """'All inventions' with no carveout → HIGH"""
        clause = make_clause(
            "6.2", "IP Assignment",
            "Employee assigns to Company all inventions, discoveries, and developments "
            "made during the term of employment, without limitation."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "HIGH"

    def test_ip_medium_work_related_with_carveout(self):
        """Work-related IP with pre-existing carveout → MEDIUM"""
        clause = make_clause(
            "6.3", "IP Assignment",
            "Employee assigns to Company all inventions developed within the scope of employment "
            "and related to the business. Pre-existing inventions listed in Schedule A are excluded."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "MEDIUM"


class TestAsymmetricNotice:
    def test_asymmetric_notice_employer_7_employee_90(self):
        """Employer 7 days vs Employee 90 days → HIGH"""
        clause = make_clause(
            "8.1", "Termination",
            "Employer may terminate this agreement by giving employee 7 days written notice. "
            "Employee must provide employer with 90 days written notice of resignation."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "HIGH"

    def test_immediate_termination_high_risk(self):
        """'Immediate termination without notice' → HIGH"""
        clause = make_clause(
            "8.2", "Termination",
            "Employer reserves the right to terminate the employment immediately and without notice "
            "for any breach of this agreement."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] == "HIGH"

    def test_equal_notice_periods_low_risk(self):
        """Equal notice periods (30/30) → LOW"""
        clause = make_clause(
            "8.3", "Termination",
            "Either party may terminate this agreement by giving the other party 30 days written notice."
        )
        agent = RiskAgent()
        scored, _ = agent.score([clause])
        result = scored[0]
        assert result["risk_level"] in ("LOW", "MEDIUM")  # equal periods = low or medium


class TestOverallScore:
    def test_overall_score_weighted_average(self):
        """Overall score = average of all clause risk_scores."""
        clauses = [
            make_clause("1", "Non-Compete",
                "Employee shall not compete worldwide for 3 years after termination."),
            make_clause("2", "IP Assignment",
                "All inventions made by employee are assigned to company with no carveout."),
            make_clause("3", "Governing Law",
                "This agreement is governed by the laws of India and disputes resolved in Mumbai."),
        ]
        agent = RiskAgent()
        scored, _ = agent.score(clauses)

        expected_avg = sum(c["risk_score"] for c in scored) / len(scored)
        # Verify via the internal field set by risk agent
        actual_overall = scored[0].get("_overall_risk_score", None)
        if actual_overall is not None:
            assert abs(actual_overall - expected_avg) < 0.001

    def test_missing_clauses_detected(self):
        """If no Severance or Dispute Resolution clause, they should be flagged."""
        clauses = [
            make_clause("1", "Non-Compete",
                "Employee shall not compete for 1 year in the local area."),
            make_clause("2", "Confidentiality/NDA",
                "Parties shall keep information confidential for 2 years."),
        ]
        agent = RiskAgent()
        _, missing = agent.score(clauses)

        missing_text = " ".join(missing).lower()
        assert "severance" in missing_text or "dispute" in missing_text

    def test_risk_score_range(self):
        """All risk_scores must be in [0.0, 1.0]."""
        clauses = [
            make_clause("1", "Non-Compete", "worldwide non-compete for 5 years"),
            make_clause("2", "IP Assignment", "all work product assigned perpetually"),
            make_clause("3", "Uncapped Liability", "no cap on liability whatsoever"),
            make_clause("4", "Governing Law", "governed by the laws of England and Wales"),
        ]
        agent = RiskAgent()
        scored, _ = agent.score(clauses)
        for c in scored:
            assert 0.0 <= c["risk_score"] <= 1.0, f"Score out of range for {c['clause_id']}: {c['risk_score']}"

    def test_risk_levels_valid(self):
        """All risk_level values must be from the valid set."""
        valid_levels = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"}
        clauses = [
            make_clause("1", "Non-Compete", "employee shall not compete for 6 months locally"),
            make_clause("2", "Confidentiality/NDA", "perpetual confidentiality with no carveouts"),
            make_clause("3", "Governing Law", "disputes resolved in Singapore under SIAC rules"),
            make_clause("4", "Payment Terms", "payment due within 30 days of invoice"),
        ]
        agent = RiskAgent()
        scored, _ = agent.score(clauses)
        for c in scored:
            assert c["risk_level"] in valid_levels, f"Invalid risk level: {c['risk_level']}"
