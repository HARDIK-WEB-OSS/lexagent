"""
Tests for ContradictionAgent
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.contradiction_agent import ContradictionAgent


def make_clause(clause_id, label, text, definitions=None, references=None):
    return {
        "clause_id": clause_id,
        "clause_text": text,
        "label": label,
        "label_score": 0.85,
        "secondary_label": None,
        "references": references or [],
        "definitions": definitions or {},
        "page_hint": 1,
        "risk_level": "NONE",
        "risk_score": 0.0,
        "risk_reason": "",
        "negotiation_point": "",
    }


class TestIPConflict:
    def test_ip_conflict_employer_vs_employee_ownership(self):
        """Clause A says 'company owns all IP', Clause B says 'employee owns' → contradiction."""
        clause_a = make_clause(
            "6.1", "IP Assignment",
            "All intellectual property created by the employee vests in the company and is "
            "owned by the company from the moment of creation. The company owns all work product.",
        )
        clause_b = make_clause(
            "6.2", "IP Assignment",
            "The employee retains ownership of all inventions and intellectual property. "
            "Any IP created by employee belongs to employee and is employee's property.",
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])

        assert len(contradictions) > 0
        types = [c["contradiction_type"] for c in contradictions]
        assert any("IP" in t for t in types)

    def test_ip_conflict_has_clause_ids(self):
        """Contradiction output must include valid clause IDs."""
        clause_a = make_clause(
            "IP-1", "IP Assignment",
            "The company owns all work product and all inventions made by employee.",
        )
        clause_b = make_clause(
            "IP-2", "IP Assignment",
            "Employee retains all IP and all inventions created by employee are employee's property.",
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])

        for c in contradictions:
            assert "clause_a_id" in c
            assert "clause_b_id" in c
            assert "severity" in c
            assert c["severity"] in ("HIGH", "MEDIUM", "LOW")

    def test_compatible_ip_clauses_no_false_positive(self):
        """Two clauses that don't conflict in IP ownership → no IP contradiction."""
        clause_a = make_clause(
            "6.1", "IP Assignment",
            "All inventions created by employee in the course of employment and related to the "
            "company's business are assigned to the company.",
        )
        clause_b = make_clause(
            "6.2", "IP Assignment",
            "Company acknowledges that inventions developed by employee on personal time using "
            "personal resources that are unrelated to company business are excluded from this assignment.",
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])

        ip_conflicts = [c for c in contradictions if "IP Ownership" in c.get("contradiction_type", "")]
        # These clauses are compatible (one is the assignment, one is the carveout)
        # The scope conflict check may fire but ownership conflict should not
        ownership_conflicts = [c for c in contradictions if c["contradiction_type"] == "IP Ownership Conflict"]
        assert len(ownership_conflicts) == 0


class TestDefinitionConflict:
    def test_definition_conflict_same_term_different_meaning(self):
        """Same term defined two completely different ways → contradiction detected."""
        clause_a = make_clause(
            "1.1", "Other",
            "For the purposes of this agreement, Confidential Information means any "
            "technical, business, or financial information disclosed by Company to Employee.",
            definitions={"Confidential Information": "any technical, business, or financial information disclosed by Company to Employee"},
        )
        clause_b = make_clause(
            "5.1", "Confidentiality/NDA",
            "Notwithstanding Section 1.1, Confidential Information in this clause refers solely "
            "to customer names and pricing data as communicated in writing.",
            definitions={"Confidential Information": "solely customer names and pricing data as communicated in writing"},
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])

        def_conflicts = [c for c in contradictions if "Definition" in c.get("contradiction_type", "")]
        assert len(def_conflicts) > 0

    def test_identical_definitions_no_conflict(self):
        """Same term defined identically in two places → no conflict."""
        same_def = "any non-public information disclosed under this agreement"
        clause_a = make_clause(
            "1.1", "Other",
            f'"Trade Secrets" means {same_def}.',
            definitions={"Trade Secrets": same_def},
        )
        clause_b = make_clause(
            "2.1", "Confidentiality/NDA",
            f'Trade Secrets means {same_def} and shall be protected accordingly.',
            definitions={"Trade Secrets": same_def},
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])

        def_conflicts = [c for c in contradictions if "Definition" in c.get("contradiction_type", "")]
        assert len(def_conflicts) == 0


class TestTerminationConflict:
    def test_notice_required_vs_immediate_termination(self):
        """One clause requires notice; another allows immediate termination → conflict."""
        clause_a = make_clause(
            "8.1", "Termination",
            "Employer may terminate this agreement by providing 30 days written notice to Employee.",
        )
        clause_b = make_clause(
            "8.2", "Termination",
            "Notwithstanding any other provision, Employer may terminate employment immediately "
            "and without notice in cases of misconduct.",
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])

        term_conflicts = [c for c in contradictions if "Termination" in c.get("contradiction_type", "")]
        assert len(term_conflicts) > 0


class TestNoFalsePositives:
    def test_completely_unrelated_clauses_no_contradiction(self):
        """Unrelated clauses (payment + governing law) should produce no contradictions."""
        clause_a = make_clause(
            "4.1", "Payment Terms",
            "All fees shall be paid within 30 days of invoice date via bank transfer.",
        )
        clause_b = make_clause(
            "10.1", "Governing Law",
            "This agreement shall be governed by the laws of India and subject to the "
            "exclusive jurisdiction of the courts at Bangalore.",
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])
        # These clauses don't touch the same subject matter
        assert len(contradictions) == 0

    def test_single_clause_no_contradiction(self):
        """A single clause cannot contradict itself."""
        clause = make_clause(
            "1.1", "Confidentiality/NDA",
            "All confidential information shall be protected for a period of 3 years "
            "after termination with standard carveouts for publicly available information.",
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause])
        assert len(contradictions) == 0

    def test_output_structure_complete(self):
        """All contradiction records must have the required fields."""
        required_fields = {
            "contradiction_id", "clause_a_id", "clause_b_id",
            "clause_a_text", "clause_b_text", "contradiction_type",
            "description", "severity",
        }
        clause_a = make_clause(
            "6.1", "IP Assignment",
            "Company owns all inventions and all work product made by employee.",
        )
        clause_b = make_clause(
            "6.2", "IP Assignment",
            "Employee retains ownership of all IP and all inventions are employee's property.",
        )
        agent = ContradictionAgent()
        contradictions = agent.detect([clause_a, clause_b])

        for c in contradictions:
            assert required_fields.issubset(c.keys()), f"Missing fields in: {c}"
