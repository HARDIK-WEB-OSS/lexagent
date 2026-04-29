"""
Agent 2: Segmentation Agent
Splits legal text into structured clauses using hybrid structural + reference parsing.
"""
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Matches numbered/lettered clause headers at the start of a line
CLAUSE_HEADER_PATTERN = re.compile(
    r'^(\d+\.[\d\.]*|\([a-z]\)|\([ivxlcdm]+\)|[A-Z]+\s+\d+|'
    r'SECTION\s+\d+|Article\s+\d+|ARTICLE\s+\d+|SCHEDULE\s+[A-Z\d]+)\s+',
    re.MULTILINE | re.IGNORECASE,
)

# Matches cross-references inside clause text
REFERENCE_PATTERN = re.compile(
    r'(?:as defined in|pursuant to|as set forth in|subject to|in accordance with|'
    r'under|referred to in|described in|specified in)\s+'
    r'(Section|Clause|Article|Schedule|Exhibit|Appendix)\s+([\d\.]+[a-z]?)',
    re.IGNORECASE,
)

# Matches definition blocks: "Term" means ... or "Term" shall mean ...
DEFINITION_PATTERN = re.compile(
    r'"([^"]{2,80})"\s+(?:means|shall mean|refers to|is defined as|includes)\s+',
    re.IGNORECASE,
)

# Extracts page markers inserted by extraction agent
PAGE_MARKER_PATTERN = re.compile(r'\[PAGE\s+(\d+)\]')


class SegmentationAgent:
    """Segments raw legal text into structured clause objects."""

    def segment(self, raw_text: str) -> List[Dict]:
        # Remove page markers but track their positions for page hints
        page_positions = self._build_page_position_map(raw_text)
        clean_text = PAGE_MARKER_PATTERN.sub("", raw_text)

        # Step 1: Split on structural clause boundaries
        raw_clauses = self._split_on_clause_boundaries(clean_text)

        if not raw_clauses:
            # Fallback: treat every paragraph as a clause
            raw_clauses = self._split_on_paragraphs(clean_text)

        # Step 2: Build clause objects with IDs
        clauses = []
        global_definitions: Dict[str, str] = {}

        for idx, (header, body, char_offset) in enumerate(raw_clauses):
            clause_id = self._normalize_clause_id(header, idx)
            clause_text = (header + " " + body).strip() if header else body.strip()

            if not clause_text:
                continue

            # Step 3a: Extract definitions in this clause
            local_defs = self._extract_definitions(clause_text)
            global_definitions.update(local_defs)

            # Step 3b: Extract cross-references
            references = self._extract_references(clause_text)

            # Approximate page number
            page_hint = self._estimate_page(char_offset, page_positions)

            clauses.append(
                {
                    "clause_id": clause_id,
                    "clause_text": clause_text,
                    "references": references,
                    "definitions": local_defs,
                    "page_hint": page_hint,
                    "_char_offset": char_offset,
                }
            )

        logger.info(
            f"Segmented {len(clauses)} clauses. "
            f"Global definitions found: {len(global_definitions)}"
        )
        return clauses

    # ------------------------------------------------------------------
    # Structural splitting
    # ------------------------------------------------------------------

    def _split_on_clause_boundaries(self, text: str):
        """
        Returns list of (header, body, char_offset) tuples.
        """
        matches = list(CLAUSE_HEADER_PATTERN.finditer(text))
        if not matches:
            return []

        results = []
        for i, match in enumerate(matches):
            header = match.group(0).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[match.end(): end].strip()
            results.append((header, body, start))

        # Capture any preamble before first clause header
        if matches and matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if len(preamble) > 50:
                results.insert(0, ("PREAMBLE", preamble, 0))

        return results

    def _split_on_paragraphs(self, text: str):
        """Fallback: split on double newlines."""
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return [
            (f"PARA-{i + 1}", para, 0) for i, para in enumerate(paragraphs)
        ]

    # ------------------------------------------------------------------
    # Clause ID normalization
    # ------------------------------------------------------------------

    def _normalize_clause_id(self, header: str, fallback_idx: int) -> str:
        """
        Extract a clean clause identifier from the header string.
        '1.2.3 Some Title' → '1.2.3'
        'SECTION 4' → 'SECTION-4'
        """
        header = header.strip()
        # Numbered: 1.2 or 1.2.3
        m = re.match(r'^(\d+(?:\.\d+)*)', header)
        if m:
            return m.group(1)
        # Lettered: (a), (i)
        m = re.match(r'^\(([a-z]+)\)', header)
        if m:
            return m.group(1).upper()
        # SECTION/ARTICLE N
        m = re.match(r'^(SECTION|Article|ARTICLE|SCHEDULE)\s+(\S+)', header, re.IGNORECASE)
        if m:
            return f"{m.group(1).upper()}-{m.group(2)}"
        # Fallback
        if header and header != "PARA":
            slug = re.sub(r'\s+', '-', header[:20]).upper()
            return slug
        return f"CLAUSE-{fallback_idx + 1}"

    # ------------------------------------------------------------------
    # Reference extraction
    # ------------------------------------------------------------------

    def _extract_references(self, text: str) -> List[str]:
        """Extract referenced clause IDs from cross-reference language."""
        refs = []
        for match in REFERENCE_PATTERN.finditer(text):
            ref_id = match.group(2).strip()
            if ref_id not in refs:
                refs.append(ref_id)
        return refs

    # ------------------------------------------------------------------
    # Definition extraction
    # ------------------------------------------------------------------

    def _extract_definitions(self, text: str) -> Dict[str, str]:
        """Extract term→definition pairs from clause text."""
        defs = {}
        for match in DEFINITION_PATTERN.finditer(text):
            term = match.group(1).strip()
            # Get the definition text after the keyword, up to next sentence or 300 chars
            def_start = match.end()
            def_snippet = text[def_start: def_start + 300]
            # Truncate at sentence boundary
            sentence_end = re.search(r'[.;]', def_snippet)
            definition = (
                def_snippet[: sentence_end.start()].strip()
                if sentence_end
                else def_snippet.strip()
            )
            if term and definition:
                defs[term] = definition
        return defs

    # ------------------------------------------------------------------
    # Page hint estimation
    # ------------------------------------------------------------------

    def _build_page_position_map(self, text: str) -> Dict[int, int]:
        """Maps character offset → page number from [PAGE N] markers."""
        page_map = {}
        for match in PAGE_MARKER_PATTERN.finditer(text):
            page_num = int(match.group(1))
            page_map[match.start()] = page_num
        return page_map

    def _estimate_page(self, char_offset: int, page_positions: Dict[int, int]) -> int:
        """Find the page number for a given character offset."""
        if not page_positions:
            return 1
        current_page = 1
        for pos, page in sorted(page_positions.items()):
            if char_offset >= pos:
                current_page = page
            else:
                break
        return current_page
