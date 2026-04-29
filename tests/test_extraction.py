"""
Tests for ExtractionAgent
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.extraction_agent import ExtractionAgent


class TestPDFExtraction:
    """Tests for PDF text extraction path."""

    def test_pdf_extraction_text(self, tmp_path):
        """Mock fitz returning rich text — should use pymupdf path."""
        dummy_pdf = tmp_path / "contract.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 fake content")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is a full clause about confidentiality. " * 30

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3
        mock_doc.load_page.return_value = mock_page
        mock_doc.__iter__ = lambda self: iter([mock_page] * 3)

        with patch("fitz.open", return_value=mock_doc):
            agent = ExtractionAgent()
            result = agent.extract(str(dummy_pdf))

        assert result["extraction_method"] == "pymupdf"
        assert len(result["raw_text"]) > 100
        assert result["pages"] == 3
        assert result["file_name"] == "contract.pdf"
        assert result["file_size_bytes"] == dummy_pdf.stat().st_size
        assert "raw_text" in result

    def test_pymupdf_returns_dict_structure(self, tmp_path):
        """Verify all required keys are present in extraction result."""
        dummy_pdf = tmp_path / "test.pdf"
        dummy_pdf.write_bytes(b"%PDF dummy")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Rich contract text here. " * 50

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 2
        mock_doc.load_page.return_value = mock_page

        with patch("fitz.open", return_value=mock_doc):
            agent = ExtractionAgent()
            result = agent.extract(str(dummy_pdf))

        required_keys = {"raw_text", "pages", "extraction_method", "file_name", "file_size_bytes"}
        assert required_keys.issubset(result.keys())


class TestOCRFallback:
    """Tests for OCR fallback when PDF has minimal text."""

    def test_ocr_fallback_triggered(self, tmp_path):
        """When pymupdf returns < 100 chars/page average, OCR should be called."""
        dummy_pdf = tmp_path / "scanned.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 scanned")

        mock_page_text = MagicMock()
        mock_page_text.get_text.return_value = "tiny"  # < 100 chars

        mock_page_pix = MagicMock()
        mock_page_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_page_ocr = MagicMock()
        mock_page_ocr.get_text.return_value = "tiny"
        mock_page_ocr.get_pixmap.return_value = mock_page_pix

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 2
        mock_doc.load_page.return_value = mock_page_ocr

        fake_image = MagicMock()

        with patch("fitz.open", return_value=mock_doc), \
             patch("PIL.Image.open", return_value=fake_image), \
             patch("pytesseract.image_to_string", return_value="OCR extracted text from scanned contract clause number one.") as mock_tess:

            agent = ExtractionAgent()
            result = agent.extract(str(dummy_pdf))

        mock_tess.assert_called()
        assert result["extraction_method"] == "ocr"
        assert "OCR extracted text" in result["raw_text"]

    def test_ocr_fallback_uses_pytesseract(self, tmp_path):
        """Confirm pytesseract.image_to_string is the OCR backend."""
        dummy_pdf = tmp_path / "scan2.pdf"
        dummy_pdf.write_bytes(b"%PDF dummy")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "x"  # triggers OCR
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"\x89PNG" + b"\x00" * 50
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page

        mock_img = MagicMock()

        with patch("fitz.open", return_value=mock_doc), \
             patch("PIL.Image.open", return_value=mock_img), \
             patch("pytesseract.image_to_string", return_value="Contract terms and conditions.") as mock_tess:
            agent = ExtractionAgent()
            result = agent.extract(str(dummy_pdf))

        assert mock_tess.called
        assert result["extraction_method"] == "ocr"


class TestDocxExtraction:
    """Tests for DOCX extraction path."""

    def test_docx_extraction(self, tmp_path):
        """Mock python-docx Document — verify paragraphs are extracted."""
        dummy_docx = tmp_path / "contract.docx"
        dummy_docx.write_bytes(b"PK fake docx bytes")

        mock_para1 = MagicMock()
        mock_para1.text = "This Agreement is entered into between the parties."
        mock_para1.style.name = "Normal"

        mock_para2 = MagicMock()
        mock_para2.text = "1. Confidentiality"
        mock_para2.style.name = "Heading 1"

        mock_para3 = MagicMock()
        mock_para3.text = "Each party shall keep all information confidential."
        mock_para3.style.name = "Normal"

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_doc.tables = []

        with patch("docx.Document", return_value=mock_doc):
            agent = ExtractionAgent()
            result = agent.extract(str(dummy_docx))

        assert result["extraction_method"] == "docx"
        assert "Confidentiality" in result["raw_text"]
        assert "confidential" in result["raw_text"].lower()
        assert result["pages"] >= 1

    def test_docx_heading_preserved(self, tmp_path):
        """Heading-style paragraphs should be prefixed with # markers."""
        dummy_docx = tmp_path / "headed.docx"
        dummy_docx.write_bytes(b"PK fake docx bytes")

        mock_h1 = MagicMock()
        mock_h1.text = "Section 1: Definitions"
        mock_h1.style.name = "Heading 1"

        mock_h2 = MagicMock()
        mock_h2.text = "1.1 Confidential Information"
        mock_h2.style.name = "Heading 2"

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_h1, mock_h2]
        mock_doc.tables = []

        with patch("docx.Document", return_value=mock_doc):
            agent = ExtractionAgent()
            result = agent.extract(str(dummy_docx))

        assert "# Section 1" in result["raw_text"]
        assert "## 1.1" in result["raw_text"]

    def test_unsupported_extension_raises(self, tmp_path):
        """Non-PDF/DOCX file should raise ValueError."""
        txt_file = tmp_path / "contract.txt"
        txt_file.write_text("Some text")
        agent = ExtractionAgent()
        with pytest.raises(ValueError, match="Unsupported file type"):
            agent.extract(str(txt_file))
