"""Unit tests for document parsing (PDF page numbering and Word DOCX extraction)."""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile


from datetime import datetime, timezone

from backend.ingestion.chunker import SemanticMarkdownChunker
from backend.models.document import Document


def test_chunker_handles_pdf_page_headers() -> None:
    """Ensure ## Page N headers are parsed into section_headers for clean PDF citations."""
    now = datetime.now(tz=timezone.utc)
    sample_doc = Document(
        doc_id="pdf-doc-1",
        title="Financial Report.pdf",
        source_url="https://drive.google.com/open?id=pdf-doc-1",
        mime_type="application/pdf",
        owner_email="cfo@acme.com",
        allowed_emails=["user@acme.com"],
        last_modified=now,
        synced_at=now,
    )
    pdf_extracted_text = (
        "## Page 1\n\n"
        "Executive Summary: Company revenue grew by 24% year over year.\n\n"
        "## Page 2\n\n"
        "Operating Expenses: R&D expenses totaled $4.2M for the fiscal quarter.\n\n"
        "## Page 3\n\n"
        "Guidance: Projected Q4 revenue range is $18M to $20M."
    )
    chunker = SemanticMarkdownChunker(max_tokens=100, overlap_tokens=10)
    chunks = chunker.chunk_document(sample_doc, pdf_extracted_text)

    assert len(chunks) >= 3
    section_headers = [c.section_header for c in chunks]
    assert any("Page 1" in (h or "") for h in section_headers)
    assert any("Page 2" in (h or "") for h in section_headers)
    assert any("Page 3" in (h or "") for h in section_headers)


def test_docx_xml_paragraph_extraction() -> None:
    """Test standard library XML extraction from a mock in-memory DOCX zip."""
    # Build a minimal valid DOCX zip archive in memory
    fh = io.BytesIO()
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        "  <w:body>\n"
        "    <w:p><w:r><w:t>## Employee Code of Conduct</w:t></w:r></w:p>\n"
        "    <w:p><w:r><w:t>Employees must act with integrity at all times.</w:t></w:r></w:p>\n"
        "  </w:body>\n"
        "</w:document>"
    )
    with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", doc_xml)

    fh.seek(0)
    # Parse using our exact logic
    with zipfile.ZipFile(fh) as z:
        xml_content = z.read("word/document.xml")
    tree = ET.fromstring(xml_content)
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for p in tree.iterfind(".//w:p", namespaces):
        texts = [node.text for node in p.iterfind(".//w:t", namespaces) if node.text]
        if texts:
            paragraphs.append("".join(texts))

    extracted = "\n\n".join(paragraphs)
    assert "## Employee Code of Conduct" in extracted
    assert "Employees must act with integrity at all times." in extracted
