"""Multi-format document byte parsers.

Extracts plain-text or structured Markdown from uploaded or downloaded file
bytes (PDF, Word DOCX, Markdown, plain text).
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from typing import Final

import structlog
from pypdf import PdfReader

logger: structlog.BoundLogger = structlog.get_logger(__name__)

DOCX_MIME: Final[str] = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_MIME: Final[str] = "application/pdf"
_WORD_NAMESPACES: Final[dict[str, str]] = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}


def parse_pdf_bytes(content: bytes) -> str:
    """Extract text from PDF bytes, inserting markdown headers for each page.

    Args:
        content: Raw bytes of the PDF file.

    Returns:
        Extracted Markdown string with '## Page N' headers.
    """
    if not content:
        return ""
    stream = io.BytesIO(content)
    reader = PdfReader(stream)
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"## Page {i + 1}\n\n{page_text}")
    return "\n\n".join(pages)


def parse_docx_bytes(content: bytes) -> str:
    """Extract text from Word (.docx) OpenXML archive bytes.

    Args:
        content: Raw bytes of the .docx file.

    Returns:
        Extracted plain text joined by double newlines.
    """
    if not content:
        return ""
    stream = io.BytesIO(content)
    with zipfile.ZipFile(stream) as z:
        if "word/document.xml" not in z.namelist():
            return ""
        xml_content = z.read("word/document.xml")
    tree = ET.fromstring(xml_content)
    paragraphs: list[str] = []
    for p in tree.iterfind(".//w:p", _WORD_NAMESPACES):
        texts = [
            node.text
            for node in p.iterfind(".//w:t", _WORD_NAMESPACES)
            if node.text
        ]
        if texts:
            paragraphs.append("".join(texts))
    return "\n\n".join(paragraphs)


def parse_bytes_to_markdown(
    content: bytes,
    filename: str = "",
    mime_type: str = "",
) -> str:
    """Parse raw bytes of various file formats into Markdown or plain text.

    Determines format by matching MIME type or file extension, then extracts
    content accordingly.

    Args:
        content: Raw binary content of the file.
        filename: Optional name of the file (used for extension detection).
        mime_type: Optional MIME type string.

    Returns:
        Extracted text as a UTF-8 Markdown/plain text string.
    """
    lower_fn = filename.lower()
    is_pdf = (
        mime_type == PDF_MIME
        or lower_fn.endswith(".pdf")
    )
    is_docx = (
        mime_type in (DOCX_MIME, "application/msword")
        or lower_fn.endswith(".docx")
    )

    if is_pdf:
        try:
            return parse_pdf_bytes(content)
        except Exception as exc:
            logger.warning("pdf_parsing_failed", filename=filename, error=str(exc))
            return ""

    if is_docx:
        try:
            return parse_docx_bytes(content)
        except Exception as exc:
            logger.warning("docx_parsing_failed", filename=filename, error=str(exc))
            return ""

    # Default: text, markdown, csv, json, code
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)
