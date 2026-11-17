"""Multi-format document byte parsers.

Extracts plain-text or structured Markdown from uploaded or downloaded file
bytes (PDF, Word DOCX, Markdown, plain text).
"""

from __future__ import annotations

import csv
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
    """Extract structured text and tables from Word (.docx) OpenXML archive bytes.

    Args:
        content: Raw bytes of the .docx file.

    Returns:
        Extracted Markdown string preserving paragraphs and tables.
    """
    if not content:
        return ""
    stream = io.BytesIO(content)
    try:
        with zipfile.ZipFile(stream) as z:
            if "word/document.xml" not in z.namelist():
                return ""
            xml_content = z.read("word/document.xml")
    except Exception as exc:
        logger.warning("docx_zip_read_failed", error=str(exc))
        return ""

    tree = ET.fromstring(xml_content)
    body = tree.find(".//w:body", _WORD_NAMESPACES)
    if body is None:
        return ""

    blocks: list[str] = []
    for elem in body:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag == "p":
            texts = [
                node.text
                for node in elem.iterfind(".//w:t", _WORD_NAMESPACES)
                if node.text
            ]
            if texts:
                blocks.append("".join(texts))
        elif tag == "tbl":
            rows: list[list[str]] = []
            for tr in elem.iterfind(".//w:tr", _WORD_NAMESPACES):
                row_cells: list[str] = []
                for tc in tr.iterfind(".//w:tc", _WORD_NAMESPACES):
                    cell_text = "".join(
                        node.text
                        for node in tc.iterfind(".//w:t", _WORD_NAMESPACES)
                        if node.text
                    ).strip()
                    row_cells.append(cell_text.replace("|", "\\|"))
                if row_cells and any(row_cells):
                    rows.append(row_cells)

            if rows:
                col_count = max(len(r) for r in rows)
                padded = [r + [""] * (col_count - len(r)) for r in rows]
                tbl_lines = [
                    "| " + " | ".join(padded[0]) + " |",
                    "| " + " | ".join(["---"] * col_count) + " |",
                ]
                for row in padded[1:]:
                    tbl_lines.append("| " + " | ".join(row) + " |")
                blocks.append("\n".join(tbl_lines))

    return "\n\n".join(blocks)


def parse_csv_bytes(content: bytes) -> str:
    """Extract CSV bytes into a clean GitHub Flavored Markdown table.

    Args:
        content: Raw binary CSV bytes.

    Returns:
        Structured Markdown table string.
    """
    if not content:
        return ""
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows: list[list[str]] = []
    for r in reader:
        if r and any(cell.strip() for cell in r):
            rows.append([cell.strip().replace("|", "\\|") for cell in r])

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    padded = [r + [""] * (col_count - len(r)) for r in rows]
    table_lines = [
        "| " + " | ".join(padded[0]) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in padded[1:]:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


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
    is_csv = (
        mime_type in ("text/csv", "application/csv")
        or lower_fn.endswith(".csv")
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

    if is_csv:
        try:
            return parse_csv_bytes(content)
        except Exception as exc:
            logger.warning("csv_parsing_failed", filename=filename, error=str(exc))
            return ""

    # Default: text, markdown, json, code
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)
