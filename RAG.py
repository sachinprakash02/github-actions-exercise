# src/tools/html_preprocessor.py
from bs4 import BeautifulSoup
import re

class HTMLPreprocessor:
    """
    Preprocesses HTML exported from Docling into structured text chunks
    with section, subsection, and block-type metadata.
    """

    def __init__(self, min_chunk_len: int = 50):
        self.min_chunk_len = min_chunk_len

    def preprocess(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        chunks = []
        current_section = None
        current_subsection = None

        # Remove non-textual or noisy elements
        for tag in soup(["script", "style", "footer", "header", "nav", "img", "svg"]):
            tag.decompose()

        # Clean excessive newlines and whitespace
        def clean_text(t: str):
            t = re.sub(r"\s+", " ", t).strip()
            return t

        for elem in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table", "caption"]):
            tag = elem.name
            text = clean_text(elem.get_text(" "))

            if not text or len(text) < self.min_chunk_len:
                continue

            # Track section hierarchy
            if tag == "h1":
                current_section = text
                current_subsection = None
                continue
            elif tag == "h2":
                current_subsection = text
                continue

            # Handle tables
            if tag == "table":
                table_text = self._table_to_text(elem)
                chunks.append({
                    "text": f"[TABLE]\n{table_text}\n[/TABLE]",
                    "section": current_section,
                    "subsection": current_subsection,
                    "block_type": "table",
                })
                continue

            # Regular paragraph or list
            chunks.append({
                "text": text,
                "section": current_section,
                "subsection": current_subsection,
                "block_type": tag,
            })

        return chunks

    def _table_to_text(self, table_elem):
        """
        Convert <table> → readable text form.
        """
        rows = []
        for tr in table_elem.find_all("tr"):
            cells = [re.sub(r"\s+", " ", c.get_text(" ").strip()) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
