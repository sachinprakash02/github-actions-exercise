# src/tools/html_preprocessor.py
from bs4 import BeautifulSoup
import re
import json

class HTMLPreprocessor:
    """
    Enhanced HTML preprocessor for Docling-exported HTML.
    Extracts structured text + tables efficiently with JSON metadata.
    """

    def __init__(self, min_chunk_len: int = 50):
        self.min_chunk_len = min_chunk_len

    def preprocess(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        chunks = []
        current_section = None
        current_subsection = None

        for tag in soup(["script", "style", "footer", "header", "nav", "img", "svg"]):
            tag.decompose()

        def clean_text(t: str):
            return re.sub(r"\s+", " ", t or "").strip()

        for elem in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table", "caption"]):
            tag = elem.name
            text = clean_text(elem.get_text(" "))

            if not text or len(text) < self.min_chunk_len:
                continue

            if tag == "h1":
                current_section = text
                current_subsection = None
                continue
            elif tag == "h2":
                current_subsection = text
                continue

            # Handle tables separately
            if tag == "table":
                table_summary, table_json = self._process_table_block(elem)
                if table_summary:
                    chunks.append({
                        "text": f"[TABLE] {table_summary} [/TABLE]",
                        "section": current_section,
                        "subsection": current_subsection,
                        "block_type": "table",
                        "table_json": table_json
                    })
                continue

            chunks.append({
                "text": text,
                "section": current_section,
                "subsection": current_subsection,
                "block_type": tag,
            })

        return chunks

    def _process_table_block(self, table_elem):
        """
        Extracts structured tabular data and returns both:
        1. A readable summary for embeddings
        2. A structured JSON version for metadata
        """
        rows = []
        headers = []
        for tr in table_elem.find_all("tr"):
            cells = [re.sub(r"\s+", " ", c.get_text(" ").strip()) for c in tr.find_all(["th", "td"])]
            if not cells:
                continue
            if not headers:
                headers = cells
            else:
                rows.append(cells)

        if not headers and not rows:
            return None, None

        # --- Build structured table JSON ---
        table_json = {"headers": headers, "rows": rows}
        table_json_str = json.dumps(table_json, ensure_ascii=False)

        # --- Build a short summary string for embedding ---
        # Try to build key-value representation for first few rows
        summary_lines = []
        max_rows = min(5, len(rows))  # limit summary to first 5 rows
        for row in rows[:max_rows]:
            if len(row) == len(headers):
                kv_pairs = ", ".join(f"{h}: {v}" for h, v in zip(headers, row))
                summary_lines.append(kv_pairs)
            else:
                summary_lines.append(" | ".join(row))
        table_summary = " | ".join(headers) + "\n" + "\n".join(summary_lines)

        return table_summary, table_json_str
