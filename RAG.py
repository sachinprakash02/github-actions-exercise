# src/tools/document_loader.py
from docling.document_converter import DocumentConverter
import re

class DoclingLoader:
    """
    Loader for Docling v2.58+ with table extraction support.
    Extracts text, headings, and tables with metadata.
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def load_markdown(self, file_path: str) -> str:
        """
        Converts document to Markdown for inspection.
        """
        result = self.converter.convert(file_path)
        md = result.export_to_markdown()
        md = md.replace("<!-- image -->", "")
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md

    def load_with_metadata(self, file_path: str):
        """
        Returns list of dicts: each with text, section, subsection, page, and source info.
        Includes flattened tables as readable text.
        """
        result = self.converter.convert(file_path)
        pages = getattr(result, "pages", [])
        structured_chunks = []
        current_section, current_subsection = None, None

        for page in pages:
            page_num = getattr(page, "page_number", None)
            for block in getattr(page, "blocks", []):
                btype = getattr(block, "type", "").lower()
                text = getattr(block, "text", "").strip()

                # ---- Handle headings ----
                if btype == "heading":
                    level = getattr(block, "level", None)
                    if level == 1:
                        current_section = text
                        current_subsection = None
                    elif level == 2:
                        current_subsection = text
                    continue

                # ---- Handle regular text ----
                if btype in ("paragraph", "list", "quote") and text:
                    structured_chunks.append({
                        "text": text,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page": page_num,
                        "block_type": btype
                    })

                # ---- Handle tables ----
                elif btype == "table":
                    table_text = self._process_table_block(block)
                    if table_text:
                        structured_chunks.append({
                            "text": table_text,
                            "section": current_section,
                            "subsection": current_subsection,
                            "page": page_num,
                            "block_type": "table"
                        })

        if not structured_chunks:
            print("⚠️ No text or tables extracted. Ensure the PDF is selectable.")
        return structured_chunks

    def _process_table_block(self, block):
        """
        Converts a Docling table block into readable text for embeddings.
        """
        # Some blocks provide pre-rendered text
        if getattr(block, "text", None):
            return f"[TABLE]\n{block.text.strip()}\n[/TABLE]"

        # Try to parse structured table data (if available)
        table_data = getattr(block, "table_data", None)
        if not table_data:
            return None

        rows = []
        for row in table_data:
            cells = [str(cell).strip() for cell in row if cell]
            rows.append(" | ".join(cells))
        table_text = "\n".join(rows)

        caption = getattr(block, "caption", "")
        if caption:
            caption = caption.strip()
        return f"[TABLE] {caption}\n{table_text}\n[/TABLE]"
