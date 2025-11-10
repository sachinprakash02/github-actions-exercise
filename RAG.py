import pandas as pd
import os
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ExcelRAGIngestor:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore

    def _row_to_chunk(self, row: pd.Series, usecase: str, source_file: str) -> Tuple[str, Dict[str, Any]]:
        """Convert one Excel row to a (text, metadata) pair."""
        object_name = str(row.get("Object", "")).strip()
        attribute = str(row.get("Attribute", "")).strip()
        full_xpath = str(row.get("Full XPath", row.get("FullXpath", ""))).strip()
        allowed = str(row.get("Allowed Values", "")).strip()
        notes = str(row.get("Notes", "")).strip()
        mod = str(row.get("Mod", "")).strip()
        sup = str(row.get("Sup", "")).strip()
        src = str(row.get("Source", "")).strip()

        text_block = f"""[Usecase: {usecase}]
[Object: {object_name}]
Attribute: {attribute}
Full XPath: {full_xpath}
Allowed Values: {allowed}
Mode: {mod}
Support: {sup}
Notes: {notes}
Source: {src}
"""

        metadata = {
            "usecase": usecase,
            "object": object_name,
            "attribute": attribute,
            "xpath": full_xpath,
            "allowed_values": allowed,
            "mode": mod,
            "support": sup,
            "notes": notes,
            "source": src,
            "sheet": usecase,
            "file": os.path.basename(source_file)
        }
        return text_block.strip(), metadata

    def ingest_excel(self, file_path: str, batch_size: int = 200):
        logger.info("Reading Excel workbook %s", file_path)
        xls = pd.ExcelFile(file_path)
        all_texts, all_metas = [], []

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = df.dropna(how="all")
            logger.info("Processing sheet %s (%d rows)", sheet, len(df))
            for _, row in df.iterrows():
                text, meta = self._row_to_chunk(row, sheet, file_path)
                all_texts.append(text)
                all_metas.append(meta)

        logger.info("Prepared %d total chunks for ingestion", len(all_texts))

        if self.vectorstore:
            for i in range(0, len(all_texts), batch_size):
                batch_t = all_texts[i:i+batch_size]
                batch_m = all_metas[i:i+batch_size]
                self.vectorstore.add_texts(batch_t, metadatas=batch_m)
            logger.info("Ingestion into vectorstore complete")

        return all_texts, all_metas
