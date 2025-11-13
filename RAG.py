import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextExtractor:
    """
    Extracts:
     - page-wise text blocks
     - font sizes for heading detection
     - logical reading order
    Using PyMuPDF (fitz) which is deterministic and stable.
    """

    def __init__(self, heading_font_threshold: float = 2.0):
        """
        heading_font_threshold:
            multiplier used to detect headings.
            If a block's average font size >= mean_font * threshold → heading.
        """
        self.heading_font_threshold = heading_font_threshold

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Returns list of blocks:
        [
            {
                "page": int,
                "text": str,
                "bbox": (x0, y0, x1, y1),
                "font_sizes": [...],
                "is_heading": bool
            },
            ...
        ]
        """
        logger.info(f"Extracting text with PyMuPDF: {file_path}")
        doc = fitz.open(file_path)

        blocks_out = []

        for page_index, page in enumerate(doc):
            blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1, text, block_no)
            spans = page.get_text("dict")["blocks"]

            # Map block IDs to font sizes (average)
            block_font_sizes = self._extract_font_sizes(spans)

            for i, b in enumerate(blocks):
                x0, y0, x1, y1, text, block_no = b
                cleaned = self._clean_text(text)

                if not cleaned.strip():
                    continue

                avg_font = block_font_sizes.get(i, 10.0)
                is_heading = avg_font >= (self._mean_font(block_font_sizes) * self.heading_font_threshold)

                blocks_out.append({
                    "page": page_index + 1,
                    "text": cleaned,
                    "bbox": (x0, y0, x1, y1),
                    "avg_font": avg_font,
                    "is_heading": is_heading
                })

        return blocks_out

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _extract_font_sizes(self, spans_blocks):
        """
        Return {block_index: avg_font_size}
        """
        block_sizes = {}

        block_id = 0
        for b in spans_blocks:
            if "lines" not in b:
                continue
            sizes = []
            for line in b["lines"]:
                for span in line["spans"]:
                    sizes.append(span.get("size", 10.0))

            if sizes:
                block_sizes[block_id] = sum(sizes) / len(sizes)
            block_id += 1

        return block_sizes

    def _mean_font(self, block_sizes: Dict[int, float]) -> float:
        if not block_sizes:
            return 10.0
        return sum(block_sizes.values()) / len(block_sizes)

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()




import re

class SectionParser:
    """
    Converts raw text blocks from PyMuPDF into:
        - section titles (H1)
        - subsection titles (H2)
        - paragraph blocks
    """

    def parse(self, blocks):
        """
        blocks: output from TextExtractor.extract()

        Returns list of:
        {
          "text": "...",
          "page": ...,
          "section": "...",
          "subsection": "...",
          "block_type": "heading" | "paragraph"
        }
        """
        current_section = None
        current_subsection = None
        out = []

        for b in blocks:
            txt = b["text"]
            is_heading = b["is_heading"]

            if is_heading:
                # Heuristic: If ALL CAPS or starts with number, treat as section
                if txt.isupper() or re.match(r"^\d+(\.\d+)*", txt):
                    current_section = txt
                    current_subsection = None
                else:
                    current_subsection = txt

                out.append({
                    "text": txt,
                    "page": b["page"],
                    "section": current_section,
                    "subsection": current_subsection,
                    "block_type": "heading"
                })
            else:
                out.append({
                    "text": txt,
                    "page": b["page"],
                    "section": current_section,
                    "subsection": current_subsection,
                    "block_type": "paragraph"
                })

        return out
# src/preprocessing/chunker.py
import re
from typing import List, Dict, Any


class Chunker:
    """
    Takes parsed blocks (from SectionParser) and produces final text chunks.
    Handles:
        - paragraph grouping
        - chunk size limits (by character or token heuristic)
        - section/subsection context prefixing
    """

    def __init__(self, max_chars: int = 1200, overlap: int = 150):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, parsed_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        parsed_blocks: each item contains:
            text, page, section, subsection, block_type
        """

        chunks = []
        buffer = []
        buffer_meta = None
        char_count = 0

        for blk in parsed_blocks:
            text = blk["text"]
            block_type = blk["block_type"]

            # ---------- Heading becomes its own chunk ----------
            if block_type == "heading":
                # Flush buffer if exists
                if buffer:
                    chunks.append(self._make_chunk(buffer, buffer_meta))
                    buffer = []
                    buffer_meta = None
                    char_count = 0

                chunks.append({
                    "text": text,
                    "metadata": {
                        "page": blk["page"],
                        "section": blk["section"],
                        "subsection": blk["subsection"],
                        "block_type": "heading"
                    }
                })
                continue

            # ---------- Paragraph grouping ----------
            if not buffer:
                buffer_meta = {
                    "page": blk["page"],
                    "section": blk["section"],
                    "subsection": blk["subsection"],
                    "block_type": "paragraph"
                }

            # Add paragraph text
            paragraph = text.strip()
            if not paragraph:
                continue

            paragraph = paragraph.replace("\n", " ")

            # Add to buffer
            buffer.append(paragraph)
            char_count += len(paragraph)

            # If buffer exceeds threshold, emit chunk
            if char_count >= self.max_chars:
                chunk = self._make_chunk(buffer, buffer_meta)
                chunks.append(chunk)

                # Prepare next buffer with overlap
                leftover = self._apply_overlap(buffer)
                buffer = leftover
                char_count = sum(len(x) for x in buffer)

        # Flush last buffer
        if buffer:
            chunks.append(self._make_chunk(buffer, buffer_meta))

        return chunks

    # --------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------

    def _make_chunk(self, buffer: List[str], meta: Dict[str, Any]):
        text = " ".join(buffer).strip()
        return {"text": text, "metadata": dict(meta)}

    def _apply_overlap(self, buffer: List[str]) -> List[str]:
        """Keep trailing tokens for semantic continuity."""
        joined = " ".join(buffer)
        if len(joined) <= self.overlap:
            return [joined]
        return [joined[-self.overlap:]]  # naive but effective
# src/preprocessing/metadata_normalizer.py
import hashlib
from datetime import datetime
from typing import Dict, Any


DEFAULT_KEYS = [
    "source",
    "file_name",
    "page",
    "section",
    "subsection",
    "object",
    "attribute",
    "xpath",
    "usecase",
    "block_type",
    "token_count",
    "ingestion_date",
    "processor",
    "embedding_version",
    "chunk_id"
]


class MetadataNormalizer:

    def normalize(self, meta: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Ensures all standard metadata fields exist.
        Adds chunk_id, ingestion_date, token_count.
        """

        clean = {}

        # Fill required keys
        for k in DEFAULT_KEYS:
            clean[k] = meta.get(k)

        # System metadata
        clean["token_count"] = self._estimate_tokens(text)
        clean["ingestion_date"] = clean["ingestion_date"] or self._now()
        clean["chunk_id"] = clean["chunk_id"] or self._make_chunk_id(meta, text)

        return clean

    # --------------------------
    # Helpers
    # --------------------------

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _make_chunk_id(self, meta: Dict[str, Any], text: str) -> str:
        payload = (
            (meta.get("file_name") or "") + "|" +
            (meta.get("section") or "") + "|" +
            (meta.get("subsection") or "") + "|" +
            text[:50]
        )
        return hashlib.md5(payload.encode()).hexdigest()

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"


# src/preprocessing/qdrant_ingestor.py
from typing import List, Dict, Any, Callable
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QdrantIngestor:
    """
    Inserts chunks into a Qdrant collection with embeddings.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_fn: Callable[[List[str]], List[List[float]]],
        vector_size: int = None,
        batch_size: int = 128
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.batch_size = batch_size
        self.vector_size = vector_size

    # --------------------------------------------------------
    # Initialize collection (idempotent)
    # --------------------------------------------------------
    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
            return
        except Exception:
            logger.info(f"Creating collection {self.collection_name}")

        # detect dim if not given
        if self.vector_size is None:
            v = self.embedding_fn(["test"])[0]
            self.vector_size = len(v)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )

    # --------------------------------------------------------
    # Ingest
    # --------------------------------------------------------
    def ingest(self, chunks: List[Dict[str, Any]]):
        self._ensure_collection()

        ids = []
        for batch in self._split_batches(chunks):
            texts = [c["text"] for c in batch]
            metas = [c["metadata"] for c in batch]
            embeddings = self.embedding_fn(texts)

            points = []
            for chunk, emb, meta in zip(batch, embeddings, metas):
                points.append({
                    "id": meta["chunk_id"],
                    "vector": emb,
                    "payload": {
                        **meta,
                        "preview": chunk["text"][:300]
                    }
                })

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            ids.extend([p["id"] for p in points])

            logger.info(f"Ingested batch of {len(batch)} chunks.")

        return ids

    def _split_batches(self, arr):
        for i in range(0, len(arr), self.batch_size):
            yield arr[i:i + self.batch_size]

