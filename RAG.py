# src/preprocessing/document_preprocessor.py
"""
Unified Document Preprocessor & Qdrant ingestor.

Supports:
 - PDF (via Docling Document -> doc.get_elements() or result.document.to_dict())
 - DOCX (via mammoth to HTML -> HTMLPreprocessor)
 - Excel (pandas; multiple sheets -> structured rows)

Outputs:
 - List[Chunk] (text + normalized metadata)
 - Provides Qdrant ingestion helper to push embeddings + metas

Requirements (install separately):
 - docling (for PDF)
 - beautifulsoup4 lxml
 - pandas openpyxl
 - python-mammoth (optional for docx)
 - qdrant-client
 - langchain (if you use LangChain Qdrant wrapper & embeddings)
"""

from __future__ import annotations
import os
import json
import time
import hashlib
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Iterable, Tuple
from datetime import datetime
import itertools
import uuid

# Optional libs - import safely
try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None

try:
    import mammoth
except Exception:
    mammoth = None

from bs4 import BeautifulSoup
import pandas as pd

# Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance
except Exception:
    QdrantClient = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------
# dataclasses & helpers
# ----------------------------
@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]

    def as_record(self) -> Tuple[str, Dict[str, Any]]:
        return self.text, dict(self.metadata)


DEFAULT_METADATA_KEYS = [
    "source", "file_name", "page", "usecase", "object",
    "attribute", "xpath", "section", "subsection",
    "block_type", "token_count", "ingestion_date",
    "processor", "embedding_version", "chunk_id"
]


def fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def ensure_keys(meta: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = defaults or {}
    out = {}
    for k in DEFAULT_METADATA_KEYS:
        out[k] = meta.get(k, defaults.get(k))
    # keep any extra keys too
    for k, v in meta.items():
        if k not in out:
            out[k] = v
    return out


def make_chunk_id(file_name: str, meta: Dict[str, Any], text: str) -> str:
    base = f"{file_name}|{meta.get('section') or ''}|{meta.get('object') or ''}|{fingerprint(text)}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


# ----------------------------
# Loaders
# ----------------------------
class BaseLoader:
    """Abstract loader: returns a list of element dicts with semantic 'type' and text and page info."""
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError()


class DoclingPDFLoader(BaseLoader):
    """Uses Docling DocumentConverter to extract semantic elements."""
    def __init__(self):
        if DocumentConverter is None:
            raise RuntimeError("Docling not available. Install docling to use DoclingPDFLoader.")
        self.converter = DocumentConverter()

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info("Docling converting %s", file_path)
        result = self.converter.convert(file_path)
        doc = result.document
        # prefer get_elements if available
        elements = []
        try:
            for el in doc.get_elements():
                elements.append({
                    "type": getattr(el, "label", None) or "text",
                    "text": getattr(el, "text", None) or "",
                    "page": getattr(el, "page_no", None),
                    "raw": getattr(el, "to_dict", lambda: None)()
                })
        except Exception:
            # fallback: doc.to_dict traversal
            logger.warning("doc.get_elements() not available; falling back to to_dict traversal")
            dd = doc.to_dict()
            # naive walker: pages -> elements
            for page in dd.get("pages", []):
                pno = page.get("page_number")
                for el in page.get("elements", []):
                    elements.append({
                        "type": el.get("type") or el.get("label") or "text",
                        "text": el.get("text") or el.get("orig") or "",
                        "page": pno,
                        "raw": el
                    })
        logger.info("Docling returned %d elements", len(elements))
        return elements


class DOCXLoader(BaseLoader):
    """Simple DOCX to HTML via mammoth, then fallback to HTMLPreprocessor."""
    def __init__(self):
        if mammoth is None:
            logger.warning("mammoth not installed. DOCXLoader may not work. Install python-mammoth.")
        pass

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info("Converting DOCX via mammoth: %s", file_path)
        if mammoth is None:
            raise RuntimeError("mammoth is required for DOCX conversion.")
        with open(file_path, "rb") as f:
            result = mammoth.convert_to_html(f)
            html = result.value
        # Use HTMLPreprocessor to parse html below (we return a single html element)
        return [{"type": "html", "text": html, "page": None, "raw": None}]


class ExcelLoader(BaseLoader):
    """Load Excel workbook -> list of structured rows with sheet name (used as usecase)."""
    def __init__(self, sheet_name_map: Optional[Dict[str, str]] = None):
        self.sheet_name_map = sheet_name_map or {}

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info("Reading Excel workbook %s", file_path)
        xls = pd.ExcelFile(file_path)
        rows = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
            df = df.fillna("")  # avoid NaN
            logger.info("Processing sheet %s rows=%d", sheet, len(df))
            for _, r in df.iterrows():
                row_dict = r.to_dict()
                row_dict["_sheet"] = sheet
                row_dict["_file"] = os.path.basename(file_path)
                rows.append({
                    "type": "excel_row",
                    "text": None,
                    "row": row_dict,
                    "page": None,
                    "raw": row_dict
                })
        return rows


# ----------------------------
# HTML Preprocessor (sectioning + tables)
# ----------------------------
class HTMLPreprocessor:
    """
    Input: HTML string or docling element with 'text' containing HTML/markdown-ish markup.
    Output: List[Chunk] with unified metadata skeleton.
    Responsibilities:
     - extract headings (h1..h4) and maintain context stack
     - emit paragraph chunks with section metadata
     - extract <table> into structured rows and table-chunks
    """
    def __init__(self, processor_name: str = "HTMLPreprocessor-v1"):
        self.processor_name = processor_name

    def _clean_text(self, s: str) -> str:
        if not s:
            return ""
        return " ".join(s.strip().split())

    def process_html(self, html: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        soup = BeautifulSoup(html, "lxml")
        body = soup.body or soup

        chunks: List[Chunk] = []
        section_stack = []  # list of (level, title)
        # Walk and collect meaningful nodes
        for node in body.descendants:
            if getattr(node, "name", None) and node.name in ["h1", "h2", "h3", "h4"]:
                title = self._clean_text(node.get_text(" ", strip=True))
                level = int(node.name[1])
                # pop until lower level
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, title))
                # also emit heading chunk
                meta = dict(base_meta)
                meta.update({
                    "section": title,
                    "subsection": None,
                    "block_type": "heading",
                    "processor": self.processor_name
                })
                text = f"[HEADING] {title}"
                chunks.append(Chunk(text=text, metadata=ensure_keys(meta)))
            elif getattr(node, "name", None) and node.name == "p":
                txt = self._clean_text(node.get_text(" ", strip=True))
                if not txt:
                    continue
                # derive section and subsection
                section = section_stack[0][1] if section_stack else base_meta.get("section")
                subsection = section_stack[-1][1] if section_stack else None
                meta = dict(base_meta)
                meta.update({
                    "section": section,
                    "subsection": subsection,
                    "block_type": "paragraph",
                    "processor": self.processor_name
                })
                chunks.append(Chunk(text=txt, metadata=ensure_keys(meta)))
            elif getattr(node, "name", None) and node.name == "table":
                # parse table into header + rows
                headers = []
                rows = []
                # try thead first
                thead = node.find("thead")
                if thead:
                    ths = thead.find_all("th")
                    headers = [self._clean_text(th.get_text(" ", strip=True)) for th in ths]
                    tbody = node.find("tbody") or node
                    for tr in tbody.find_all("tr"):
                        tds = [self._clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["td","th"])]
                        if any(td.strip() for td in tds):
                            rows.append(tds)
                else:
                    # fallback: first tr as header if it has th or many cells
                    trs = node.find_all("tr")
                    if not trs:
                        continue
                    first = trs[0]
                    first_cells = [self._clean_text(x.get_text(" ", strip=True)) for x in first.find_all(["th","td"])]
                    if first.find_all("th") or len(first_cells) > 1:
                        headers = first_cells
                        for tr in trs[1:]:
                            tds = [self._clean_text(x.get_text(" ", strip=True)) for x in tr.find_all(["td","th"])]
                            if any(tds):
                                rows.append(tds)
                    else:
                        # no headers
                        for tr in trs:
                            tds = [self._clean_text(x.get_text(" ", strip=True)) for x in tr.find_all(["td","th"])]
                            if any(tds):
                                rows.append(tds)
                # create one or more chunks: either one chunk per row or grouped
                section = section_stack[0][1] if section_stack else base_meta.get("section")
                meta_base = dict(base_meta)
                meta_base.update({
                    "section": section,
                    "block_type": "table",
                    "processor": self.processor_name
                })
                # convert rows into readable lines
                if headers:
                    for r in rows:
                        # align to headers
                        pairs = []
                        for i, cell in enumerate(r):
                            h = headers[i] if i < len(headers) else f"col{i}"
                            pairs.append(f"{h}: {cell}")
                        text = "\n".join(pairs)
                        meta = dict(meta_base)
                        meta["table_headers"] = headers
                        chunks.append(Chunk(text=text, metadata=ensure_keys(meta)))
                else:
                    # flatten row text
                    for r in rows:
                        text = " | ".join(r)
                        meta = dict(meta_base)
                        chunks.append(Chunk(text=text, metadata=ensure_keys(meta)))

        # assign chunk ids and ingestion date
        for ch in chunks:
            ch.metadata["file_name"] = ch.metadata.get("file_name") or base_meta.get("file_name")
            ch.metadata["ingestion_date"] = now_iso()
            ch.metadata["chunk_id"] = ch.metadata.get("chunk_id") or make_chunk_id(ch.metadata["file_name"], ch.metadata, ch.text)
        return chunks


# ----------------------------
# Document Preprocessor - orchestrates loaders + preprocessors
# ----------------------------
class DocumentPreprocessor:
    """
    Orchestrator to process files and return Chunk lists ready for embedding/ingestion.

    Usage:
        pre = DocumentPreprocessor(qdrant_client=..., embedding_fn=..., batch_size=128)
        chunks = pre.process_file("/path/to/TR-547.pdf")
        pre.ingest_chunks(chunks)  # optional: direct to Qdrant
    """
    def __init__(self,
                 html_processor: Optional[HTMLPreprocessor] = None,
                 excel_loader: Optional[ExcelLoader] = None,
                 pdf_loader: Optional[DoclingPDFLoader] = None,
                 docx_loader: Optional[DOCXLoader] = None,
                 qdrant_client: Optional[QdrantClient] = None,
                 collection_name: str = "tapi_rag",
                 embedding_version: str = "local-embed-v1",
                 batch_size: int = 256):
        self.html_processor = html_processor or HTMLPreprocessor()
        self.excel_loader = excel_loader or ExcelLoader()
        self.pdf_loader = pdf_loader or (DoclingPDFLoader() if DocumentConverter else None)
        self.docx_loader = docx_loader or (DOCXLoader() if mammoth else None)
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_version = embedding_version
        self.batch_size = batch_size

        # in-memory dedupe for this run
        self._seen_fps = set()

    def _normalize_base_meta(self, file_path: str, source: str) -> Dict[str, Any]:
        return {
            "source": source,
            "file_name": os.path.basename(file_path),
            "processor": "DocumentPreprocessor",
            "embedding_version": self.embedding_version
        }

    def process_pdf(self, file_path: str) -> List[Chunk]:
        if self.pdf_loader is None:
            raise RuntimeError("PDF loader not configured (Docling missing).")
        elements = self.pdf_loader.load(file_path)
        chunks: List[Chunk] = []
        base_meta = self._normalize_base_meta(file_path, "pdf")
        # elements are dicts with type/text/page/raw
        for el in elements:
            t = el.get("type", "text")
            text = el.get("text") or ""
            page = el.get("page")
            if t == "table" or ("table" in (el.get("raw") or {}).get("type", "")):
                # if docling provides table raw, convert to html string and feed to html preprocessor
                html = el.get("raw", {}).get("html") or el.get("text", "")
                chunks.extend(self.html_processor.process_html(html, {**base_meta, "page": page}))
            elif t.startswith("heading"):
                # emit heading chunk
                meta = {**base_meta, "page": page, "block_type": "heading", "section": text}
                c = Chunk(text=f"[HEADING] {text}", metadata=ensure_keys(meta))
                chunks.append(c)
            elif t in ("paragraph", "text"):
                meta = {**base_meta, "page": page, "block_type": "paragraph"}
                c = Chunk(text=text.strip(), metadata=ensure_keys(meta))
                chunks.append(c)
            elif t == "html":
                chunks.extend(self.html_processor.process_html(text, {**base_meta, "page": page}))
            else:
                # fallback - small text
                if text and len(text.strip()) > 10:
                    meta = {**base_meta, "page": page, "block_type": "other"}
                    chunks.append(Chunk(text=text.strip(), metadata=ensure_keys(meta)))
        # finalize chunk IDs and ingestion time
        for ch in chunks:
            ch.metadata["ingestion_date"] = ch.metadata.get("ingestion_date") or now_iso()
            ch.metadata["chunk_id"] = ch.metadata.get("chunk_id") or make_chunk_id(ch.metadata["file_name"], ch.metadata, ch.text)
        return chunks

    def process_docx(self, file_path: str) -> List[Chunk]:
        if self.docx_loader is None:
            raise RuntimeError("DOCX loader not configured.")
        elements = self.docx_loader.load(file_path)
        chunks: List[Chunk] = []
        base_meta = self._normalize_base_meta(file_path, "docx")
        for el in elements:
            if el["type"] == "html":
                chunks.extend(self.html_processor.process_html(el["text"], base_meta))
        for ch in chunks:
            ch.metadata["ingestion_date"] = ch.metadata.get("ingestion_date") or now_iso()
            ch.metadata["chunk_id"] = ch.metadata.get("chunk_id") or make_chunk_id(ch.metadata["file_name"], ch.metadata, ch.text)
        return chunks

    def process_excel(self, file_path: str) -> List[Chunk]:
        elements = self.excel_loader.load(file_path)
        chunks: List[Chunk] = []
        base_meta = self._normalize_base_meta(file_path, "excel")
        for el in elements:
            if el["type"] == "excel_row":
                row = el["row"]
                # map column names to normalized metadata keys where possible
                sheet = row.get("_sheet")
                # heuristics for common column names
                object_name = row.get("Object") or row.get("object") or row.get("OBJECT") or row.get("Object Name") or ""
                attribute = row.get("Attribute") or row.get("attribute") or row.get("ATTRIBUTE") or ""
                xpath = row.get("Full XPath") or row.get("FullXpath") or row.get("xpath") or ""
                notes = row.get("Notes") or row.get("notes") or row.get("NOTE") or ""
                source_col = row.get("Source") or row.get("Yang File") or row.get("source") or ""
                meta = dict(base_meta)
                meta.update({
                    "usecase": sheet,
                    "object": object_name,
                    "attribute": attribute,
                    "xpath": xpath,
                    "section": sheet,
                    "block_type": "excel_row",
                    "source": "excel",
                    "file_name": os.path.basename(file_path),
                    "processor": "ExcelLoader-v1",
                })
                text_parts = [
                    f"[USECASE] {sheet}",
                    f"[OBJECT] {object_name}",
                    f"[ATTRIBUTE] {attribute}",
                    f"[XPATH] {xpath}",
                    f"[SOURCE] {source_col}",
                    f"[NOTES] {notes}"
                ]
                text = "\n".join(p for p in text_parts if p)
                meta = ensure_keys(meta)
                meta["ingestion_date"] = now_iso()
                meta["chunk_id"] = make_chunk_id(meta["file_name"], meta, text)
                chunks.append(Chunk(text=text, metadata=meta))
        return chunks

    def process_file(self, file_path: str) -> List[Chunk]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".pdf"]:
            return self.process_pdf(file_path)
        if ext in [".docx", ".doc"]:
            return self.process_docx(file_path)
        if ext in [".xlsx", ".xls"]:
            return self.process_excel(file_path)
        raise RuntimeError(f"Unsupported file extension: {ext}")

    # ----------------------------
    # chunk utilities
    # ----------------------------
    def dedupe_chunks(self, chunks: Iterable[Chunk]) -> List[Chunk]:
        out = []
        for ch in chunks:
            fp = fingerprint(ch.text + json.dumps(ch.metadata, sort_keys=True))
            if fp in self._seen_fps:
                continue
            self._seen_fps.add(fp)
            out.append(ch)
        return out

    def chunk_batch(self, chunks: Iterable[Chunk], batch_size: int = None) -> Iterable[List[Chunk]]:
        bs = batch_size or self.batch_size
        it = iter(chunks)
        while True:
            batch = list(itertools.islice(it, bs))
            if not batch:
                break
            yield batch

    # ----------------------------
    # Qdrant ingestion helper
    # ----------------------------
    def ingest_to_qdrant(self, chunks: List[Chunk], qdrant_client: Optional[QdrantClient] = None,
                         collection_name: Optional[str] = None, embeddings_fn=None, ids_prefix: str = None):
        """
        Ingest chunks into Qdrant.
        - qdrant_client: QdrantClient instance (if None, uses self.qdrant_client)
        - embeddings_fn: callable(texts: List[str]) -> List[List[float]]
        - returns list of inserted ids
        """
        client = qdrant_client or self.qdrant_client
        if client is None:
            raise RuntimeError("Qdrant client not provided.")

        coll = collection_name or self.collection_name

        # create collection if not exists (simple config)
        try:
            client.get_collection(coll)
        except Exception:
            logger.info("Creating qdrant collection %s", coll)
            # vector_size must match embedding dim; if embeddings_fn unavailable we skip creating
            # user should ensure the collection exists if using external embedding service
            try:
                dim = embeddings_fn(["test"])[0] and len(embeddings_fn(["test"])[0])
            except Exception:
                dim = 1536  # fallback guess; user should set correctly
            client.recreate_collection(coll_name=coll, vectors_config={"size": dim, "distance": Distance.COSINE})

        inserted_ids = []
        for batch in self.chunk_batch(chunks):
            texts = [c.text for c in batch]
            metas = [c.metadata for c in batch]
            # compute embeddings
            if embeddings_fn is None:
                raise RuntimeError("embeddings_fn callable required to compute vector representations.")
            embeddings = embeddings_fn(texts)
            # generate deterministic ids if desired
            ids = []
            for i, c in enumerate(batch):
                cid = c.metadata.get("chunk_id") or (ids_prefix or "") + str(uuid.uuid4())
                ids.append(cid)
            # upsert via qdrant client (native)
            # qdrant-client expects points with id, vector, payload
            points = []
            for _id, emb, meta, text in zip(ids, embeddings, metas, texts):
                payload = dict(meta)
                payload["text_preview"] = text[:400]
                points.append({"id": _id, "vector": emb, "payload": payload})
            # insert
            client.upsert(coll_name=coll, points=points)
            inserted_ids.extend(ids)
            logger.info("Inserted %d points into %s", len(points), coll)
        return inserted_ids
