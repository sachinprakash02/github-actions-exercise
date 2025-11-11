from qdrant_client import QdrantClient
from src.preprocessing.document_preprocessor import DocumentPreprocessor

# instantiate Qdrant client (local)
qc = QdrantClient(url="http://localhost:7333")

# create preprocessor with qdrant client
pre = DocumentPreprocessor(qdrant_client=qc, collection_name="tapi_rag", embedding_version="nomic-v1")

# 1) process pdf/docx/excel into chunks
chunks = pre.process_file("data/TR-547.pdf")
excel_chunks = pre.process_file("data/tapi_usecases.xlsx")

# 2) dedupe
chunks = pre.dedupe_chunks(chunks)
excel_chunks = pre.dedupe_chunks(excel_chunks)

# 3) provide embedding function (example using your embedding model wrapper)
def my_embedder(texts):
    # return list of vectors (list[float]) for each input text
    return [get_embedding_for(t) for t in texts]

# 4) ingest
pre.ingest_to_qdrant(chunks + excel_chunks, embeddings_fn=my_embedder, ids_prefix="tr547:")
