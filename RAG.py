import os
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Alternative
from langchain_core.documents import Document
import json
from typing import List, Dict, Any

class TAPIProcessor:
    def __init__(self, embedding_model="text-embedding-3-small", chroma_persist_dir="./chroma_tapi_db"):
        self.embedding_model = embedding_model
        self.chroma_persist_dir = chroma_persist_dir
        self.vector_store = None
        
        # Initialize embeddings - choose one option below
        self.embeddings = self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Setup embedding model based on configuration"""
        # Option 1: OpenAI Embeddings (requires API key)
        os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # Set your key
        return OpenAIEmbeddings(model=self.embedding_model)
        
        # Option 2: Local HuggingFace Embeddings (no API required)
        # return HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     model_kwargs={'device': 'cpu'},  # or 'cuda' if you have GPU
        #     encode_kwargs={'normalize_embeddings': True}
        # )
    
    def process_document_to_chroma(self, document_path: str):
        """
        Complete pipeline: Process document → Chunk → Create embeddings → Store in Chroma
        """
        print("🚀 Starting TAPI TR-547 processing pipeline...")
        
        # Step 1: Process document with Docling and create chunks
        chunks = self._process_with_docling(document_path)
        
        # Step 2: Create Chroma vector store with embeddings
        self._create_chroma_vector_store(chunks)
        
        # Step 3: Verify and test the retrieval
        self._verify_chroma_setup()
        
        print("✅ Processing complete! Chroma vector store is ready for queries.")
        return self.vector_store
    
    def _process_with_docling(self, document_path: str) -> List[Document]:
        """Process document using Docling and create chunks"""
        print("📄 Processing document with Docling...")
        
        # Initialize Docling converter
        pipeline_options = PipelineOptions()
        pipeline_options.ocr.enabled = True
        pipeline_options.table.enabled = True
        
        converter = DocumentConverter(pipeline_options=pipeline_options)
        result = converter.convert(document_path)
        
        # Extract document structure and create chunks
        document_structure = self._extract_document_structure(result.document)
        prepared_docs = self._prepare_documents_for_chunking(document_structure)
        chunks = self._apply_recursive_chunking(prepared_docs)
        
        print(f"📊 Created {len(chunks)} chunks from document")
        return chunks
    
    def _extract_document_structure(self, docling_doc) -> Dict[str, Any]:
        """Extract comprehensive document structure"""
        document_structure = {
            "document_metadata": {
                "title": docling_doc.title or "TAPI TR-547",
                "page_count": len(docling_doc.pages),
            },
            "sections": [],
            "tables": [],
        }
        
        # Extract sections
        for section in docling_doc.sections:
            section_data = self._extract_section_content(section)
            document_structure["sections"].append(section_data)
        
        # Extract tables
        for i, table in enumerate(docling_doc.tables):
            table_data = {
                "table_id": i,
                "rows": table.shape[0] if hasattr(table, 'shape') else 0,
                "columns": table.shape[1] if hasattr(table, 'shape') else 0,
                "caption": table.caption.text if table.caption else None,
                "content": self._format_table_for_embedding(table),
                "page_number": table.bbox.page_number if table.bbox else None,
            }
            document_structure["tables"].append(table_data)
        
        return document_structure
    
    def _extract_section_content(self, section, depth=0):
        """Extract content from document sections"""
        section_data = {
            "depth": depth,
            "title": section.title,
            "text": section.text or "",
            "text_length": len(section.text) if section.text else 0,
            "subsections": [],
        }
        
        for subsection in section.subsections:
            subsection_data = self._extract_section_content(subsection, depth + 1)
            section_data["subsections"].append(subsection_data)
        
        return section_data
    
    def _format_table_for_embedding(self, table) -> str:
        """Format table content for embedding"""
        try:
            if hasattr(table, 'df') and table.df is not None:
                # Convert DataFrame to readable text
                table_text = f"Table: {table.caption.text if table.caption else 'Data Table'}\n"
                for _, row in table.df.iterrows():
                    row_text = " | ".join([str(cell) for cell in row])
                    table_text += f"{row_text}\n"
                return table_text
            else:
                return str(table)
        except:
            return f"Table content with {table.shape[0]} rows and {table.shape[1]} columns"
    
    def _prepare_documents_for_chunking(self, document_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare documents for text chunking"""
        documents = []
        
        # Process sections
        for section in document_structure["sections"]:
            section_docs = self._create_section_documents(section)
            documents.extend(section_docs)
        
        # Process tables
        for table in document_structure["tables"]:
            table_doc = {
                "content": table["content"],
                "metadata": {
                    "type": "table",
                    "table_id": table["table_id"],
                    "rows": table["rows"],
                    "columns": table["columns"],
                    "page_number": table["page_number"],
                    "source": "TAPI_TR-547"
                }
            }
            documents.append(table_doc)
        
        return documents
    
    def _create_section_documents(self, section: Dict) -> List[Dict[str, Any]]:
        """Create document chunks from section content"""
        documents = []
        
        if section["text"] and len(section["text"].strip()) > 0:
            section_doc = {
                "content": f"Section: {section['title']}\n\n{section['text']}",
                "metadata": {
                    "type": "section",
                    "section_title": section["title"],
                    "depth": section["depth"],
                    "text_length": section["text_length"],
                    "source": "TAPI_TR-547"
                }
            }
            documents.append(section_doc)
        
        for subsection in section["subsections"]:
            subsection_docs = self._create_section_documents(subsection)
            documents.extend(subsection_docs)
        
        return documents
    
    def _apply_recursive_chunking(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Apply recursive character text splitting"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for technical content
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        final_chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Only chunk large text sections, keep tables intact
            if metadata["type"] == "section" and len(content) > 600:
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk)
                    })
                    
                    final_chunks.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
            else:
                # Keep tables and small sections as single chunks
                final_chunks.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        return final_chunks
    
    def _create_chroma_vector_store(self, chunks: List[Document]):
        """Create Chroma vector store with embeddings"""
        print("🔮 Creating embeddings and storing in Chroma...")
        
        # Create Chroma vector store from documents
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_persist_dir,
            collection_name="tapi_tr_547",
            collection_metadata={"description": "TAPI TR-547 Technical Documentation"}
        )
        
        print(f"💾 Vector store persisted to: {self.chroma_persist_dir}")
        print(f"📚 Collection: tapi_tr_547")
        print(f"📄 Documents stored: {len(chunks)}")
    
    def _verify_chroma_setup(self):
        """Verify the Chroma setup is working"""
        print("🔍 Verifying Chroma setup...")
        
        # Test retrieval
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        test_queries = [
            "TAPI architecture",
            "YANG model", 
            "network configuration"
        ]
        
        for query in test_queries:
            try:
                results = retriever.get_relevant_documents(query)
                print(f"✅ Query '{query}': Found {len(results)} results")
            except Exception as e:
                print(f"❌ Query '{query}' failed: {e}")
    
    def get_retriever(self, search_type="similarity", k=5, score_threshold=0.7):
        """Get a retriever for querying the vector store"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call process_document_to_chroma first.")
        
        search_kwargs = {"k": k}
        
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def query_document(self, query: str, k: int = 5):
        """Query the processed document"""
        retriever = self.get_retriever(k=k)
        results = retriever.get_relevant_documents(query)
        
        print(f"\n🔎 Query: '{query}'")
        print(f"📋 Found {len(results)} relevant chunks:\n")
        
        for i, doc in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"Type: {doc.metadata.get('type', 'N/A')}")
            print(f"Section: {doc.metadata.get('section_title', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...")
            print()

def main():
    """Main execution function"""
    # Configuration
    document_path = "tapi_tr_547.pdf"  # Update with your file path
    
    try:
        # Initialize processor
        processor = TAPIProcessor(
            embedding_model="text-embedding-3-small",  # or "text-embedding-3-large"
            chroma_persist_dir="./chroma_tapi_tr547"
        )
        
        # Process document and create embeddings
        processor.process_document_to_chroma(document_path)
        
        # Example queries
        print("\n🧪 Testing with sample queries:")
        test_queries = [
            "What is TAPI architecture?",
            "Explain YANG models in TAPI",
            "How does network configuration work?",
            "TAPI service interfaces"
        ]
        
        for query in test_queries:
            processor.query_document(query, k=3)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
