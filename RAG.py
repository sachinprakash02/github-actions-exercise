# enhanced_pymupdf_parser.py
import os
import re
import fitz  # PyMuPDF
import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict
import json
from typing import List, Dict, Any

class EnhancedPyMuPDFParser:
    """
    Enhanced PDF parser using PyMuPDF4LLM with robust section detection
    even when TOC is empty.
    """
    
    def __init__(self):
        self.heading_patterns = [
            r'^(#{1,6})\s+(.+)',  # Markdown headers: #, ##, ###
            r'^(CHAPTER|Chapter|\d+\.\d+)\s+(.+)',  # Chapter patterns
            r'^(\d+\.)\s+(.+)',  # Numbered sections: 1., 2.
            r'^([A-Z][A-Z\s]{10,})$',  # ALL CAPS headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$',  # Title Case with colon
        ]
    
    def parse_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Main method to parse PDF and create chunks with rich metadata.
        """
        print(f"Parsing PDF: {pdf_path}")
        
        # Step 1: Extract raw documents with page chunks
        raw_documents = self._extract_with_pymupdf(pdf_path)
        if not raw_documents:
            print("No documents extracted from PDF")
            return []
        
        # Step 2: Extract headings from content (fallback when TOC is empty)
        sections_by_page = self._extract_headings_from_content(raw_documents)
        
        # Step 3: Enhance documents with section information
        enhanced_documents = self._enhance_with_sections(raw_documents, sections_by_page)
        
        # Step 4: Create semantic chunks
        chunks = self._create_semantic_chunks(enhanced_documents, chunk_size, chunk_overlap)
        
        print(f"Successfully created {len(chunks)} chunks from {len(raw_documents)} pages")
        return chunks
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract content using PyMuPDF4LLM with page chunks.
        """
        try:
            # First try to get TOC using PyMuPDF directly
            toc = []
            try:
                doc = fitz.open(pdf_path)
                toc = doc.get_toc()
                doc.close()
                print(f"TOC entries found: {len(toc)}")
            except Exception as e:
                print(f"Could not extract TOC: {e}")
            
            # Parse with page chunks
            chunked_data = pymupdf4llm.to_markdown(
                doc=pdf_path,
                page_chunks=True
            )
            
            documents = []
            for i, chunk in enumerate(chunked_data):
                page_num = i + 1
                content = chunk.get('text', '').strip()
                
                # Skip empty or very short chunks
                if len(content) < 10:
                    continue
                
                document = {
                    "content": content,
                    "metadata": {
                        # Tier 1: Essential
                        "page_number": page_num,
                        "section_title": "Introduction",  # Default, will be updated
                        "source_document": os.path.basename(pdf_path),
                        "content_type": self._detect_content_type(content),
                        
                        # Tier 2: Important
                        "document_type": "pdf",
                        "chunk_sequence": i,
                        "keywords": self._extract_keywords(content),
                        "entities": self._extract_entities(content),
                        
                        # Processing info
                        "parser": "pymupdf4llm",
                        "chunk_size": len(content),
                        "has_toc": len(toc) > 0
                    }
                }
                documents.append(document)
            
            return documents
            
        except Exception as e:
            print(f"Error parsing PDF with PyMuPDF4LLM: {e}")
            return []
    
    def _extract_headings_from_content(self, documents: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract headings directly from document content when TOC is empty.
        Returns dict mapping page numbers to list of headings on that page.
        """
        sections_by_page = defaultdict(list)
        
        for doc in documents:
            page_num = doc['metadata']['page_number']
            content = doc['content']
            
            # Split content into lines and look for heading patterns
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                heading_info = self._detect_heading(line)
                if heading_info:
                    level, title = heading_info
                    sections_by_page[page_num].append({
                        'level': level,
                        'title': title,
                        'type': 'extracted'
                    })
        
        print(f"Extracted {sum(len(v) for v in sections_by_page.values())} headings from content")
        return sections_by_page
    
    def _detect_heading(self, line: str) -> tuple[int, str] | None:
        """
        Detect if a line is a heading and return (level, title).
        """
        line = line.strip()
        
        # Markdown headers (#, ##, ###)
        match = re.match(r'^(#{1,6})\s+(.+)', line)
        if match:
            level = len(match.group(1))  # Number of # determines level
            return level, match.group(2).strip()
        
        # Numbered sections (1., 1.1, 2.3.1, etc.)
        match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.+)', line)
        if match:
            numbers = match.group(1).split('.')
            level = len(numbers)
            return level, match.group(2).strip()
        
        # ALL CAPS headings (likely main titles)
        if line.isupper() and len(line) > 5 and len(line) < 100:
            return 1, line
        
        # Title case with colon
        if re.match(r'^[A-Z][a-zA-Z\s]+:$', line) and len(line) < 80:
            return 2, line.rstrip(':')
        
        # Check if line is significantly different from surrounding text
        words = line.split()
        if (len(words) < 10 and 
            any(word.istitle() for word in words) and 
            len(line) < 200):
            return 3, line
        
        return None
    
    def _enhance_with_sections(self, 
                             documents: List[Dict[str, Any]], 
                             sections_by_page: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Enhance documents with proper section titles based on extracted headings.
        """
        current_section = "Introduction"
        current_level = 1
        
        enhanced_docs = []
        
        for doc in documents:
            page_num = doc['metadata']['page_number']
            
            # Check if this page has new sections
            if page_num in sections_by_page:
                page_sections = sections_by_page[page_num]
                
                # Use the most prominent section (lowest level number)
                if page_sections:
                    main_section = min(page_sections, key=lambda x: x['level'])
                    current_section = main_section['title']
                    current_level = main_section['level']
            
            # Update document with current section
            enhanced_doc = doc.copy()
            enhanced_doc['metadata'] = doc['metadata'].copy()
            enhanced_doc['metadata']['section_title'] = current_section
            enhanced_doc['metadata']['section_level'] = current_level
            
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs
    
    def _create_semantic_chunks(self, 
                              documents: List[Dict[str, Any]], 
                              chunk_size: int = 1000, 
                              chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Create smart chunks that respect section boundaries.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n\n# ", "\n\n", "\n", ". ", " ", ""]
        )
        
        all_chunks = []
        
        # Group by section first
        sections = defaultdict(list)
        for doc in documents:
            section_title = doc['metadata']['section_title']
            sections[section_title].append(doc)
        
        # Chunk within each section
        chunk_counter = 0
        for section_title, section_docs in sections.items():
            section_content = "\n\n".join([doc['content'] for doc in section_docs])
            base_metadata = section_docs[0]['metadata'].copy()
            
            # Split section content
            chunks = text_splitter.split_text(section_content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"chunk_{chunk_counter:04d}",
                    "chunk_sequence": i,
                    "total_chunks_in_section": len(chunks),
                    "is_section_boundary": i == 0 or i == len(chunks) - 1,
                    "final_chunk_size": len(chunk),
                    "token_estimate": len(chunk.split())  # Rough token estimate
                })
                
                all_chunks.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
                chunk_counter += 1
        
        return all_chunks
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type based on patterns."""
        lines = text.strip().split('\n')
        
        # Table detection
        if any(re.search(r'^\|.*\|$', line) for line in lines[:3]):
            return "table"
        
        # Code detection
        if re.search(r'[{};=><+\-*/]', text) and len(text.split()) < 100:
            return "code"
        
        # Heading detection
        if len(text.split()) < 15:
            return "heading"
        
        # List detection
        if re.match(r'^[\-\*•]\s', text.strip()):
            return "list"
        
        return "text"
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract keywords based on frequency."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 
            'this', 'that', 'these', 'those', 'it', 'its', 'be', 'been'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq, key=word_freq.get, reverse=True)[:top_n]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction."""
        entities = []
        
        # Proper nouns (capitalized, multi-word)
        potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter common false positives
        common_false = ['This', 'That', 'These', 'Those', 'The', 'A', 'An']
        entities = [entity for entity in potential_entities if entity not in common_false]
        
        return entities[:8]
    
    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save chunks to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to: {output_path}")
    
    def print_detailed_summary(self, chunks: List[Dict[str, Any]]):
        """Print detailed summary of parsing results."""
        print(f"\n=== PARSING SUMMARY ===")
        print(f"Total chunks created: {len(chunks)}")
        
        # Count by section
        sections = defaultdict(int)
        content_types = defaultdict(int)
        for chunk in chunks:
            section = chunk['metadata']['section_title']
            content_type = chunk['metadata']['content_type']
            sections[section] += 1
            content_types[content_type] += 1
        
        print(f"\nSections found: {len(sections)}")
        for section, count in list(sections.items())[:10]:  # Show top 10
            print(f"  {section}: {count} chunks")
        
        print(f"\nContent types:")
        for content_type, count in content_types.items():
            print(f"  {content_type}: {count}")
        
        # Show chunk size distribution
        sizes = [len(chunk['content']) for chunk in chunks]
        print(f"\nChunk size stats:")
        print(f"  Min: {min(sizes)} chars")
        print(f"  Max: {max(sizes)} chars")
        print(f"  Avg: {sum(sizes)/len(sizes):.0f} chars")
    
    def print_chunk_samples(self, chunks: List[Dict[str, Any]], num_samples: int = 3):
        """Print sample chunks for inspection."""
        print(f"\n=== SAMPLE CHUNKS ({num_samples} of {len(chunks)}) ===")
        
        for i, chunk in enumerate(chunks[:num_samples]):
            print(f"\n--- Chunk {i+1} ---")
            content_preview = chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
            print(f"Content: {content_preview}")
            print("Metadata:")
            for key, value in chunk['metadata'].items():
                print(f"  {key}: {value}")

# Usage example
def main():
    parser = EnhancedPyMuPDFParser()
    
    # Parse PDF
    chunks = parser.parse_pdf(
        pdf_path="your_document.pdf",  # Replace with your PDF path
        chunk_size=1000,
        chunk_overlap=200
    )
    
    if chunks:
        # Print summaries
        parser.print_detailed_summary(chunks)
        parser.print_chunk_samples(chunks, num_samples=2)
        
        # Save to JSON
        parser.save_chunks_to_json(chunks, "enhanced_chunks.json")
        
        print(f"\n✅ Success! Created {len(chunks)} chunks with enhanced section detection.")
    else:
        print("❌ No chunks were created. Please check the PDF file.")

if __name__ == "__main__":
    main()
