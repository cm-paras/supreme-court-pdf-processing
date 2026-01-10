"""
Document chunking functionality
"""
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Handles document chunking for search indexing"""
    
    def __init__(self, config):
        """
        Initialize document chunker
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, text, metadata, blob_name=None):
        """
        Split a document into chunks with metadata
        
        Args:
            text: Document text
            metadata: Document metadata
            blob_name: Optional blob name
            
        Returns:
            list: List of document chunks with metadata
        """
        if not text:
            return []
        
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        document_chunks = []
        identifier = self._get_identifier(metadata, blob_name)
        
        for i, chunk_text in enumerate(chunks):
            doc_id_str = f"{identifier}_chunk_{i}"
            chunk_id = base64.urlsafe_b64encode(doc_id_str.encode('utf-8')).decode('utf-8')
            
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["chunk_total"] = len(chunks)
            chunk_metadata["document_id"] = identifier
            
            document_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata,
                "blob_name": blob_name
            })
        
        return document_chunks
    
    def _get_identifier(self, metadata, blob_name):
        """Get document identifier for chunk IDs"""
        # Handle existing Cosmos DB metadata format
        identifier = metadata.get("Case Number", "Unknown")
        if identifier in [None, "", "Unknown"]:
            identifier = metadata.get("Case Name", "Unknown")
        if identifier in [None, "", "Unknown"] and blob_name:
            identifier = blob_name
        
        # Sanitize identifier
        identifier = re.sub(r'[^\w\-]', '_', identifier)
        return identifier
    
    def chunk_batch(self, documents):
        """
        Process multiple documents into chunks in parallel
        
        Args:
            documents: List of documents with text and metadata
            
        Returns:
            list: All chunks from all documents
        """
        all_chunks = []
        
        with ThreadPoolExecutor(
            max_workers=min(len(documents), self.config.MAX_WORKERS)
        ) as executor:
            future_to_doc = {}
            for doc in documents:
                if doc.get("success", False) and doc.get("text") and doc.get("metadata"):
                    future = executor.submit(
                        self.chunk_document,
                        doc["text"],
                        doc["metadata"],
                        doc.get("blob_name")
                    )
                    future_to_doc[future] = doc["blob_name"]
            
            for future in as_completed(future_to_doc):
                blob_name = future_to_doc[future]
                try:
                    chunks = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                        logger.info(f"Created {len(chunks)} chunks for {blob_name}")
                except Exception as e:
                    logger.error(f"Error chunking document {blob_name}: {e}")
        
        return all_chunks
