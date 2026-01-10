"""Azure Cognitive Search indexer."""

import logging
from typing import List, Dict, Any
from datetime import datetime
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from ..config.settings import AZURE_CONFIG
from ..models.document import Document, Chunk, ChunkStatus
from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

class SearchIndexer:
    """Indexes validated chunks in Azure Cognitive Search."""
    
    def __init__(self):
        self.client = SearchClient(
            endpoint=AZURE_CONFIG.SEARCH_ENDPOINT,
            index_name=AZURE_CONFIG.SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_CONFIG.SEARCH_KEY)
        )
    
    def index_chunks(self, document: Document, chunks: List[Chunk]) -> List[str]:
        """Index valid chunks in Azure Search."""
        document.processing_timestamps.indexing_start = datetime.utcnow()
        
        try:
            # Filter chunks ready for indexing
            valid_chunks = [
                chunk for chunk in chunks 
                if (chunk.status == ChunkStatus.EMBEDDED and 
                    len(chunk.text.strip()) >= 150 and
                    len(chunk.embedding_vector) > 0)
            ]
            
            if not valid_chunks:
                logger.warning(f"No valid chunks to index for document {document.pdf_id}")
                return []
            
            logger.info(f"Indexing {len(valid_chunks)} chunks for document {document.pdf_id}")
            
            # Prepare documents for indexing
            search_documents = []
            for chunk in valid_chunks:
                search_doc = self._prepare_search_document(chunk)
                search_documents.append(search_doc)
            
            # Upload to search index
            indexed_chunk_ids = self._upload_documents(search_documents)
            
            document.processing_timestamps.indexing_end = datetime.utcnow()
            
            logger.info(f"Successfully indexed {len(indexed_chunk_ids)} chunks for document {document.pdf_id}")
            return indexed_chunk_ids
            
        except Exception as e:
            logger.error(f"Indexing failed for document {document.pdf_id}: {e}")
            document.processing_timestamps.indexing_end = datetime.utcnow()
            raise
    
    def _prepare_search_document(self, chunk: Chunk) -> Dict[str, Any]:
        """Prepare chunk for search index."""
        metadata = chunk.metadata or {}
        
        # Handle keywords - can be list or string
        keywords = metadata.get("Keywords", [])
        if not isinstance(keywords, list):
            try:
                import json
                keywords = json.loads(keywords) if isinstance(keywords, str) else []
            except:
                keywords = []
        
        return {
            "id": chunk.chunk_id,
            "pdf_id": chunk.pdf_id,
            "content": chunk.text,
            "content_vector": chunk.embedding_vector,
            "chunk_index": chunk.chunk_index,
            "chunk_total": chunk.chunk_total,
            "case_name": metadata.get("Case Name", ""),
            "case_number": metadata.get("Case Number", ""),
            "citation": metadata.get("Citation", ""),
            "date_of_judgment": metadata.get("Date of Judgment", ""),
            "bench": metadata.get("Bench", ""),
            "court": metadata.get("Court", ""),
            "summary": metadata.get("Summary", ""),
            "keywords": keywords,
            "petitioner_advocates": metadata.get("Petitioner Advocates", []),
            "respondent_advocates": metadata.get("Respondent Advocates", []),
            "created_at": datetime.utcnow().isoformat()
        }
    
    @retry_with_backoff()
    def _upload_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Upload documents to search index."""
        try:
            result = self.client.upload_documents(documents)
            
            successful_ids = []
            for item in result:
                if item.succeeded:
                    successful_ids.append(item.key)
                else:
                    logger.error(f"Failed to index document {item.key}: {item.error_message}")
            
            return successful_ids
            
        except Exception as e:
            logger.error(f"Search index upload failed: {e}")
            raise