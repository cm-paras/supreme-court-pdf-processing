"""Main PDF processing pipeline."""

import os
import logging
from typing import List, Optional

from ..config.config import Config
from ..clients.azure_clients import AzureClientManager
from ..processors.pdf_downloader import PDFDownloader
from ..processors.text_extractor import TextExtractor
from ..processors.metadata_extractor import MetadataExtractor
from ..processors.document_chunker import DocumentChunker
from ..processors.embedding_generator import EmbeddingGenerator
from ..storage.cosmos_storage import CosmosStorage
from ..storage.search_indexer import SearchIndexer

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Sequential PDF processing pipeline."""
    
    def __init__(self):
        self.config = Config()
        self.config.validate()
        
        self.azure_clients = AzureClientManager(self.config)
        
        self.downloader = PDFDownloader(self.azure_clients, self.config)
        self.text_extractor = TextExtractor(self.config)
        self.metadata_extractor = MetadataExtractor(self.azure_clients, self.config)
        self.chunker = DocumentChunker(self.config)
        self.embedding_generator = EmbeddingGenerator(self.azure_clients, self.config)
        self.storage = CosmosStorage(self.azure_clients, self.config)
        self.indexer = SearchIndexer(self.azure_clients, self.config)
    
    def process_single_pdf(self, blob_url: str, pdf_id: str) -> bool:
        """Process a single PDF through the complete pipeline."""
        logger.info(f"Starting processing for PDF {pdf_id}")
        
        try:
            # Download PDF
            local_paths = self.downloader.download_batch([blob_url])
            if not local_paths:
                logger.error(f"Failed to download PDF {pdf_id}")
                return False
            
            # Extract text
            texts = self.text_extractor.extract_batch(local_paths)
            if not texts:
                logger.error(f"Failed to extract text from PDF {pdf_id}")
                return False
            
            # Extract metadata
            metadata = self.metadata_extractor.extract_batch(texts)
            if not metadata:
                logger.error(f"Failed to extract metadata from PDF {pdf_id}")
                return False
            
            # Store in Cosmos DB
            blob_name = list(metadata.keys())[0]
            text_sample = list(texts.values())[0][:1000]
            
            success = self.storage.store_document(
                blob_name,
                list(metadata.values())[0],
                text_sample
            )
            
            if not success:
                logger.error(f"Failed to store document {pdf_id} in Cosmos DB")
                return False
            
            # Chunk and index using the extracted metadata
            documents = [{
                "blob_name": blob_name,
                "success": True,
                "metadata": list(metadata.values())[0],
                "text": list(texts.values())[0]
            }]
            
            chunks = self.chunker.chunk_batch(documents)
            if chunks:
                chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
                succeeded, failed = self.indexer.upload_chunks(chunks_with_embeddings)
                
                if succeeded > 0:
                    logger.info(f"Successfully processed PDF {pdf_id}: {succeeded} chunks indexed")
                    return True
            
            logger.error(f"Failed to index chunks for PDF {pdf_id}")
            return False
            
        except Exception as e:
            logger.error(f"Processing failed for PDF {pdf_id}: {e}")
            return False
        finally:
            # Clean up downloaded PDF files
            try:
                if 'local_paths' in locals() and local_paths:
                    for path in local_paths.values():
                        if path and os.path.exists(path):
                            os.remove(path)
                            logger.debug(f"Deleted temporary file: {path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary files for {pdf_id}: {cleanup_error}")
    

    
    def process_batch(self, pdf_urls: List[tuple], max_pdfs: Optional[int] = None, mode: str = "full") -> dict:
        """Process multiple PDFs sequentially.

        Args:
            pdf_urls: List of (blob_url, pdf_id) tuples.
            max_pdfs: Optional limit on number of PDFs to process.
            mode: "metadata" for metadata-only, "full" for full processing.
        """
        results = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }

        logger.info(f"Processing mode: {mode}")
        
        pdfs_to_process = pdf_urls[:max_pdfs] if max_pdfs else pdf_urls
        blob_urls = [url for url, _ in pdfs_to_process]

        # Batch check for existing documents in Cosmos DB
        existing_in_cosmos = self.storage.documents_exist_batch(blob_urls)
        logger.info(f"{len(existing_in_cosmos)} PDFs already have metadata in Cosmos DB")

        # Batch check for indexed documents in Azure Search
        try:
            indexed_in_search = self.indexer.documents_indexed_batch(blob_urls)
            logger.info(f"{len(indexed_in_search)} PDFs already indexed in Azure Search")
        except Exception as e:
            logger.warning(f"Batch index check failed, falling back to per-document check: {e}")
            indexed_in_search = set()

        for blob_url, pdf_id in pdfs_to_process:
            results['total'] += 1

            # Skip if already processed
            if blob_url in existing_in_cosmos:
                if blob_url in indexed_in_search:
                    logger.info(f"Skipping already processed PDF for blob_url: {blob_url}")
                    results['skipped'] += 1
                    continue
                else:
                    logger.info(f"Document {blob_url} found in Cosmos DB but not indexed. Indexing now...")
                    try:
                        existing_doc = self.storage.get_document_by_blob_name(blob_url)
                        if existing_doc:
                            # Use existing metadata from Cosmos DB
                            cosmos_metadata = existing_doc.get("metadata", {})
                            
                            # Download PDF to get full text for chunking
                            local_paths = self.downloader.download_batch([blob_url])
                            if local_paths:
                                texts = self.text_extractor.extract_batch(local_paths)
                                if texts:
                                    full_text = list(texts.values())[0]
                                    documents = [{
                                        "blob_name": blob_url,
                                        "success": True,
                                        "metadata": cosmos_metadata,
                                        "text": full_text
                                    }]
                                    chunks = self.chunker.chunk_batch(documents)
                                    if chunks:
                                        chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
                                        succeeded, failed = self.indexer.upload_chunks(chunks_with_embeddings)
                                        if succeeded > 0:
                                            logger.info(f"Indexed existing document {blob_url} successfully.")
                                            results['skipped'] += 1
                                            # Clean up
                                            for path in local_paths.values():
                                                if path and os.path.exists(path):
                                                    os.remove(path)
                                            continue
                                        else:
                                            logger.warning(f"Failed to index existing document {blob_url}. Proceeding to reprocess.")
                                    # Clean up
                                    for path in local_paths.values():
                                        if path and os.path.exists(path):
                                            os.remove(path)
                                else:
                                    logger.warning(f"Failed to extract text for existing document {blob_url}. Proceeding to reprocess.")
                            else:
                                logger.warning(f"Failed to download existing document {blob_url}. Proceeding to reprocess.")
                        else:
                            logger.warning(f"Document {blob_url} metadata not found in Cosmos DB. Proceeding to reprocess.")
                    except Exception as e:
                        logger.error(f"Error indexing existing document {blob_url}: {e}")
            
            try:
                if mode == "metadata":
                    # Metadata-only mode: skip indexing
                    logger.info(f"Processing {pdf_id} in metadata-only mode")
                    local_paths = self.downloader.download_batch([blob_url])
                    if not local_paths:
                        logger.error(f"Failed to download PDF {pdf_id}")
                        results['failed'] += 1
                        continue

                    texts = self.text_extractor.extract_batch(local_paths)
                    if not texts:
                        logger.error(f"Failed to extract text from PDF {pdf_id}")
                        results['failed'] += 1
                        continue

                    metadata = self.metadata_extractor.extract_batch(texts)
                    if not metadata:
                        logger.error(f"Failed to extract metadata from PDF {pdf_id}")
                        results['failed'] += 1
                        continue

                    blob_name = list(metadata.keys())[0]
                    text_sample = list(texts.values())[0][:1000]
                    success = self.storage.store_document(
                        blob_name,
                        list(metadata.values())[0],
                        text_sample
                    )
                    if success:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1

                    # Clean up
                    for path in local_paths.values():
                        if path and os.path.exists(path):
                            os.remove(path)
                    continue

                # Full processing mode
                success = self.process_single_pdf(blob_url, pdf_id)
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1

            except Exception as e:
                logger.error(f"Unexpected error processing {pdf_id}: {e}")
                results['failed'] += 1
        
        logger.info(f"Batch processing complete: {results}")
        return results
