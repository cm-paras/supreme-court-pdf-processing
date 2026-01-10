"""
PDF Processing Pipeline Orchestration
"""
import os
import time
from tqdm import tqdm

import logging
from src.clients import AzureClientManager
from src.processors import (
    PDFDownloader, TextExtractor, MetadataExtractor,
    DocumentChunker, EmbeddingGenerator
)
from src.storage import SearchIndexer, CosmosStorage

logger = logging.getLogger(__name__)


class PDFProcessingPipeline:
    """Orchestrates the PDF processing pipeline"""
    
    def __init__(self, config, server_count=1, server_number=0):
        """
        Initialize the pipeline
        
        Args:
            config: Configuration object
            server_count: Total number of servers
            server_number: Current server number (0-indexed)
        """
        config.validate()
        
        self.config = config
        self.server_count = server_count
        self.server_number = server_number
        
        # Initialize Azure clients
        self.azure_clients = AzureClientManager(config)
        
        # Initialize components
        self.downloader = PDFDownloader(self.azure_clients, config)
        self.text_extractor = TextExtractor(config)
        self.metadata_extractor = MetadataExtractor(self.azure_clients, config)
        self.chunker = DocumentChunker(config)
        self.embedding_generator = EmbeddingGenerator(self.azure_clients, config)
        self.search_indexer = SearchIndexer(self.azure_clients, config)
        self.cosmos_storage = CosmosStorage(self.azure_clients, config)
        
        logger.info(f"Pipeline initialized with {config.MAX_WORKERS} workers")
        logger.info(f"Server configuration: {self.server_number + 1}/{self.server_count}")
    
    def process_pdf_batch(self, blob_names):
        """
        Process a batch of PDFs through the complete pipeline
        
        Args:
            blob_names: List of blob names to process
            
        Returns:
            list: Results for each processed document
        """
        results = []
        
        try:
            # Stage 1: Download PDFs
            print(f"[Stage 1/4] Downloading {len(blob_names)} PDFs...")
            local_paths_dict = self.downloader.download_batch(blob_names)
            
            if not local_paths_dict:
                logger.warning("No PDFs were successfully downloaded")
                return results
            
            print(f"[Stage 1/4] Completed: {len(local_paths_dict)}/{len(blob_names)} PDFs downloaded")
            
            # Stage 2: Extract text
            print(f"[Stage 2/4] Extracting text from {len(local_paths_dict)} PDFs...")
            texts_dict = self.text_extractor.extract_batch(local_paths_dict)
            
            # Cleanup downloaded files
            for local_path in local_paths_dict.values():
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file {local_path}: {e}")
            
            if not texts_dict:
                logger.warning("No text was successfully extracted")
                return results
            
            print(f"[Stage 2/4] Completed: {len(texts_dict)}/{len(local_paths_dict)} PDFs text extracted")
            
            # Stage 3: Extract metadata
            print(f"[Stage 3/4] Extracting metadata from {len(texts_dict)} documents...")
            metadata_dict = self.metadata_extractor.extract_batch(texts_dict)
            
            if not metadata_dict:
                logger.warning("No metadata was successfully extracted")
                return results
            
            print(f"[Stage 3/4] Completed: {len(metadata_dict)}/{len(texts_dict)} metadata extracted")
            
            # Stage 4: Store in Cosmos DB
            print(f"[Stage 4/4] Storing metadata for {len(metadata_dict)} documents...")
            text_samples = {blob_name: texts_dict.get(blob_name, "")[:1000] for blob_name in metadata_dict}
            
            for blob_name, metadata in metadata_dict.items():
                success = self.cosmos_storage.store_document(
                    blob_name,
                    metadata,
                    text_samples.get(blob_name, "")
                )
                
                if success:
                    results.append({
                        "blob_name": blob_name,
                        "success": True,
                        "metadata": metadata,
                        "text": texts_dict.get(blob_name)
                    })
                else:
                    results.append({
                        "blob_name": blob_name,
                        "success": False,
                        "error": "Failed to store in Cosmos DB"
                    })
            
            success_count = sum(1 for r in results if r.get('success', False))
            print(f"[Stage 4/4] Completed: {success_count}/{len(metadata_dict)} documents stored")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return results
    
    def process_all_pdfs(self, pdf_blobs, max_pdfs=None):
        """
        Process all PDFs with batch processing
        
        Args:
            pdf_blobs: List of PDF blob names
            max_pdfs: Optional limit on number of PDFs to process
            
        Returns:
            list: All processing results
        """
        if max_pdfs:
            pdf_blobs = pdf_blobs[:max_pdfs]
        
        all_results = []
        batch_count = (len(pdf_blobs) + self.config.MAX_BATCH_SIZE - 1) // self.config.MAX_BATCH_SIZE
        
        with tqdm(total=len(pdf_blobs), desc="Processing PDFs") as pbar:
            for i in range(0, len(pdf_blobs), self.config.MAX_BATCH_SIZE):
                batch = pdf_blobs[i:i + self.config.MAX_BATCH_SIZE]
                logger.info(f"Processing batch {i//self.config.MAX_BATCH_SIZE + 1}/{batch_count}: {len(batch)} PDFs")
                
                batch_results = self.process_pdf_batch(batch)
                all_results.extend(batch_results)
                
                pbar.update(len(batch))
        
        return all_results
    
    def index_from_cosmos(self):
        """
        Retrieve documents from Cosmos DB and index them in search
        
        Returns:
            dict: Indexing statistics
        """
        print("\n" + "-"*50)
        print("INDEXING DOCUMENTS FROM COSMOS DB")
        print("-"*50)
        
        # Retrieve all documents from Cosmos DB
        all_documents = self.cosmos_storage.query_all_documents()
        print(f"Found {len(all_documents)} documents in Cosmos DB")
        
        if not all_documents:
            return {"total_documents": 0, "indexed_chunks": 0}
        
        # Get all blob names and check which are already indexed (batch mode)
        blob_names = [doc.get("blob_name") for doc in all_documents if doc.get("blob_name")]
        print(f"Checking which of {len(blob_names)} documents are already indexed...")
        
        already_indexed = self.search_indexer.documents_indexed_batch(blob_names)
        print(f"Found {len(already_indexed)} documents already indexed")
        
        # Filter out already indexed documents
        documents_to_index = []
        for doc in all_documents:
            blob_name = doc.get("blob_name")
            if blob_name and blob_name not in already_indexed:
                documents_to_index.append(doc)
        
        print(f"Filtering: {len(documents_to_index)} documents need indexing")
        
        if not documents_to_index:
            return {"total_documents": len(all_documents), "indexed_chunks": 0}
        
        # Partition documents across servers for parallel indexing
        total_docs = len(documents_to_index)
        documents_to_index = documents_to_index[self.server_number::self.server_count]
        print(f"Server {self.server_number + 1}/{self.server_count} processing {len(documents_to_index)} of {total_docs} documents")
        
        # Process in batches
        indexed_count = 0
        batch_size = 100
        
        for i in range(0, len(documents_to_index), batch_size):
            batch = documents_to_index[i:i + batch_size]
            print(f"\n[Batch {i//batch_size + 1}] Processing {len(batch)} documents")
            
            # Download PDFs and extract text
            blob_names = [doc.get("blob_name") for doc in batch]
            local_paths_dict = self.downloader.download_batch(blob_names)
            texts_dict = self.text_extractor.extract_batch(local_paths_dict)
            
            # Cleanup
            for local_path in local_paths_dict.values():
                try:
                    os.remove(local_path)
                except:
                    pass
            
            # Prepare documents for indexing using existing metadata from Cosmos DB
            documents = []
            for doc in batch:
                blob_name = doc.get("blob_name")
                if blob_name in texts_dict and texts_dict[blob_name]:
                    # Use the metadata from Cosmos DB directly
                    cosmos_metadata = doc.get("metadata", {})
                    documents.append({
                        "blob_name": blob_name,
                        "success": True,
                        "metadata": cosmos_metadata,
                        "text": texts_dict[blob_name]
                    })
            
            if documents:
                # Chunk, embed, and upload
                chunks = self.chunker.chunk_batch(documents)
                if chunks:
                    chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
                    succeeded, failed = self.search_indexer.upload_chunks(chunks_with_embeddings)
                    indexed_count += succeeded
        
        print(f"\nCompleted indexing: {indexed_count} chunks indexed")
        return {"total_documents": len(all_documents), "indexed_chunks": indexed_count}
    
    def run(self, pdf_blobs=None, max_pdfs=None, skip_to_indexing=False):
        """
        Run the complete pipeline
        
        Args:
            pdf_blobs: Optional list of PDF blobs to process
            max_pdfs: Optional limit on number of PDFs
            skip_to_indexing: Skip PDF processing and go to indexing
            
        Returns:
            dict: Pipeline execution statistics
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print("STARTING PDF PROCESSING PIPELINE")
        print("="*80)
        
        # Create search index
        print("\n[Phase 1/3] Creating search index")
        if not self.search_indexer.create_index():
            print("Failed to create search index")
            return {"status": "error", "message": "Failed to create search index"}
        
        print("Search index created successfully")
        
        if not skip_to_indexing:
            # Process PDFs
            print("\n[Phase 2/3] Processing PDFs")
            if not pdf_blobs:
                print("No PDF blobs provided")
                return {"status": "error", "message": "No PDF blobs provided"}
            
            results = self.process_all_pdfs(pdf_blobs, max_pdfs)
            success_count = sum(1 for r in results if r.get("success", False))
            print(f"PDF processing complete: {success_count} successes")
        
        # Index documents
        print("\n[Phase 3/3] Indexing documents from Cosmos DB")
        indexing_results = self.index_from_cosmos()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print(f"PIPELINE COMPLETED in {total_time:.2f} seconds")
        print(f"Indexed {indexing_results.get('indexed_chunks', 0)} chunks")
        print("="*80 + "\n")
        
        return {
            "status": "success",
            "total_time": total_time,
            "documents_processed": indexing_results.get("total_documents", 0),
            "chunks_indexed": indexing_results.get("indexed_chunks", 0)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.azure_clients.cleanup()
