"""PDF download and validation processor."""

import logging
import requests
import tempfile
import os
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class PDFDownloader:
    """Downloads PDF files from URLs or Azure Blob Storage."""
    
    def __init__(self, azure_clients, config):
        self.azure_clients = azure_clients
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_single_pdf(self, blob_name: str) -> str:
        """Download a single PDF and return local path."""
        try:
            # Construct full URL from blob name
            if blob_name.startswith('http'):
                url = blob_name
            else:
                url = f"https://courtdata.blob.core.windows.net/supremecourt-judgement/{blob_name}"
            
            logger.info(f"Downloading PDF from {url[:100]}...")
            
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(response.content)
            temp_file.close()
            
            # Validate PDF
            if not self._validate_pdf(response.content):
                os.unlink(temp_file.name)
                raise ValueError("Invalid PDF content")
            
            logger.info(f"Downloaded PDF ({len(response.content)} bytes) to {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to download PDF from {blob_name}: {e}")
            return None
    
    def download_batch(self, blob_names):
        """Download multiple PDFs in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(blob_names), self.config.MAX_WORKERS)) as executor:
            future_to_blob = {}
            for blob_name in blob_names:
                future = executor.submit(self.download_single_pdf, blob_name)
                future_to_blob[future] = blob_name
            
            for future in as_completed(future_to_blob):
                blob_name = future_to_blob[future]
                try:
                    local_path = future.result()
                    if local_path:
                        results[blob_name] = local_path
                except Exception as e:
                    logger.error(f"Error downloading {blob_name}: {e}")
        
        return results
    
    def _validate_pdf(self, content: bytes) -> bool:
        """Validate PDF content."""
        if len(content) < 1024:  # Minimum size
            return False
        
        # Check PDF header
        if not content.startswith(b'%PDF-'):
            return False
        
        return True