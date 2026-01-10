#!/usr/bin/env python3
"""
Get all chunks for a specific PDF from Azure Search index.
Usage: python get_pdf_chunks.py <pdf_url>
"""
import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config.settings import AZURE_CONFIG
import json

def get_pdf_chunks(pdf_url):
    config = AZURE_CONFIG
    api_version = "2023-11-01"
    index_name = config.SEARCH_INDEX_NAME
    endpoint = config.SEARCH_ENDPOINT
    api_key = config.SEARCH_KEY
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    search_url = f"{endpoint}/indexes/{index_name}/docs/search?api-version={api_version}"
    
    # Extract filename from URL
    pdf_filename = pdf_url.split('/')[-1]
    
    print(f"🔍 Searching for chunks of: {pdf_filename}")
    print(f"📄 Full URL: {pdf_url}\n")
    
    # Search for chunks with this pdf_id
    search_payload = {
        "search": "*",
        "filter": f"pdf_id eq '{pdf_url}'",
        "orderby": "chunk_index asc",
        "top": 1000,
        "select": "chunk_index,chunk_total,content,case_name"
    }
    
    response = requests.post(search_url, headers=headers, json=search_payload)
    
    if response.status_code != 200:
        print(f"❌ Error searching index: {response.status_code}")
        print(f"   {response.text}")
        return
    
    data = response.json()
    with open("response.json", "w") as f:
        json.dump(data, f, indent=4)
    chunks = data.get("value", [])
    
    if not chunks:
        print("❌ No chunks found for this PDF")
        return
    
    print(f"✅ Found {len(chunks)} chunks\n")
    
    # Display chunks
    for chunk in chunks:
        chunk_idx = chunk.get("chunk_index", "?")
        chunk_total = chunk.get("chunk_total", "?")
        content = chunk.get("content", "")
        
        print(f"{'='*60}")
        print(f"📄 Chunk {chunk_idx + 1}/{chunk_total}")
        print(f"{'='*60}")
        print(content[:500] + ("..." if len(content) > 500 else ""))
        print()

if __name__ == "__main__":
    get_pdf_chunks("https://courtdata.blob.core.windows.net/highcourt-judgement/f5bb5b04-8401-4148-9994-d50e99b83566.pdf")