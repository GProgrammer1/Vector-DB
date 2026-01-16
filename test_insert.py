#!/usr/bin/env python3
"""
Test script for inserting embeddings into the vector database.

Usage:
    python test_insert.py
"""

import requests
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"


def insert_document(content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Insert a document into the vector database.
    
    Args:
        content: Text content to embed and store
        metadata: Optional metadata dictionary
        
    Returns:
        Response JSON from the API
    """
    url = f"{API_BASE_URL}/embed"
    payload = {
        "content": content,
        "metadata": metadata or {}
    }
    
    print(f"Inserting: '{content}'")
    if metadata:
        print(f"  Metadata: {metadata}")
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    print(f"  ✓ Success: {result['message']}")
    print()
    
    return result


def check_health() -> Dict[str, Any]:
    """Check the health of the indexing service."""
    url = f"{API_BASE_URL}/health"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def main():
    """Main test function."""
    print("=" * 60)
    print("Vector DB - Embedding Insertion Test")
    print("=" * 60)
    print()
    
    # Check health first
    try:
        health = check_health()
        print(f"Service Status: {health['status']}")
        print(f"Index Size: {health['index_size']}")
        print(f"Storage Size: {health['storage_size']}")
        print()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print(f"   Make sure the indexing service is running on {API_BASE_URL}")
        print("   Start it with: uvicorn src.vector_db.api.app:app --host 0.0.0.0 --port 8000")
        return
    except Exception as e:
        print(f"❌ Error checking health: {e}")
        return
    
    # Test inserting various documents
    print("Inserting test documents...")
    print("-" * 60)
    
    documents = [
        {
            "content": "Python is a high-level programming language",
            "metadata": {"category": "programming", "language": "python"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence",
            "metadata": {"category": "AI", "topic": "ML"}
        },
        {
            "content": "FastAPI is a modern web framework for building APIs",
            "metadata": {"category": "programming", "framework": "FastAPI"}
        },
        {
            "content": "Vector databases are optimized for similarity search",
            "metadata": {"category": "database", "type": "vector"}
        },
        {
            "content": "Docker containers provide isolated runtime environments",
            "metadata": {"category": "devops", "tool": "Docker"}
        },
        {
            "content": "Natural language processing enables computers to understand text",
            "metadata": {"category": "AI", "topic": "NLP"}
        },
    ]
    
    inserted_count = 0
    for doc in documents:
        try:
            insert_document(doc["content"], doc["metadata"])
            inserted_count += 1
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ Failed: {e}")
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"    Detail: {error_detail}")
                except:
                    print(f"    Response: {e.response.text}")
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()
    
    # Check final health
    print("-" * 60)
    print("Final Status:")
    try:
        health = check_health()
        print(f"  Index Size: {health['index_size']}")
        print(f"  Storage Size: {health['storage_size']}")
        print(f"  Index Modified: {health.get('index_modified', False)}")
        print()
        print(f"✓ Successfully inserted {inserted_count}/{len(documents)} documents")
    except Exception as e:
        print(f"  Error checking final status: {e}")


if __name__ == "__main__":
    main()

