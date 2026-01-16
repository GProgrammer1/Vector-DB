#!/usr/bin/env python3
"""
Test script for searching the vector database.

Usage:
    python test_search.py
"""

import requests
import json
from typing import Dict, Any, Optional, List

# Configuration
API_BASE_URL = "http://localhost:8000"


def search(
    query: str,
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
    ef: Optional[int] = None
) -> Dict[str, Any]:
    """
    Search for similar documents in the vector database.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        metadata_filter: Optional metadata filter dictionary
        ef: Optional ef parameter for HNSW search
        
    Returns:
        Response JSON from the API
    """
    url = f"{API_BASE_URL}/search"
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    if metadata_filter:
        payload["metadata_filter"] = metadata_filter
    if ef:
        payload["ef"] = ef
    
    print(f"Searching for: '{query}'")
    if metadata_filter:
        print(f"  Filter: {metadata_filter}")
    print(f"  Top K: {top_k}")
    if ef:
        print(f"  EF: {ef}")
    print()
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result


def display_results(results: List[Dict[str, Any]]):
    """Display search results in a formatted way."""
    if not results:
        print("  No results found.")
        print()
        return
    
    print(f"  Found {len(results)} result(s):")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"  [{i}] ID: {result['id']}")
        print(f"      Content: {result['content']}")
        print(f"      Distance: {result['distance']:.4f}")
        if result.get('metadata'):
            print(f"      Metadata: {result['metadata']}")
        print()


def check_health() -> Dict[str, Any]:
    """Check the health of the indexing service."""
    url = f"{API_BASE_URL}/health"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def main():
    """Main test function."""
    print("=" * 60)
    print("Vector DB - Search Test")
    print("=" * 60)
    print()
    
    # Check health first
    try:
        health = check_health()
        print(f"Service Status: {health['status']}")
        print(f"Index Size: {health['index_size']}")
        print(f"Storage Size: {health['storage_size']}")
        print()
        
        if health['index_size'] == 0:
            print("⚠️  Warning: Index is empty. Run test_insert.py first to add documents.")
            print()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API.")
        print(f"   Make sure the indexing service is running on {API_BASE_URL}")
        print("   Start it with: uvicorn src.vector_db.api.app:app --host 0.0.0.0 --port 8000")
        return
    except Exception as e:
        print(f"❌ Error checking health: {e}")
        return
    
    # Test various search queries
    print("Running search tests...")
    print("-" * 60)
    print()
    
    test_queries = [
        {
            "query": "programming language",
            "top_k": 3,
            "description": "General programming search"
        },
        {
            "query": "artificial intelligence",
            "top_k": 5,
            "description": "AI-related search"
        },
        {
            "query": "web framework",
            "top_k": 3,
            "description": "Framework search"
        },
        {
            "query": "machine learning",
            "top_k": 5,
            "metadata_filter": {"category": "AI"},
            "description": "Filtered search (AI category only)"
        },
        {
            "query": "programming",
            "top_k": 5,
            "metadata_filter": {"category": "programming"},
            "description": "Filtered search (programming category only)"
        },
        {
            "query": "database",
            "top_k": 3,
            "description": "Database-related search"
        },
    ]
    
    successful_searches = 0
    for test in test_queries:
        try:
            print(f"Test: {test['description']}")
            result = search(
                query=test["query"],
                top_k=test["top_k"],
                metadata_filter=test.get("metadata_filter")
            )
            
            display_results(result.get("results", []))
            successful_searches += 1
            
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ Search failed: {e}")
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
    
    print("-" * 60)
    print(f"✓ Completed {successful_searches}/{len(test_queries)} search tests")
    print()
    
    # Interactive mode
    print("=" * 60)
    print("Interactive Search Mode")
    print("=" * 60)
    print("Enter search queries (or 'quit' to exit)")
    print()
    
    while True:
        try:
            query = input("Search query: ").strip()
            if not query or query.lower() in ['quit', 'exit', 'q']:
                break
            
            top_k = input("Number of results (default 5): ").strip()
            top_k = int(top_k) if top_k else 5
            
            use_filter = input("Use metadata filter? (y/n, default n): ").strip().lower()
            metadata_filter = None
            if use_filter == 'y':
                filter_key = input("  Filter key: ").strip()
                filter_value = input("  Filter value: ").strip()
                if filter_key and filter_value:
                    metadata_filter = {filter_key: filter_value}
            
            print()
            result = search(query=query, top_k=top_k, metadata_filter=metadata_filter)
            display_results(result.get("results", []))
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

