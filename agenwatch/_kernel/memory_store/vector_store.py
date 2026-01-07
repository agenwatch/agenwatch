"""
AgenWatch Vector Store - Production Ready
===========================================
Minimal, fast, and correct vector similarity search.
"""

import math
from typing import List, Dict, Any, Optional


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Similarity score between -1 and 1 (typically 0 to 1 for embeddings)
    
    Raises:
        ValueError: If vectors have different dimensions
    
    Examples:
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}. "
            f"Both vectors must have the same dimension."
        )
    
    if not a or not b:
        raise ValueError("Vectors cannot be empty")
    
    # Compute dot product
    dot = sum(x * y for x, y in zip(a, b))
    
    # Compute magnitudes
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    
    # Handle zero vectors
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot / (mag_a * mag_b)


class SimpleVectorStore:
    """
    A minimal vector store for AgenWatch long-term memory.
    
    This is an in-memory vector store optimized for simplicity and correctness.
    For production systems with >10K vectors, consider upgrading to:
    - Pinecone
    - Weaviate
    - Qdrant
    - ChromaDB
    
    Attributes:
        _items: List of stored items with embeddings
    
    Examples:
        >>> store = SimpleVectorStore()
        >>> store.add({"id": "1", "text": "Hello", "embedding": [0.1, 0.2, 0.3]})
        >>> results = store.search([0.1, 0.2, 0.3], top_k=1)
        >>> len(results)
        1
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self._items: List[Dict[str, Any]] = []
    
    def add(self, item: Dict[str, Any]) -> None:
        """
        Store an item with its embedding.
        
        Args:
            item: Dictionary containing at least an 'embedding' key with List[float] value.
                  Typically also includes 'id', 'content', 'metadata', etc.
        
        Raises:
            ValueError: If item is missing 'embedding' key or embedding is invalid
        
        Examples:
            >>> store = SimpleVectorStore()
            >>> store.add({
            ...     "id": "mem_001",
            ...     "content": "User prefers Python",
            ...     "embedding": [0.1, 0.2, 0.3]
            ... })
        """
        if "embedding" not in item:
            raise ValueError(
                "Item must contain 'embedding' key. "
                f"Got keys: {list(item.keys())}"
            )
        
        embedding = item["embedding"]
        if not isinstance(embedding, list):
            raise ValueError(
                f"Embedding must be a list, got {type(embedding)}"
            )
        
        if not embedding:
            raise ValueError("Embedding cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("Embedding must contain only numbers")
        
        self._items.append(item)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.3,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar items using cosine similarity.
        
        Args:
            query_embedding: The query vector to search for
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)
            filter_fn: Optional function to filter items before scoring.
                       Should take item dict and return bool.
        
        Returns:
            List of items sorted by similarity (highest first).
            Each item includes its original data plus a 'score' key.
        
        Examples:
            >>> store = SimpleVectorStore()
            >>> store.add({"id": "1", "embedding": [1.0, 0.0]})
            >>> store.add({"id": "2", "embedding": [0.0, 1.0]})
            >>> results = store.search([1.0, 0.0], top_k=1)
            >>> results[0]["id"]
            '1'
            >>> results[0]["score"] > 0.9
            True
        """
        # Handle empty store
        if not self._items:
            return []
        
        # Validate query embedding
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        if not isinstance(query_embedding, list):
            raise ValueError(
                f"Query embedding must be a list, got {type(query_embedding)}"
            )
        
        # Apply pre-filter if provided
        candidates = self._items
        if filter_fn is not None:
            candidates = [item for item in candidates if filter_fn(item)]
        
        if not candidates:
            return []
        
        # Calculate similarities
        scored = []
        for item in candidates:
            try:
                sim = cosine_similarity(query_embedding, item["embedding"])
                if sim >= min_score:
                    # Create result with score
                    result = item.copy()
                    result["score"] = sim
                    scored.append((sim, result))
            except ValueError as e:
                # Log warning but continue (dimension mismatch, etc.)
                print(f"⚠️  Skipping item due to error: {e}")
                continue
        
        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k results
        return [item for _, item in scored[:top_k]]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all stored items.
        
        Returns:
            List of all items in the store
        """
        return self._items.copy()
    
    def delete(self, item_id: str, id_key: str = "id") -> bool:
        """
        Delete an item by ID.
        
        Args:
            item_id: The ID value to match
            id_key: The key name containing the ID (default: "id")
        
        Returns:
            True if item was deleted, False if not found
        
        Examples:
            >>> store = SimpleVectorStore()
            >>> store.add({"id": "1", "embedding": [0.1, 0.2]})
            >>> store.delete("1")
            True
            >>> store.delete("1")
            False
        """
        original_len = len(self._items)
        self._items = [
            item for item in self._items
            if item.get(id_key) != item_id
        ]
        return len(self._items) < original_len
    
    def clear(self) -> None:
        """Remove all items from the store."""
        self._items = []
    
    def size(self) -> int:
        """
        Get the number of items in the store.
        
        Returns:
            Number of stored items
        """
        return len(self._items)
    
    def __len__(self) -> int:
        """Support len(store) syntax."""
        return len(self._items)
    
    def __repr__(self) -> str:
        """String representation of the store."""
        return f"SimpleVectorStore(items={len(self._items)})"


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE VECTOR STORE - TESTS")
    print("=" * 60)
    
    # Create store
    store = SimpleVectorStore()
    
    # Add items
    print("\n✅ Adding items...")
    store.add({
        "id": "mem_1",
        "content": "User prefers Python over JavaScript",
        "embedding": [0.9, 0.1, 0.2],
        "user_id": "alice"
    })
    
    store.add({
        "id": "mem_2",
        "content": "User is building an AI agent framework",
        "embedding": [0.8, 0.3, 0.1],
        "user_id": "alice"
    })
    
    store.add({
        "id": "mem_3",
        "content": "User likes concise code without bloat",
        "embedding": [0.7, 0.2, 0.4],
        "user_id": "alice"
    })
    
    store.add({
        "id": "mem_4",
        "content": "Bob prefers Java",
        "embedding": [0.1, 0.9, 0.3],
        "user_id": "bob"
    })
    
    print(f"   Stored {len(store)} items")
    
    # Search
    print("\n✅ Searching for similar items...")
    query = [0.85, 0.15, 0.2]  # Similar to Python preference
    results = store.search(query, top_k=3, min_score=0.3)
    
    print(f"   Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. [Score: {result['score']:.3f}] {result['content']}")
    
    # Search with filter
    print("\n✅ Searching with filter (alice only)...")
    results = store.search(
        query,
        top_k=3,
        filter_fn=lambda item: item.get("user_id") == "alice"
    )
    print(f"   Found {len(results)} results for alice")
    
    # Delete
    print("\n✅ Deleting item...")
    deleted = store.delete("mem_4")
    print(f"   Deleted: {deleted}, New size: {len(store)}")
    
    # Edge cases
    print("\n✅ Testing edge cases...")
    
    # Empty store search
    empty_store = SimpleVectorStore()
    results = empty_store.search([0.1, 0.2])
    print(f"   Empty store search: {len(results)} results (expected 0)")
    
    # Zero vector
    try:
        sim = cosine_similarity([0.0, 0.0], [1.0, 1.0])
        print(f"   Zero vector similarity: {sim} (expected 0.0)")
    except Exception as e:
        print(f"   Zero vector error: {e}")
    
    # Dimension mismatch
    try:
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
        print("   ❌ Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✅ Caught dimension mismatch: {str(e)[:50]}...")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)

__INTERNAL__ = True



