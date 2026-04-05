import faiss
import numpy as np
import json
import pickle
from typing import List, Dict, Optional
from config import Config
from pathlib import Path

class TemporalVectorStore:
    """FAISS-based vector database with metadata filtering for RAG"""
    
    def __init__(self, vector_dim: int = Config.VECTOR_DIM):
        self.vector_dim = vector_dim
        self.index = None
        self.metadata = []
        self.index_path = None
        self.metadata_path = None
    
    def create_index(self, embeddings: np.ndarray, metadata: List[Dict], author_id: str = None):
        """
        Create FAISS index from embeddings and store metadata.
        
        Args:
            embeddings: numpy array of shape (n_docs, vector_dim)
            metadata: list of dicts containing paper metadata
            author_id: optional author ID for cache organization
        """
        
        if len(embeddings) == 0:
            Config.log_event("VECTOR_STORE_ERROR", "No embeddings provided")
            return {"error": "No embeddings"}
        
        if embeddings.shape[1] != self.vector_dim:
            Config.log_event("VECTOR_STORE_ERROR", f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.vector_dim}")
            return {"error": "Dimension mismatch"}
        
        # Create FAISS index (simple Flat index for exact search)
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # Add embeddings (must be float32)
        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)
        
        # Store metadata
        self.metadata = metadata
        
        # Save to disk
        if author_id:
            self.index_path = Config.EMBEDDINGS_CACHE_DIR / f"{author_id}_index.faiss"
            self.metadata_path = Config.EMBEDDINGS_CACHE_DIR / f"{author_id}_metadata.json"
        
        if self.index_path:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        
        Config.log_event(
            "VECTOR_STORE_CREATED",
            f"Created FAISS index with {len(embeddings)} vectors",
            {
                "vector_dim": self.vector_dim,
                "num_documents": len(embeddings),
                "metadata_count": len(self.metadata)
            }
        )
        
        return {
            "num_vectors": len(embeddings),
            "vector_dim": self.vector_dim,
            "num_papers": len(self.metadata)
        }
    
    def load_index(self, author_id: str):
        """Load previously created index and metadata from disk"""
        
        index_path = Config.EMBEDDINGS_CACHE_DIR / f"{author_id}_index.faiss"
        metadata_path = Config.EMBEDDINGS_CACHE_DIR / f"{author_id}_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            Config.log_event("VECTOR_STORE_NOT_FOUND", f"No cached index for author {author_id}")
            return False
        
        try:
            self.index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.index_path = index_path
            self.metadata_path = metadata_path
            
            Config.log_event(
                "VECTOR_STORE_LOADED",
                f"Loaded FAISS index with {self.index.ntotal} vectors"
            )
            return True
        
        except Exception as e:
            Config.log_event("VECTOR_STORE_LOAD_ERROR", f"Failed to load index: {str(e)}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        year_range: tuple = None,
        citation_threshold: int = None
    ) -> List[Dict]:
        """
        Search vector store with optional metadata filtering.
        
        Args:
            query_embedding: query vector
            k: number of results
            year_range: (start_year, end_year) tuple
            citation_threshold: minimum citation count
        
        Returns:
            List of matching papers with metadata
        """
        
        if self.index is None or len(self.metadata) == 0:
            return []
        
        # Convert to float32 for FAISS
        query = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search with larger k to account for filtering
        search_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)
        
        results = []
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(self.metadata):
                continue
            
            metadata = self.metadata[idx]
            
            # Apply filters
            if year_range:
                start_year, end_year = year_range
                if not (start_year <= metadata.get("year", 0) <= end_year):
                    continue
            
            if citation_threshold is not None:
                if metadata.get("citationCount", 0) < citation_threshold:
                    continue
            
            results.append({
                "distance": float(distance),  # L2 distance
                "similarity": 1 / (1 + float(distance)),  # Convert to similarity
                **metadata
            })
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_topic(
        self,
        topic_id: int,
        k: int = 5,
        year_range: tuple = None
    ) -> List[Dict]:
        """
        Search documents by topic ID.
        This is for analyzing topic-specific content.
        """
        
        if not self.metadata:
            return []
        
        results = []
        
        for i, metadata in enumerate(self.metadata):
            if metadata.get("topic_id") != topic_id:
                continue
            
            if year_range:
                start_year, end_year = year_range
                if not (start_year <= metadata.get("year", 0) <= end_year):
                    continue
            
            results.append({
                "index": i,
                **metadata
            })
        
        # Sort by citation count (most influential first)
        results.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        
        return results[:k]
    
    def get_statistics(self) -> Dict:
        """Get statistics about the vector store"""
        
        if self.index is None:
            return {"error": "No index"}
        
        num_vectors = self.index.ntotal
        
        if num_vectors == 0 or not self.metadata:
            return {
                "num_vectors": 0,
                "num_papers": 0,
                "vector_dim": self.vector_dim
            }
        
        # Calculate statistics from metadata
        years = [m.get("year") for m in self.metadata if m.get("year")]
        citations = [m.get("citationCount", 0) for m in self.metadata]
        
        return {
            "num_vectors": num_vectors,
            "num_papers": len(self.metadata),
            "vector_dim": self.vector_dim,
            "year_range": (min(years), max(years)) if years else (None, None),
            "avg_citations": sum(citations) / len(citations) if citations else 0,
            "max_citations": max(citations) if citations else 0,
            "min_citations": min(citations) if citations else 0
        }


# Convenience functions
def create_vector_store(embeddings: np.ndarray, papers_df, author_id: str):
    """Create and save vector store"""
    
    metadata = [
        {
            "paperId": row["paperId"],
            "title": row["title"],
            "year": int(row["year"]),
            "authors": row["authors"],
            "citationCount": int(row["citationCount"]),
            "fieldsOfStudy": row.get("fieldsOfStudy", []),
            "venue": row.get("venue", ""),
            "topic_id": int(row.get("topic_id", -1))
        }
        for _, row in papers_df.iterrows()
    ]
    
    store = TemporalVectorStore()
    result = store.create_index(embeddings, metadata, author_id)
    
    return store, result
