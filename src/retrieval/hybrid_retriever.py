import yaml
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging


class HybridRetriever:
    def __init__(self, sparse_retriever, dense_retriever, config_path: str = "retrieval_config.yaml"):
        """
        Initialize hybrid retriever that combines sparse and dense retrieval.
        
        Args:
            sparse_retriever: SparseRetriever instance
            dense_retriever: DenseRetriever instance
            config_path: Path to retrieval configuration
        """
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.config = self._load_config(config_path)
        self.hybrid_config = self.config.get("hybrid", {})
        
        self.weight_sparse = self.hybrid_config.get("weight_sparse", 0.3)
        self.weight_dense = self.hybrid_config.get("weight_dense", 0.7)
        self.top_k = self.hybrid_config.get("top_k", 15)
        self.rerank = self.hybrid_config.get("rerank", True)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load retrieval configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading retrieval config: {e}")
            return {}
    
    def retrieve(self, prompt: str, query_embedding: np.ndarray = None, domain: str = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining sparse and dense results.
        
        Args:
            prompt: Text query for sparse retrieval
            query_embedding: Embedding vector for dense retrieval
            domain: Optional domain filter
            
        Returns:
            List of retrieved documents with combined scores
        """
        try:
            # Get sparse results
            sparse_hits = self.sparse.retrieve(prompt, self.top_k * 2)
            
            # Get dense results
            dense_hits = []
            if query_embedding is not None:
                dense_hits = self.dense.retrieve(query_embedding, self.top_k * 2)
            
            # Combine and score results
            combined_results = self._combine_results(sparse_hits, dense_hits)
            
            # Apply domain filter if specified
            if domain:
                combined_results = self._filter_by_domain(combined_results, domain)
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Return top_k results
            return combined_results[:self.top_k]
            
        except Exception as e:
            logging.error(f"Error during hybrid retrieval: {e}")
            return []
    
    def _combine_results(self, sparse_hits: List[Dict], dense_hits: List[Dict]) -> List[Dict]:
        """
        Combine sparse and dense retrieval results.
        
        Args:
            sparse_hits: Results from sparse retrieval
            dense_hits: Results from dense retrieval
            
        Returns:
            Combined results with normalized scores
        """
        combined = {}
        
        # Process sparse hits
        for hit in sparse_hits:
            segment_id = hit.get("segment_id", str(hash(hit.get("text", ""))))
            combined.setdefault(segment_id, {
                "hit": hit,
                "sparse_score": 0.0,
                "dense_score": 0.0,
                "combined_score": 0.0
            })
            combined[segment_id]["sparse_score"] = hit.get("score", 0.0)
        
        # Process dense hits
        for hit in dense_hits:
            segment_id = hit.get("segment_id", str(hash(hit.get("text", ""))))
            combined.setdefault(segment_id, {
                "hit": hit,
                "sparse_score": 0.0,
                "dense_score": 0.0,
                "combined_score": 0.0
            })
            combined[segment_id]["dense_score"] = hit.get("score", 0.0)
        
        # Normalize scores and calculate combined score
        sparse_scores = [item["sparse_score"] for item in combined.values() if item["sparse_score"] > 0]
        dense_scores = [item["dense_score"] for item in combined.values() if item["dense_score"] > 0]
        
        sparse_max = max(sparse_scores) if sparse_scores else 1.0
        dense_max = max(dense_scores) if dense_scores else 1.0
        
        # Calculate combined scores
        for item in combined.values():
            normalized_sparse = item["sparse_score"] / sparse_max if sparse_max > 0 else 0.0
            normalized_dense = item["dense_score"] / dense_max if dense_max > 0 else 0.0
            
            item["combined_score"] = (
                self.weight_sparse * normalized_sparse + 
                self.weight_dense * normalized_dense
            )
        
        # Convert to list format
        results = []
        for segment_id, item in combined.items():
            hit = item["hit"]
            result = {
                "segment_id": segment_id,
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
                "domain": hit.get("domain", "unknown"),
                "sparse_score": item["sparse_score"],
                "dense_score": item["dense_score"],
                "combined_score": item["combined_score"],
                "highlights": hit.get("highlights", [])
            }
            results.append(result)
        
        return results
    
    def _filter_by_domain(self, results: List[Dict], domain: str) -> List[Dict]:
        """Filter results by domain."""
        return [result for result in results if result.get("domain") == domain]
    
    def retrieve_with_reranking(self, prompt: str, query_embedding: np.ndarray = None, 
                               domain: str = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval with additional reranking.
        
        Args:
            prompt: Text query for sparse retrieval
            query_embedding: Embedding vector for dense retrieval
            domain: Optional domain filter
            
        Returns:
            Reranked list of retrieved documents
        """
        # Get initial hybrid results
        initial_results = self.retrieve(prompt, query_embedding, domain)
        
        if not self.rerank:
            return initial_results
        
        # Apply reranking logic
        reranked_results = self._rerank_results(initial_results, prompt, query_embedding)
        
        return reranked_results
    
    def _rerank_results(self, results: List[Dict], prompt: str, 
                       query_embedding: np.ndarray = None) -> List[Dict]:
        """
        Rerank results using additional signals.
        
        Args:
            results: Initial retrieval results
            prompt: Original query
            query_embedding: Query embedding
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        # Simple reranking based on text length and domain relevance
        for result in results:
            rerank_score = 0.0
            
            # Boost shorter, more focused segments
            text_length = len(result.get("text", ""))
            if 100 <= text_length <= 500:
                rerank_score += 0.1
            elif text_length > 1000:
                rerank_score -= 0.1
            
            # Boost segments with highlights
            if result.get("highlights"):
                rerank_score += 0.2
            
            # Boost segments with higher combined score
            rerank_score += result.get("combined_score", 0.0) * 0.3
            
            result["rerank_score"] = rerank_score
        
        # Sort by rerank score
        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        return results
    
    def get_retrieval_stats(self, prompt: str, query_embedding: np.ndarray = None) -> Dict[str, Any]:
        """
        Get statistics about the retrieval process.
        
        Args:
            prompt: Text query
            query_embedding: Query embedding
            
        Returns:
            Retrieval statistics
        """
        try:
            # Get individual results
            sparse_hits = self.sparse.retrieve(prompt, self.top_k)
            dense_hits = []
            if query_embedding is not None:
                dense_hits = self.dense.retrieve(query_embedding, self.top_k)
            
            # Get hybrid results
            hybrid_hits = self.retrieve(prompt, query_embedding)
            
            return {
                "sparse_results": len(sparse_hits),
                "dense_results": len(dense_hits),
                "hybrid_results": len(hybrid_hits),
                "weight_sparse": self.weight_sparse,
                "weight_dense": self.weight_dense,
                "top_k": self.top_k,
                "rerank_enabled": self.rerank
            }
            
        except Exception as e:
            logging.error(f"Error getting retrieval stats: {e}")
            return {}
    
    def update_weights(self, weight_sparse: float, weight_dense: float):
        """
        Update the weights for sparse and dense retrieval.
        
        Args:
            weight_sparse: New weight for sparse retrieval
            weight_dense: New weight for dense retrieval
        """
        if weight_sparse + weight_dense != 1.0:
            # Normalize weights
            total = weight_sparse + weight_dense
            weight_sparse /= total
            weight_dense /= total
        
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense
        
        logging.info(f"Updated hybrid weights: sparse={weight_sparse}, dense={weight_dense}")
    
    def set_top_k(self, top_k: int):
        """Set the number of top results to return."""
        self.top_k = top_k
        logging.info(f"Updated top_k to {top_k}")
    
    def enable_reranking(self, enable: bool = True):
        """Enable or disable reranking."""
        self.rerank = enable
        logging.info(f"Reranking {'enabled' if enable else 'disabled'}") 