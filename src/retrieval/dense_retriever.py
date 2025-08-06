import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import faiss
import pickle
import os
from pathlib import Path
import logging


class DenseRetriever:
    def __init__(self, config_path: str = "retrieval_config.yaml"):
        """Initialize dense retriever with FAISS configuration."""
        self.config = self._load_config(config_path)
        self.dense_config = self.config.get("dense", {})
        
        self.index_name = self.dense_config.get("index_name", "claims-dense-index")
        self.model_name = self.dense_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.similarity_metric = self.dense_config.get("similarity_metric", "cosine")
        
        # Initialize FAISS index and metadata storage
        self.index = None
        self.metadata_store = {}
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Load or create index
        self._initialize_index()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load retrieval configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading retrieval config: {e}")
            return {}
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        index_path = f"{self.index_name}.faiss"
        metadata_path = f"{self.index_name}_metadata.pkl"
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing index
            try:
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                logging.info(f"Loaded existing FAISS index: {index_path}")
            except Exception as e:
                logging.error(f"Error loading existing index: {e}")
                self._create_new_index()
        else:
            # Create new index
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        try:
            if self.similarity_metric == "cosine":
                # Use IndexFlatIP for cosine similarity (inner product of normalized vectors)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                # Use IndexFlatL2 for L2 distance
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            self.metadata_store = {}
            logging.info(f"Created new FAISS index with {self.embedding_dim} dimensions")
            
        except Exception as e:
            logging.error(f"Error creating FAISS index: {e}")
            self.index = None
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        if self.similarity_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        return embeddings
    
    def add_documents(self, embeddings: List[np.ndarray], metadata: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the FAISS index.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            logging.error("FAISS index not initialized")
            return False
        
        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings
            normalized_embeddings = self._normalize_embeddings(embeddings_array)
            
            # Add to index
            self.index.add(normalized_embeddings)
            
            # Store metadata
            start_id = len(self.metadata_store)
            for i, meta in enumerate(metadata):
                self.metadata_store[start_id + i] = meta
            
            logging.info(f"Added {len(embeddings)} documents to FAISS index")
            return True
            
        except Exception as e:
            logging.error(f"Error adding documents to FAISS index: {e}")
            return False
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents using dense vector search.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.index:
            logging.error("FAISS index not initialized")
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            normalized_query = self._normalize_embeddings(query_embedding)
            
            # Search
            scores, indices = self.index.search(normalized_query, min(top_k, self.index.ntotal))
            
            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx in self.metadata_store:  # -1 indicates no result
                    metadata = self.metadata_store[idx]
                    result = {
                        "segment_id": metadata.get("segment_id", str(idx)),
                        "text": metadata.get("text", ""),
                        "metadata": metadata,
                        "domain": metadata.get("domain", "unknown"),
                        "score": float(score),
                        "index": int(idx)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error during dense retrieval: {e}")
            return []
    
    def retrieve_by_domain(self, query_embedding: np.ndarray, domain: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents filtered by domain.
        
        Args:
            query_embedding: Query embedding vector
            domain: Target domain
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # First get more results than needed
        all_results = self.retrieve(query_embedding, top_k * 3)
        
        # Filter by domain
        domain_results = [
            result for result in all_results 
            if result.get("domain") == domain
        ]
        
        return domain_results[:top_k]
    
    def save_index(self) -> bool:
        """Save the FAISS index and metadata to disk."""
        if not self.index:
            return False
        
        try:
            # Save FAISS index
            index_path = f"{self.index_name}.faiss"
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = f"{self.index_name}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logging.info(f"Saved FAISS index to {index_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")
            return False
    
    def load_index(self, index_path: str, metadata_path: str) -> bool:
        """Load a FAISS index from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata_store = pickle.load(f)
            
            logging.info(f"Loaded FAISS index from {index_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index."""
        if not self.index:
            return {}
        
        try:
            return {
                "index_name": self.index_name,
                "document_count": self.index.ntotal,
                "embedding_dimensions": self.index.d,
                "similarity_metric": self.similarity_metric,
                "metadata_count": len(self.metadata_store)
            }
            
        except Exception as e:
            logging.error(f"Error getting index stats: {e}")
            return {}
    
    def clear_index(self) -> bool:
        """Clear all documents from the index."""
        if not self.index:
            return False
        
        try:
            # Reset index
            if self.similarity_metric == "cosine":
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Clear metadata
            self.metadata_store = {}
            
            logging.info("Cleared FAISS index")
            return True
            
        except Exception as e:
            logging.error(f"Error clearing index: {e}")
            return False
    
    def delete_document(self, segment_id: str) -> bool:
        """
        Delete a document from the index by segment ID.
        Note: FAISS doesn't support direct deletion, so this rebuilds the index.
        
        Args:
            segment_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.index:
            return False
        
        try:
            # Find the document index
            doc_index = None
            for idx, metadata in self.metadata_store.items():
                if metadata.get("segment_id") == segment_id:
                    doc_index = idx
                    break
            
            if doc_index is None:
                logging.warning(f"Document {segment_id} not found in index")
                return False
            
            # Rebuild index without the deleted document
            # This is a simplified approach - in production, you might want a more efficient method
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            new_metadata = {}
            new_vectors = []
            
            for i in range(self.index.ntotal):
                if i != doc_index:
                    new_vectors.append(all_vectors[i])
                    new_metadata[len(new_vectors) - 1] = self.metadata_store[i]
            
            # Create new index
            if new_vectors:
                new_vectors_array = np.array(new_vectors, dtype=np.float32)
                normalized_vectors = self._normalize_embeddings(new_vectors_array)
                
                if self.similarity_metric == "cosine":
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                
                self.index.add(normalized_vectors)
                self.metadata_store = new_metadata
            
            logging.info(f"Deleted document {segment_id} from index")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting document: {e}")
            return False 