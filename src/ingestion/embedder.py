import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class DocumentEmbedder:
    def __init__(self, config_path: str = "ingestion_config.yaml"):
        """Initialize document embedder with configuration."""
        self.config = self._load_config(config_path)
        self.embedding_config = self.config.get("ingestion", {}).get("embeddings", {})
        
        self.model_name = self.embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = self.embedding_config.get("batch_size", 32)
        self.max_length = self.embedding_config.get("max_length", 512)
        
        # Initialize the embedding model
        self.model = self._load_model()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load ingestion configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        try:
            model = SentenceTransformer(self.model_name)
            return model
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to a simpler model
            return SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_embeddings(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            metadata: Optional list of metadata dictionaries
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = metadata[i:i + self.batch_size]
            
            batch_embeddings = self._generate_batch_embeddings(batch_texts, batch_metadata)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _generate_batch_embeddings(self, texts: List[str], metadata: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of texts."""
        try:
            # Encode texts to embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Create EmbeddingResult objects
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    metadata=metadata[i].copy() if metadata[i] else {}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []
    
    def generate_single_embedding(self, text: str, metadata: Dict[str, Any] = None) -> Optional[EmbeddingResult]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            metadata: Optional metadata dictionary
            
        Returns:
            EmbeddingResult object or None if failed
        """
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                metadata=metadata.copy() if metadata else {}
            )
            
        except Exception as e:
            print(f"Error generating single embedding: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Ensure embeddings are normalized
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_documents(self, query_embedding: np.ndarray, document_embeddings: List[EmbeddingResult], 
                             top_k: int = 5, threshold: float = 0.5) -> List[Tuple[EmbeddingResult, float]]:
        """
        Find documents similar to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embeddings
            top_k: Number of top similar documents to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (document, similarity_score) tuples
        """
        similarities = []
        
        for doc_embedding in document_embeddings:
            similarity = self.calculate_similarity(query_embedding, doc_embedding.embedding)
            if similarity >= threshold:
                similarities.append((doc_embedding, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
    
    def batch_similarity_search(self, query_embeddings: List[np.ndarray], 
                              document_embeddings: List[EmbeddingResult],
                              top_k: int = 5, threshold: float = 0.5) -> List[List[Tuple[EmbeddingResult, float]]]:
        """
        Perform batch similarity search for multiple queries.
        
        Args:
            query_embeddings: List of query embedding vectors
            document_embeddings: List of document embeddings
            top_k: Number of top similar documents to return per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of results for each query
        """
        results = []
        
        for query_embedding in query_embeddings:
            query_results = self.find_similar_documents(
                query_embedding, document_embeddings, top_k, threshold
            )
            results.append(query_results)
        
        return results
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        try:
            # Create a dummy embedding to get dimensions
            dummy_text = "test"
            dummy_embedding = self.model.encode([dummy_text], convert_to_numpy=True)[0]
            return len(dummy_embedding)
        except Exception as e:
            print(f"Error getting embedding dimensions: {e}")
            return 384  # Default for all-MiniLM-L6-v2
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Basic text preprocessing
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            'model_name': self.model_name,
            'embedding_dimensions': self.get_embedding_dimensions(),
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'device': str(self.model.device)
        } 