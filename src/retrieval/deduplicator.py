import yaml
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import logging


class Deduplicator:
    def __init__(self, config_path: str = "retrieval_config.yaml"):
        """Initialize deduplicator with configuration."""
        self.config = self._load_config(config_path)
        self.dedupe_config = self.config.get("dedupe", {})
        
        self.method = self.dedupe_config.get("method", "agglomerative")
        self.similarity_threshold = self.dedupe_config.get("similarity_threshold", 0.85)
        self.min_cluster_size = self.dedupe_config.get("min_cluster_size", 1)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load retrieval configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading retrieval config: {e}")
            return {}
    
    def dedupe_segments(self, segments: List[Dict[str, Any]], embedder=None) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate segments using clustering.
        
        Args:
            segments: List of segment dictionaries
            embedder: Optional embedder for generating embeddings
            
        Returns:
            List of unique segments
        """
        if not segments:
            return []
        
        if len(segments) == 1:
            return segments
        
        try:
            # Extract embeddings from segments
            embeddings = []
            valid_segments = []
            
            for segment in segments:
                if "embedding" in segment:
                    embeddings.append(segment["embedding"])
                    valid_segments.append(segment)
                elif embedder:
                    # Generate embedding if not present
                    text = segment.get("text", "")
                    if text:
                        embedding = embedder.generate_single_embedding(text)
                        if embedding:
                            segment["embedding"] = embedding.embedding
                            embeddings.append(embedding.embedding)
                            valid_segments.append(segment)
                else:
                    # Skip segments without embeddings
                    logging.warning(f"Segment {segment.get('segment_id', 'unknown')} has no embedding")
                    continue
            
            if not embeddings:
                logging.warning("No valid embeddings found for deduplication")
                return segments
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            # Perform clustering
            if self.method == "agglomerative":
                unique_segments = self._agglomerative_clustering(embeddings_array, valid_segments)
            else:
                # Fallback to simple similarity-based deduplication
                unique_segments = self._simple_deduplication(embeddings_array, valid_segments)
            
            logging.info(f"Deduplication: {len(segments)} -> {len(unique_segments)} segments")
            return unique_segments
            
        except Exception as e:
            logging.error(f"Error during deduplication: {e}")
            return segments
    
    def _agglomerative_clustering(self, embeddings: np.ndarray, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform agglomerative clustering for deduplication.
        
        Args:
            embeddings: Embedding vectors
            segments: Segment data
            
        Returns:
            List of unique segments
        """
        try:
            # Calculate distance threshold from similarity threshold
            distance_threshold = 1.0 - self.similarity_threshold
            
            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average"
            ).fit(embeddings)
            
            # Group segments by cluster
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(segments[i])
            
            # Select representative segment from each cluster
            unique_segments = []
            for cluster_id, cluster_segments in clusters.items():
                if len(cluster_segments) >= self.min_cluster_size:
                    # Select the segment with the highest score or first one
                    representative = self._select_representative(cluster_segments)
                    unique_segments.append(representative)
            
            return unique_segments
            
        except Exception as e:
            logging.error(f"Error in agglomerative clustering: {e}")
            return segments
    
    def _simple_deduplication(self, embeddings: np.ndarray, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simple similarity-based deduplication.
        
        Args:
            embeddings: Embedding vectors
            segments: Segment data
            
        Returns:
            List of unique segments
        """
        try:
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find duplicates
            unique_indices = []
            for i in range(len(embeddings)):
                is_duplicate = False
                for j in unique_indices:
                    if similarity_matrix[i, j] >= self.similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_indices.append(i)
            
            # Return unique segments
            return [segments[i] for i in unique_indices]
            
        except Exception as e:
            logging.error(f"Error in simple deduplication: {e}")
            return segments
    
    def _select_representative(self, cluster_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select a representative segment from a cluster.
        
        Args:
            cluster_segments: Segments in the cluster
            
        Returns:
            Representative segment
        """
        if len(cluster_segments) == 1:
            return cluster_segments[0]
        
        # Try to select based on combined score
        best_segment = cluster_segments[0]
        best_score = 0.0
        
        for segment in cluster_segments:
            # Calculate a composite score
            score = 0.0
            
            # Use combined score if available
            if "combined_score" in segment:
                score += segment["combined_score"] * 0.5
            
            # Use sparse score if available
            if "sparse_score" in segment:
                score += segment["sparse_score"] * 0.3
            
            # Use dense score if available
            if "dense_score" in segment:
                score += segment["dense_score"] * 0.2
            
            # Prefer segments with highlights
            if segment.get("highlights"):
                score += 0.1
            
            # Prefer segments with reasonable length
            text_length = len(segment.get("text", ""))
            if 100 <= text_length <= 500:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_segment = segment
        
        return best_segment
    
    def dedupe_by_text_similarity(self, segments: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Deduplicate segments based on text similarity.
        
        Args:
            segments: List of segments
            threshold: Similarity threshold
            
        Returns:
            List of unique segments
        """
        if not segments:
            return []
        
        try:
            unique_segments = []
            for segment in segments:
                is_duplicate = False
                text = segment.get("text", "").lower()
                
                for unique_segment in unique_segments:
                    unique_text = unique_segment.get("text", "").lower()
                    similarity = self._calculate_text_similarity(text, unique_text)
                    
                    if similarity >= threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_segments.append(segment)
            
            return unique_segments
            
        except Exception as e:
            logging.error(f"Error in text similarity deduplication: {e}")
            return segments
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple metrics.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Simple Jaccard similarity on words
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    def get_deduplication_stats(self, original_count: int, unique_count: int) -> Dict[str, Any]:
        """
        Get statistics about the deduplication process.
        
        Args:
            original_count: Number of original segments
            unique_count: Number of unique segments after deduplication
            
        Returns:
            Deduplication statistics
        """
        if original_count == 0:
            return {}
        
        reduction_ratio = (original_count - unique_count) / original_count
        
        return {
            "original_count": original_count,
            "unique_count": unique_count,
            "duplicates_removed": original_count - unique_count,
            "reduction_ratio": reduction_ratio,
            "method": self.method,
            "similarity_threshold": self.similarity_threshold
        }
    
    def update_threshold(self, threshold: float):
        """Update the similarity threshold for deduplication."""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        logging.info(f"Updated similarity threshold to {self.similarity_threshold}")
    
    def update_method(self, method: str):
        """Update the deduplication method."""
        if method in ["agglomerative", "simple"]:
            self.method = method
            logging.info(f"Updated deduplication method to {method}")
        else:
            logging.warning(f"Unknown deduplication method: {method}") 