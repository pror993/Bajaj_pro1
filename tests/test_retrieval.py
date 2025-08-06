import unittest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import yaml

# Import retrieval components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval.prompt_builder import PromptBuilder
from retrieval.sparse_retriever import SparseRetriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.deduplicator import Deduplicator


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.prompt_builder = PromptBuilder()
    
    def test_build_search_prompt_health(self):
        """Test building search prompt for health domain."""
        raw_query = "What is covered for knee surgery?"
        slots = {
            "procedure": "knee surgery",
            "duration": "3-month policy"
        }
        domain = "health"
        
        prompt = self.prompt_builder.build_search_prompt(raw_query, slots, domain)
        
        self.assertIn("knee surgery", prompt)
        self.assertIn("3-month policy", prompt)
        self.assertIn(raw_query, prompt)
    
    def test_build_search_prompt_motor(self):
        """Test building search prompt for motor domain."""
        raw_query = "What coverage applies to my accident?"
        slots = {
            "vehicle_type": "car",
            "accident_date": "2024-01-15",
            "location": "New York",
            "damage_type": "collision"
        }
        domain = "motor"
        
        prompt = self.prompt_builder.build_search_prompt(raw_query, slots, domain)
        
        self.assertIn("car", prompt)
        self.assertIn("2024-01-15", prompt)
        self.assertIn("New York", prompt)
        self.assertIn("collision", prompt)
    
    def test_build_search_prompt_travel(self):
        """Test building search prompt for travel domain."""
        raw_query = "Can I cancel my trip?"
        slots = {
            "destination": "Paris",
            "trip_duration": "2 weeks",
            "cancellation_reason": "medical emergency"
        }
        domain = "travel"
        
        prompt = self.prompt_builder.build_search_prompt(raw_query, slots, domain)
        
        self.assertIn("Paris", prompt)
        self.assertIn("2 weeks", prompt)
        self.assertIn("medical emergency", prompt)
    
    def test_build_generic_prompt(self):
        """Test building generic prompt when domain template fails."""
        raw_query = "What is covered?"
        slots = {"age": 30, "gender": "male"}
        domain = "unknown"
        
        prompt = self.prompt_builder.build_search_prompt(raw_query, slots, domain)
        
        self.assertIn("unknown", prompt)
        self.assertIn("age: 30", prompt)
        self.assertIn("gender: male", prompt)
    
    def test_validate_slots_for_domain(self):
        """Test slot validation for domains."""
        slots = {"procedure": "surgery", "duration": "6 months"}
        domain = "health"
        
        validation = self.prompt_builder.validate_slots_for_domain(slots, domain)
        
        self.assertIsInstance(validation, dict)
        self.assertIn("valid", validation)
        self.assertIn("missing", validation)
    
    def test_get_available_domains(self):
        """Test getting available domains."""
        domains = self.prompt_builder.get_available_domains()
        
        self.assertIsInstance(domains, list)
        self.assertIn("health", domains)
        self.assertIn("motor", domains)
        self.assertIn("travel", domains)


class TestSparseRetriever(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with mocked Elasticsearch."""
        self.patcher = patch('retrieval.sparse_retriever.Elasticsearch')
        mock_elasticsearch = self.patcher.start()
        mock_es = Mock()
        mock_es.ping.return_value = True
        mock_elasticsearch.return_value = mock_es
        self.sparse_retriever = SparseRetriever()
        self.sparse_retriever.es = mock_es
    
    def tearDown(self):
        self.patcher.stop()

    @patch('retrieval.sparse_retriever.Elasticsearch')
    def test_initialize_elasticsearch(self, mock_elasticsearch):
        """Test Elasticsearch initialization."""
        mock_es = Mock()
        mock_es.ping.return_value = True
        mock_elasticsearch.return_value = mock_es
        
        es_client = self.sparse_retriever._initialize_elasticsearch()
        
        self.assertIsNotNone(es_client)
        mock_elasticsearch.assert_called_once()
    
    @patch('retrieval.sparse_retriever.Elasticsearch')
    def test_retrieve(self, mock_elasticsearch):
        """Test sparse retrieval."""
        # Mock Elasticsearch response
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "doc1",
                        "_source": {
                            "text": "Health insurance covers knee surgery",
                            "metadata": {"domain": "health"},
                            "domain": "health"
                        },
                        "_score": 0.8,
                        "highlight": {
                            "text": ["Health insurance covers <em>knee surgery</em>"]
                        }
                    }
                ]
            }
        }
        
        mock_es = Mock()
        mock_es.search.return_value = mock_response
        mock_elasticsearch.return_value = mock_es
        
        self.sparse_retriever.es = mock_es
        
        results = self.sparse_retriever.retrieve("knee surgery", top_k=5)
        
        self.assertIsInstance(results, list)
        if results:
            self.assertIn("segment_id", results[0])
            self.assertIn("text", results[0])
            self.assertIn("score", results[0])
    
    def test_index_document(self):
        """Test document indexing."""
        with patch.object(self.sparse_retriever, 'es') as mock_es:
            mock_es.index.return_value = {"result": "created"}
            
            success = self.sparse_retriever.index_document(
                "doc1", 
                "Test document", 
                {"domain": "health"}, 
                "health"
            )
            
            self.assertTrue(success)
            mock_es.index.assert_called_once()


class TestDenseRetriever(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.dense_retriever = DenseRetriever()
        self.dense_retriever.clear_index()  # Ensure a fresh FAISS index for each test
    
    def test_create_new_index(self):
        """Test creating new FAISS index."""
        self.dense_retriever._create_new_index()
        
        self.assertIsNotNone(self.dense_retriever.index)
        if self.dense_retriever.index is not None:
            self.assertEqual(self.dense_retriever.index.d, 384)
    
    def test_normalize_embeddings(self):
        """Test embedding normalization."""
        embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        
        normalized = self.dense_retriever._normalize_embeddings(embeddings)
        
        self.assertEqual(normalized.shape, embeddings.shape)
        # Check that vectors are normalized
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3))
    
    def test_add_documents(self):
        """Test adding documents to FAISS index."""
        embeddings = [np.random.rand(384).astype(np.float32) for _ in range(3)]
        metadata = [
            {"segment_id": f"doc{i}", "text": f"Document {i}", "domain": "health"}
            for i in range(3)
        ]
        
        success = self.dense_retriever.add_documents(embeddings, metadata)
        
        self.assertTrue(success)
        if self.dense_retriever.index is not None:
            self.assertEqual(self.dense_retriever.index.ntotal, 3)
    
    def test_retrieve(self):
        """Test dense retrieval."""
        # Add some test documents
        embeddings = [np.random.rand(384).astype(np.float32) for _ in range(3)]
        metadata = [
            {"segment_id": f"doc{i}", "text": f"Document {i}", "domain": "health"}
            for i in range(3)
        ]
        self.dense_retriever.add_documents(embeddings, metadata)
        
        # Test retrieval
        query_embedding = np.random.rand(384).astype(np.float32)
        results = self.dense_retriever.retrieve(query_embedding, top_k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        if results:
            self.assertIn("segment_id", results[0])
            self.assertIn("text", results[0])
            self.assertIn("score", results[0])
    
    def test_save_and_load_index(self):
        """Test saving and loading FAISS index."""
        # Add test documents
        embeddings = [np.random.rand(384).astype(np.float32)]
        metadata = [{"segment_id": "doc1", "text": "Test document"}]
        self.dense_retriever.add_documents(embeddings, metadata)
        
        # Save index
        success = self.dense_retriever.save_index()
        self.assertTrue(success)
        
        # Create new retriever and load index
        new_retriever = DenseRetriever()
        if new_retriever.index is not None:
            self.assertEqual(new_retriever.index.ntotal, 1)


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.sparse_retriever = Mock()
        self.dense_retriever = Mock()
        self.hybrid_retriever = HybridRetriever(self.sparse_retriever, self.dense_retriever)
    
    def test_combine_results(self):
        """Test combining sparse and dense results."""
        sparse_hits = [
            {
                "segment_id": "doc1",
                "text": "Health insurance covers surgery",
                "score": 0.8,
                "domain": "health"
            }
        ]
        
        dense_hits = [
            {
                "segment_id": "doc1",
                "text": "Health insurance covers surgery",
                "score": 0.9,
                "domain": "health"
            }
        ]
        
        combined = self.hybrid_retriever._combine_results(sparse_hits, dense_hits)
        
        self.assertIsInstance(combined, list)
        if combined:
            self.assertIn("combined_score", combined[0])
            self.assertIn("sparse_score", combined[0])
            self.assertIn("dense_score", combined[0])
    
    def test_retrieve(self):
        """Test hybrid retrieval."""
        # Mock sparse and dense retrievers
        self.sparse_retriever.retrieve.return_value = [
            {"segment_id": "doc1", "text": "Test", "score": 0.8, "domain": "health"}
        ]
        self.dense_retriever.retrieve.return_value = [
            {"segment_id": "doc1", "text": "Test", "score": 0.9, "domain": "health"}
        ]
        
        results = self.hybrid_retriever.retrieve("test query", np.random.rand(384))
        
        self.assertIsInstance(results, list)
        self.sparse_retriever.retrieve.assert_called_once()
        self.dense_retriever.retrieve.assert_called_once()
    
    def test_filter_by_domain(self):
        """Test domain filtering."""
        results = [
            {"domain": "health", "text": "Health document"},
            {"domain": "motor", "text": "Motor document"},
            {"domain": "health", "text": "Another health document"}
        ]
        
        filtered = self.hybrid_retriever._filter_by_domain(results, "health")
        
        self.assertEqual(len(filtered), 2)
        for result in filtered:
            self.assertEqual(result["domain"], "health")
    
    def test_update_weights(self):
        """Test updating hybrid weights."""
        self.hybrid_retriever.update_weights(0.4, 0.6)
        
        self.assertEqual(self.hybrid_retriever.weight_sparse, 0.4)
        self.assertEqual(self.hybrid_retriever.weight_dense, 0.6)


class TestDeduplicator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.deduplicator = Deduplicator()
    
    def test_dedupe_segments(self):
        """Test segment deduplication."""
        segments = [
            {
                "segment_id": "doc1",
                "text": "Health insurance covers surgery",
                "embedding": np.random.rand(384),
                "score": 0.8
            },
            {
                "segment_id": "doc2",
                "text": "Health insurance covers surgery",  # Near duplicate
                "embedding": np.random.rand(384),
                "score": 0.9
            },
            {
                "segment_id": "doc3",
                "text": "Motor insurance covers accidents",
                "embedding": np.random.rand(384),
                "score": 0.7
            }
        ]
        
        unique_segments = self.deduplicator.dedupe_segments(segments)
        
        self.assertIsInstance(unique_segments, list)
        self.assertLessEqual(len(unique_segments), len(segments))
    
    def test_select_representative(self):
        """Test selecting representative from cluster."""
        cluster_segments = [
            {"text": "Document 1", "combined_score": 0.8},
            {"text": "Document 2", "combined_score": 0.9},
            {"text": "Document 3", "combined_score": 0.7}
        ]
        
        representative = self.deduplicator._select_representative(cluster_segments)
        
        self.assertIsInstance(representative, dict)
        self.assertIn("text", representative)
    
    def test_calculate_text_similarity(self):
        """Test text similarity calculation."""
        text1 = "Health insurance covers surgery"
        text2 = "Health insurance covers surgery"
        text3 = "Motor insurance covers accidents"
        
        similarity1 = self.deduplicator._calculate_text_similarity(text1, text2)
        similarity2 = self.deduplicator._calculate_text_similarity(text1, text3)
        
        self.assertGreater(similarity1, 0.8)  # High similarity for same text
        self.assertLess(similarity2, 0.5)     # Low similarity for different text
    
    def test_dedupe_by_text_similarity(self):
        """Test text-based deduplication."""
        segments = [
            {"text": "Health insurance covers surgery"},
            {"text": "Health insurance covers surgery"},  # Duplicate
            {"text": "Motor insurance covers accidents"}
        ]
        
        unique_segments = self.deduplicator.dedupe_by_text_similarity(segments, threshold=0.8)
        
        self.assertIsInstance(unique_segments, list)
        self.assertLessEqual(len(unique_segments), len(segments))
    
    def test_get_deduplication_stats(self):
        """Test deduplication statistics."""
        original_count = 10
        unique_count = 7
        
        stats = self.deduplicator.get_deduplication_stats(original_count, unique_count)
        
        self.assertIn("original_count", stats)
        self.assertIn("unique_count", stats)
        self.assertIn("duplicates_removed", stats)
        self.assertIn("reduction_ratio", stats)
        self.assertEqual(stats["duplicates_removed"], 3)
        self.assertEqual(stats["reduction_ratio"], 0.3)


class TestRetrievalIntegration(unittest.TestCase):
    """Integration tests for the complete retrieval pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = {
            "sparse": {
                "engine": "elasticsearch",
                "index_name": "test-sparse-index",
                "host": "http://localhost:9200"
            },
            "dense": {
                "vector_store": "faiss",
                "index_name": "test-dense-index",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "hybrid": {
                "weight_sparse": 0.3,
                "weight_dense": 0.7,
                "top_k": 5
            },
            "dedupe": {
                "method": "agglomerative",
                "similarity_threshold": 0.85
            },
            "prompt_templates": {
                "health": {
                    "template": "Health claim: {procedure}",
                    "fallback": "Health insurance claim"
                }
            }
        }
        
        # Save config to file
        self.config_path = os.path.join(self.temp_dir, "test_retrieval_config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_retrieval_pipeline(self):
        """Test the complete retrieval pipeline."""
        # Initialize components
        prompt_builder = PromptBuilder(self.config_path)
        sparse_retriever = SparseRetriever(self.config_path)
        dense_retriever = DenseRetriever(self.config_path)
        hybrid_retriever = HybridRetriever(sparse_retriever, dense_retriever, self.config_path)
        deduplicator = Deduplicator(self.config_path)
        
        # Test prompt building
        raw_query = "What is covered for knee surgery?"
        slots = {"procedure": "knee surgery"}
        domain = "health"
        
        prompt = prompt_builder.build_search_prompt(raw_query, slots, domain)
        self.assertIn("knee surgery", prompt)
        
        # Test retrieval (mocked)
        with patch.object(sparse_retriever, 'retrieve') as mock_sparse:
            with patch.object(dense_retriever, 'retrieve') as mock_dense:
                mock_sparse.return_value = [
                    {"segment_id": "doc1", "text": "Health covers surgery", "score": 0.8, "domain": "health"}
                ]
                mock_dense.return_value = [
                    {"segment_id": "doc1", "text": "Health covers surgery", "score": 0.9, "domain": "health"}
                ]
                
                results = hybrid_retriever.retrieve(prompt, np.random.rand(384), domain)
                self.assertIsInstance(results, list)
        
        # Test deduplication
        segments = [
            {"text": "Health insurance covers surgery", "embedding": np.random.rand(384)},
            {"text": "Health insurance covers surgery", "embedding": np.random.rand(384)},
            {"text": "Motor insurance covers accidents", "embedding": np.random.rand(384)}
        ]
        
        unique_segments = deduplicator.dedupe_segments(segments)
        self.assertLessEqual(len(unique_segments), len(segments))


if __name__ == '__main__':
    unittest.main()