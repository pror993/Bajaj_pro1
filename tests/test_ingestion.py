import unittest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import ingestion components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingestion.loader import DocumentLoader
from ingestion.ocr import OCRProcessor
from ingestion.segmenter import DocumentSegmenter, DocumentSegment
from ingestion.embedder import DocumentEmbedder, EmbeddingResult
from ingestion.indexer import VectorIndexer, IndexedDocument


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader()
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test text file
        self.test_text_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_text_file, 'w') as f:
            f.write("This is a test document for claims processing.")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_text_file(self):
        """Test loading a text file."""
        result = self.loader.load_document(self.test_text_file)
        
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['file_type'], 'text')
        self.assertIn('test document', result['content'])
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_document("nonexistent.txt")
    
    def test_unsupported_format(self):
        """Test loading an unsupported file format."""
        unsupported_file = os.path.join(self.temp_dir, "test.xyz")
        with open(unsupported_file, 'w') as f:
            f.write("test")
        
        with self.assertRaises(ValueError):
            self.loader.load_document(unsupported_file)


class TestOCRProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.ocr = OCRProcessor()
    
    @patch('ingestion.ocr.pytesseract')
    @patch('ingestion.ocr.Image')
    def test_extract_text_from_image(self, mock_image, mock_tesseract):
        """Test OCR text extraction from image."""
        # Mock OCR results
        mock_tesseract.image_to_data.return_value = {
            'text': ['Hello', 'World'],
            'conf': [90, 85],
            'left': [10, 50],
            'top': [20, 20],
            'width': [40, 30],
            'height': [15, 15]
        }
        
        # Mock image processing
        mock_img = Mock()
        mock_image.open.return_value = mock_img
        mock_img.__array__ = Mock(return_value=np.zeros((100, 100, 3)))
        
        result = self.ocr.extract_text_from_image("test.jpg")
        
        self.assertIn('text', result)
        self.assertIn('text_blocks', result)
        self.assertIn('metadata', result)
        self.assertEqual(len(result['text_blocks']), 2)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a mock image
        mock_img = Mock()
        mock_img.__array__ = Mock(return_value=np.zeros((100, 100, 3)))
        
        processed = self.ocr._preprocess_image(mock_img)
        self.assertIsNotNone(processed)


class TestDocumentSegmenter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.segmenter = DocumentSegmenter()
    
    def test_extract_tables(self):
        """Test table extraction."""
        content = """
        | Name | Age | City |
        |------|-----|------|
        | John | 25  | NYC  |
        | Jane | 30  | LA   |
        """
        
        segments = self.segmenter._extract_tables(content)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].segment_type, 'table')
    
    def test_extract_lists(self):
        """Test list extraction."""
        content = """
        - Item 1
        - Item 2
        - Item 3
        
        1. Numbered item 1
        2. Numbered item 2
        """
        
        segments = self.segmenter._extract_lists(content)
        self.assertEqual(len(segments), 2)  # One bullet list, one numbered list
    
    def test_extract_clauses(self):
        """Test clause extraction."""
        content = """
        This is a regular paragraph.
        
        WHEREAS the parties agree to the following terms and conditions:
        The party shall provide the services as specified.
        
        Another paragraph here.
        """
        
        segments = self.segmenter._extract_clauses(content)
        self.assertGreater(len(segments), 0)
        
        # Check if clause is detected
        clause_segments = [s for s in segments if s.segment_type == 'clause']
        self.assertGreater(len(clause_segments), 0)
    
    def test_is_clause(self):
        """Test clause detection."""
        clause_text = "WHEREAS the parties agree to the terms and conditions, the party shall provide services."
        non_clause_text = "This is just a regular sentence about something."
        
        self.assertTrue(self.segmenter._is_clause(clause_text))
        self.assertFalse(self.segmenter._is_clause(non_clause_text))
    
    def test_segment_document(self):
        """Test complete document segmentation."""
        content = """
        | Name | Age |
        |------|-----|
        | John | 25  |
        
        - List item 1
        - List item 2
        
        WHEREAS the parties agree to terms.
        
        Regular paragraph here.
        """
        
        segments = self.segmenter.segment_document(content)
        self.assertGreater(len(segments), 0)
        
        # Check segment types
        segment_types = [s.segment_type for s in segments]
        self.assertIn('table', segment_types)
        self.assertIn('list', segment_types)
        self.assertIn('clause', segment_types)


class TestDocumentEmbedder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.embedder = DocumentEmbedder()
    
    @patch('ingestion.embedder.SentenceTransformer')
    def test_generate_single_embedding(self, mock_transformer):
        """Test single embedding generation."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model
        
        result = self.embedder.generate_single_embedding("Test text")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "Test text")
        self.assertEqual(len(result.embedding), 384)
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        embedding1 = np.array([1, 0, 0])
        embedding2 = np.array([0, 1, 0])
        embedding3 = np.array([1, 0, 0])
        
        # Orthogonal vectors should have similarity close to 0
        similarity_orthogonal = self.embedder.calculate_similarity(embedding1, embedding2)
        self.assertLess(similarity_orthogonal, 0.1)
        
        # Same vectors should have similarity close to 1
        similarity_same = self.embedder.calculate_similarity(embedding1, embedding3)
        self.assertGreater(similarity_same, 0.9)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "  This   is   a   test   text   "
        processed = self.embedder.preprocess_text(text)
        
        self.assertEqual(processed, "This is a test text")
    
    def test_get_embedding_dimensions(self):
        """Test getting embedding dimensions."""
        dimensions = self.embedder.get_embedding_dimensions()
        self.assertIsInstance(dimensions, int)
        self.assertGreater(dimensions, 0)


class TestVectorIndexer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.indexer = VectorIndexer()
    
    @patch('ingestion.indexer.chromadb')
    def test_initialize_client(self, mock_chroma):
        """Test client initialization."""
        mock_client = Mock()
        mock_chroma.Client.return_value = mock_client
        
        client = self.indexer._initialize_client()
        self.assertIsNotNone(client)
    
    def test_create_indexed_document(self):
        """Test creating an IndexedDocument."""
        doc = IndexedDocument(
            id="test_id",
            text="Test document",
            embedding=np.random.rand(384),
            metadata={"source": "test"}
        )
        
        self.assertEqual(doc.id, "test_id")
        self.assertEqual(doc.text, "Test document")
        self.assertEqual(len(doc.embedding), 384)
        self.assertEqual(doc.metadata["source"], "test")
    
    @patch('ingestion.indexer.chromadb')
    def test_add_single_document(self, mock_chroma):
        """Test adding a single document to index."""
        # Mock collection
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.Client.return_value = mock_client
        
        doc = IndexedDocument(
            id="test_id",
            text="Test document",
            embedding=np.random.rand(384),
            metadata={"source": "test"}
        )
        
        doc_id = self.indexer.add_single_document(doc)
        self.assertIsNotNone(doc_id)
    
    def test_get_collection_stats(self):
        """Test getting collection statistics."""
        stats = self.indexer.get_collection_stats()
        self.assertIsInstance(stats, dict)


class TestIngestionIntegration(unittest.TestCase):
    """Integration tests for the complete ingestion pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test document
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, 'w') as f:
            f.write("""
            CLAIM DOCUMENT
            
            WHEREAS the insured party has submitted a claim for damages:
            
            - Vehicle: Honda Civic
            - Date: 2024-01-15
            - Location: New York City
            - Damage: Front bumper collision
            
            | Item | Amount |
            |------|--------|
            | Repair | $2,500 |
            | Rental | $300 |
            
            The insurance company shall process this claim according to the terms.
            """)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test the complete ingestion pipeline."""
        # 1. Load document
        loader = DocumentLoader()
        doc = loader.load_document(self.test_file)
        
        self.assertIn('content', doc)
        self.assertIn('CLAIM DOCUMENT', doc['content'])
        
        # 2. Segment document
        segmenter = DocumentSegmenter()
        segments = segmenter.segment_document(doc['content'])
        
        self.assertGreater(len(segments), 0)
        
        # Check for different segment types
        segment_types = [s.segment_type for s in segments]
        self.assertIn('table', segment_types)
        self.assertIn('list', segment_types)
        self.assertIn('clause', segment_types)
        
        # 3. Generate embeddings (mocked)
        with patch('ingestion.embedder.SentenceTransformer'):
            embedder = DocumentEmbedder()
            
            # Mock embedding generation
            texts = [s.content for s in segments]
            embeddings = embedder.generate_embeddings(texts)
            
            self.assertEqual(len(embeddings), len(segments))
        
        # 4. Index documents (mocked)
        with patch('ingestion.indexer.chromadb'):
            indexer = VectorIndexer()
            
            # Create indexed documents
            indexed_docs = [
                IndexedDocument(
                    id=f"doc_{i}",
                    text=emb.text,
                    embedding=emb.embedding,
                    metadata={"segment_type": segments[i].segment_type}
                )
                for i, emb in enumerate(embeddings)
            ]
            
            # Add to index
            doc_ids = indexer.add_documents(indexed_docs)
            self.assertEqual(len(doc_ids), len(indexed_docs))


if __name__ == '__main__':
    unittest.main() 