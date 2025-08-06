import yaml
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import uuid
from dataclasses import dataclass


@dataclass
class IndexedDocument:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class VectorIndexer:
    def __init__(self, config_path: str = "ingestion_config.yaml"):
        """Initialize vector indexer with configuration."""
        self.config = self._load_config(config_path)
        self.vector_db_config = self.config.get("ingestion", {}).get("vector_db", {})
        
        self.db_type = self.vector_db_config.get("type", "chroma")
        self.collection_name = self.vector_db_config.get("collection_name", "claims_documents")
        self.similarity_metric = self.vector_db_config.get("similarity_metric", "cosine")
        
        # Initialize vector database client
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load ingestion configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _initialize_client(self):
        """Initialize the vector database client."""
        if self.db_type == "chroma":
            return chromadb.Client()
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _get_or_create_collection(self):
        """Get or create the document collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll handle embeddings ourselves
            )
            return collection
        except:
            # Create new collection if it doesn't exist
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Claims documents collection"}
            )
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """
        Add documents to the vector index.
        
        Args:
            documents: List of IndexedDocument objects
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        try:
            # Prepare data for insertion
            ids = []
            texts = []
            embeddings = []
            metadatas = []
            
            for doc in documents:
                doc_id = doc.id if doc.id else str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(doc.text)
                embeddings.append(doc.embedding.tolist())  # Convert numpy array to list
                metadatas.append(doc.metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return ids
            
        except Exception as e:
            print(f"Error adding documents to index: {e}")
            return []
    
    def add_single_document(self, document: IndexedDocument) -> Optional[str]:
        """
        Add a single document to the vector index.
        
        Args:
            document: IndexedDocument object
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            doc_id = document.id if document.id else str(uuid.uuid4())
            
            self.collection.add(
                ids=[doc_id],
                documents=[document.text],
                embeddings=[document.embedding.tolist()],
                metadatas=[document.metadata]
            )
            
            return doc_id
            
        except Exception as e:
            print(f"Error adding single document to index: {e}")
            return None
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      filter_metadata: Dict[str, Any] = None) -> List[Tuple[IndexedDocument, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top similar documents to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            # Prepare query
            query_embedding_list = [query_embedding.tolist()]
            
            # Perform search
            results = self.collection.query(
                query_embeddings=query_embedding_list,
                n_results=top_k,
                where=filter_metadata
            )
            
            # Process results
            similar_documents = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    text = results['documents'][0][i]
                    embedding = np.array(results['embeddings'][0][i])
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score
                    similarity = 1.0 - distance  # Assuming cosine distance
                    
                    document = IndexedDocument(
                        id=doc_id,
                        text=text,
                        embedding=embedding,
                        metadata=metadata
                    )
                    
                    similar_documents.append((document, similarity))
            
            return similar_documents
            
        except Exception as e:
            print(f"Error searching similar documents: {e}")
            return []
    
    def search_by_text(self, query_text: str, top_k: int = 5, 
                      filter_metadata: Dict[str, Any] = None) -> List[Tuple[IndexedDocument, float]]:
        """
        Search for documents by text query.
        
        Args:
            query_text: Text query
            top_k: Number of top similar documents to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            # Perform text search
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Process results
            similar_documents = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    text = results['documents'][0][i]
                    embedding = np.array(results['embeddings'][0][i])
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score
                    similarity = 1.0 - distance  # Assuming cosine distance
                    
                    document = IndexedDocument(
                        id=doc_id,
                        text=text,
                        embedding=embedding,
                        metadata=metadata
                    )
                    
                    similar_documents.append((document, similarity))
            
            return similar_documents
            
        except Exception as e:
            print(f"Error searching by text: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[IndexedDocument]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            IndexedDocument object or None if not found
        """
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results['ids']:
                return IndexedDocument(
                    id=results['ids'][0],
                    text=results['documents'][0],
                    embedding=np.array(results['embeddings'][0]),
                    metadata=results['metadatas'][0]
                )
            else:
                return None
                
        except Exception as e:
            print(f"Error retrieving document by ID: {e}")
            return None
    
    def update_document(self, doc_id: str, text: str = None, 
                       embedding: np.ndarray = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID
            text: New text content
            embedding: New embedding vector
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current document
            current_doc = self.get_document_by_id(doc_id)
            if not current_doc:
                return False
            
            # Update fields
            new_text = text if text is not None else current_doc.text
            new_embedding = embedding if embedding is not None else current_doc.embedding
            new_metadata = metadata if metadata is not None else current_doc.metadata
            
            # Update in collection
            self.collection.update(
                ids=[doc_id],
                documents=[new_text],
                embeddings=[new_embedding.tolist()],
                metadatas=[new_metadata]
            )
            
            return True
            
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'db_type': self.db_type,
                'similarity_metric': self.similarity_metric
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.collection.delete(where={})
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False 