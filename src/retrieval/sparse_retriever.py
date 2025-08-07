import yaml
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import logging


class SparseRetriever:
    def __init__(self, config_path: str = "retrieval_config.yaml"):
        """Initialize sparse retriever with Elasticsearch configuration."""
        self.config = self._load_config(config_path)
        self.sparse_config = self.config.get("sparse", {})
        
        self.host = self.sparse_config.get("host", "http://localhost:9200")
        self.index_name = self.sparse_config.get("index_name", "claims-sparse-index")
        self.timeout = self.sparse_config.get("timeout", 30)
        self.max_retries = self.sparse_config.get("max_retries", 3)
        
        # Initialize Elasticsearch client
        self.es = self._initialize_elasticsearch()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load retrieval configuration."""
        import os
        config_path = os.path.abspath(config_path)
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            print(f"Error loading retrieval config: {e}")
            return {}
    
    def _initialize_elasticsearch(self) -> Elasticsearch:
        """Initialize Elasticsearch client."""
        try:
            es = Elasticsearch(
                [self.host],
                request_timeout=self.timeout,
                retry_on_timeout=True
            )
            
            # Test connection
            if es.ping():
                logging.info(f"Connected to Elasticsearch at {self.host}")
                return es
            else:
                logging.error(f"Failed to connect to Elasticsearch at {self.host}")
                return None
                
        except Exception as e:
            logging.error(f"Error initializing Elasticsearch: {e}")
            return None
    
    def retrieve(self, prompt: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 sparse retrieval.
        
        Args:
            prompt: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.es:
            logging.error("Elasticsearch client not initialized")
            return []
        
        try:
            # Build BM25 query
            query_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "text": {
                                        "query": prompt,
                                        "operator": "or",
                                        "boost": 2.0
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "text": {
                                        "query": prompt,
                                        "boost": 3.0
                                    }
                                }
                            }
                        ],
                        "filter": [
                            {
                                "term": {
                                    "segment_type": "clause"
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "_source": ["text", "metadata", "segment_id", "domain"],
                "highlight": {
                    "fields": {
                        "text": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
                    }
                }
            }
            
            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=query_body
            )
            
            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "segment_id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "domain": hit["_source"].get("domain", "unknown"),
                    "score": hit["_score"],
                    "highlights": hit.get("highlight", {}).get("text", [])
                }
                results.append(result)
            
            return results
            
        except ConnectionError as e:
            logging.error(f"Elasticsearch connection error: {e}")
            return []
        except NotFoundError as e:
            logging.error(f"Index {self.index_name} not found: {e}")
            return []
        except Exception as e:
            logging.error(f"Error during sparse retrieval: {e}")
            return []
    
    def retrieve_by_domain(self, prompt: str, domain: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents filtered by domain.
        
        Args:
            prompt: Search query
            domain: Target domain
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        if not self.es:
            return []
        
        try:
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "text": {
                                        "query": prompt,
                                        "operator": "or"
                                    }
                                }
                            },
                            {
                                "term": {
                                    "domain": domain
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "_source": ["text", "metadata", "segment_id", "domain"]
            }
            
            response = self.es.search(
                index=self.index_name,
                body=query_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "segment_id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "domain": hit["_source"].get("domain", "unknown"),
                    "score": hit["_score"]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error during domain-specific retrieval: {e}")
            return []
    
    def index_document(self, doc_id: str, text: str, metadata: Dict[str, Any], domain: str = "unknown") -> bool:
        """
        Index a document in Elasticsearch.
        
        Args:
            doc_id: Document ID
            text: Document text
            metadata: Document metadata
            domain: Document domain
            
        Returns:
            True if successful, False otherwise
        """
        if not self.es:
            return False
        
        try:
            doc_body = {
                "text": text,
                "metadata": metadata,
                "domain": domain,
                "segment_type": metadata.get("segment_type", "clause")
            }
            
            self.es.index(
                index=self.index_name,
                id=doc_id,
                body=doc_body
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Error indexing document: {e}")
            return False
    
    def create_index(self) -> bool:
        """Create the Elasticsearch index with proper mapping."""
        if not self.es:
            return False
        
        try:
            # Check if index exists
            if self.es.indices.exists(index=self.index_name):
                logging.info(f"Index {self.index_name} already exists")
                return True
            
            # Create index with mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                            "search_analyzer": "standard"
                        },
                        "metadata": {
                            "type": "object"
                        },
                        "domain": {
                            "type": "keyword"
                        },
                        "segment_type": {
                            "type": "keyword"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
            
            self.es.indices.create(
                index=self.index_name,
                body=mapping
            )
            
            logging.info(f"Created index {self.index_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            return False
    
    def delete_index(self) -> bool:
        """Delete the Elasticsearch index."""
        if not self.es:
            return False
        
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logging.info(f"Deleted index {self.index_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if not self.es:
            return {}
        
        try:
            stats = self.es.indices.stats(index=self.index_name)
            return {
                "index_name": self.index_name,
                "document_count": stats["indices"][self.index_name]["total"]["docs"]["count"],
                "size": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
            }
            
        except Exception as e:
            logging.error(f"Error getting index stats: {e}")
            return {}