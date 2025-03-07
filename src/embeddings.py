#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vector embeddings module for the RAG-powered SQL Assistant.

This module handles the creation and management of vector embeddings
for database schema elements, enabling semantic search for relevant
context during RAG retrieval.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import logging

from .logging_config import register_logger

# Register this module's logger
LOGGER_NAME = 'embeddings'
register_logger(LOGGER_NAME)
logger = logging.getLogger(LOGGER_NAME)

# Paths
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)

# Define embedding model
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class EmbeddingManager:
    """Manager for creating and searching embeddings for database schema elements"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.embedding_map = {}
        self.texts = []
        
        # Check if embeddings already exist
        self.embedding_file = EMBEDDINGS_DIR / f"schema_embeddings_{model_name.replace('/', '_')}.npz"
        self.mapping_file = EMBEDDINGS_DIR / f"schema_mapping_{model_name.replace('/', '_')}.json"
        
        # Load existing embeddings if available
        if self.embedding_file.exists() and self.mapping_file.exists():
            self._load_embeddings()
    
    def _load_model(self):
        """Load the embedding model if not already loaded"""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def _load_embeddings(self):
        """Load existing embeddings from disk"""
        print(f"Loading existing embeddings from {self.embedding_file}")
        
        # Load the embeddings
        loaded = np.load(self.embedding_file)
        embeddings = loaded["embeddings"]
        
        # Load the mapping file
        with open(self.mapping_file, "r") as f:
            data = json.load(f)
            self.embedding_map = data["mapping"]
            self.texts = data["texts"]
        
        # Create FAISS index
        self._create_index(embeddings)
    
    def _create_index(self, embeddings: np.ndarray):
        """
        Create a FAISS index for fast similarity search.
        
        Args:
            embeddings: Numpy array of embeddings
        """
        embedding_dimension = embeddings.shape[1]
        
        # Create a flat index - simple but effective for small to medium datasets
        self.index = faiss.IndexFlatL2(embedding_dimension)
        
        # Add the embeddings to the index
        self.index.add(embeddings.astype(np.float32))
    
    def generate_schema_embeddings(self, schema_data: Dict[str, Any]):
        """
        Generate embeddings for all relevant elements in the database schema.
        
        Args:
            schema_data: Database schema information
        """
        self._load_model()
        
        # Extract relevant text elements from the schema
        texts = []
        mapping = {}
        
        print("Generating schema embeddings...")
        
        # Add database description
        if "description" in schema_data:
            text = f"DATABASE: {schema_data['database_name']} - {schema_data['description']}"
            texts.append(text)
            mapping[len(texts) - 1] = {
                "type": "database",
                "name": schema_data["database_name"],
                "description": schema_data["description"]
            }
        
        # Process tables
        for table in schema_data["tables"]:
            # Table description
            table_text = f"TABLE: {table['name']} - {table['description']}"
            texts.append(table_text)
            mapping[len(texts) - 1] = {
                "type": "table",
                "name": table["name"],
                "description": table["description"]
            }
            
            # Columns for each table
            for column in table["columns"]:
                constraints = column.get("constraints", "")
                description = column.get("description", "")
                column_text = f"COLUMN: {table['name']}.{column['name']} ({column['type']}) {constraints} - {description}"
                texts.append(column_text)
                mapping[len(texts) - 1] = {
                    "type": "column",
                    "table": table["name"],
                    "name": column["name"],
                    "data_type": column["type"],
                    "constraints": constraints,
                    "description": description
                }
        
        # Process relationships
        if "relationships" in schema_data:
            for relationship in schema_data["relationships"]:
                rel_text = f"RELATIONSHIP: {relationship['from_table']}.{relationship['from_column']} -> {relationship['to_table']}.{relationship['to_column']} ({relationship['relationship']}) - {relationship['description']}"
                texts.append(rel_text)
                mapping[len(texts) - 1] = {
                    "type": "relationship",
                    "from_table": relationship["from_table"],
                    "from_column": relationship["from_column"],
                    "to_table": relationship["to_table"],
                    "to_column": relationship["to_column"],
                    "relationship_type": relationship["relationship"],
                    "description": relationship["description"]
                }
        
        # Process common queries if available
        if "common_queries" in schema_data:
            for query in schema_data["common_queries"]:
                query_text = f"QUERY: {query['name']} - {query['description']}"
                texts.append(query_text)
                mapping[len(texts) - 1] = {
                    "type": "query",
                    "name": query["name"],
                    "description": query["description"],
                    "sql": query["sql"]
                }
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} schema elements...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Save the texts and mapping
        self.texts = texts
        self.embedding_map = mapping
        
        # Create the index
        self._create_index(embeddings)
        
        # Save to disk
        self._save_embeddings(embeddings)
        
        print(f"Generated and saved embeddings for {len(texts)} schema elements")
    
    def _save_embeddings(self, embeddings: np.ndarray):
        """
        Save embeddings and mapping to disk.
        
        Args:
            embeddings: Numpy array of embeddings
        """
        # Save embeddings
        np.savez(
            self.embedding_file, 
            embeddings=embeddings
        )
        
        # Save mapping
        with open(self.mapping_file, "w") as f:
            json.dump({
                "model": self.model_name,
                "mapping": self.embedding_map,
                "texts": self.texts
            }, f, indent=2)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for schema elements most relevant to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with relevant schema elements and similarity scores
        """
        if self.index is None:
            raise ValueError("No embeddings available. Please generate embeddings first.")
        
        # Load model if needed
        self._load_model()
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                result = {
                    "text": self.texts[idx],
                    "similarity": 1.0 - (distances[0][i] / 2),  # Convert L2 distance to similarity score
                    **self.embedding_map[str(idx)]  # Add the metadata
                }
                results.append(result)
        
        return results

def get_embedding_manager() -> EmbeddingManager:
    """
    Get the embedding manager instance.
    Creates the instance on first call.
    
    Returns:
        EmbeddingManager instance
    """
    if not hasattr(get_embedding_manager, 'embedding_manager'):
        get_embedding_manager.embedding_manager = EmbeddingManager()
    return get_embedding_manager.embedding_manager