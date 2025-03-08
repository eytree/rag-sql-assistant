#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge base for the RAG-powered SQL Assistant.

This module manages the retrieval and formatting of relevant context
from the database schema for use in the RAG system. It serves as the
intermediary between raw schema data and the LLM module.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from .database import get_database
from .embeddings import get_embedding_manager
from .logging_config import register_logger
from .timing import get_timing_manager

# Register this module's logger
LOGGER_NAME = 'knowledge_base'
register_logger(LOGGER_NAME)
logger = logging.getLogger(LOGGER_NAME)

class KnowledgeBase:
    """
    Knowledge base for retrieving and formatting schema information.
    """
    
    def __init__(self):
        """Initialize the knowledge base."""
        self.db = get_database()
        self.embedding_manager = get_embedding_manager()
        
        # Load schema once
        schema_path = Path("data/schema.json")
        if schema_path.exists():
            with open(schema_path, "r") as f:
                self.schema = json.load(f)
        else:
            self.schema = None
    
    def setup(self):
        """Set up the knowledge base and generate embeddings if needed."""
        # Generate embeddings for schema if they don't exist
        if not hasattr(self.embedding_manager, "index") or self.embedding_manager.index is None:
            print("Setting up knowledge base...")
            if self.schema:
                self.embedding_manager.generate_schema_embeddings(self.schema)
            else:
                raise ValueError("Schema data not found. Please ensure data/schema.json exists.")
    
    def search(self, query: str, top_k: int = 7) -> List[Dict[str, Any]]:
        """
        Search for schema elements most relevant to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with relevant schema elements
        """
        return self.embedding_manager.search(query, top_k)
    
    def extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract mentioned tables and columns from a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with extracted entities: {"tables": [...], "columns": [...]}
        """
        entities = {
            "tables": [],
            "columns": []
        }
        
        query_lower = query.lower()
        
        # Extract tables based on names in schema
        if self.schema:
            for table in self.schema["tables"]:
                table_name = table["name"].lower()
                if table_name in query_lower or table_name.rstrip('s') in query_lower:
                    entities["tables"].append(table["name"])
                
                # Check for table name variations (singular/plural forms)
                if table_name.endswith('s') and table_name[:-1] in query_lower:
                    if table["name"] not in entities["tables"]:
                        entities["tables"].append(table["name"])
                elif table_name + 's' in query_lower:
                    if table["name"] not in entities["tables"]:
                        entities["tables"].append(table["name"])
                
                # Extract columns for matched tables
                for column in table["columns"]:
                    column_name = column["name"].lower()
                    if column_name in query_lower:
                        entities["columns"].append(f"{table['name']}.{column['name']}")
        
        return entities
    
    def get_table_context(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed context for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table context information
        """
        context = {}
        
        # Get basic table info
        table_info = self.db.get_table_schema(table_name)
        if table_info:
            context["table"] = table_info
            
            # Get relationships
            context["relationships"] = self.db.get_table_relationships(table_name)
            
            # Sample data
            try:
                sample_data = self.db.execute_query(f"SELECT * FROM {table_name} LIMIT 2")
                context["sample_data"] = sample_data
            except Exception:
                context["sample_data"] = []
        
        return context
    
    def retrieve_context(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant context for a natural language query.
        
        This is the main RAG retrieval function that combines entity extraction
        and semantic search to gather relevant context for the LLM.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with retrieved context
        """
        timer = get_timing_manager().timer
        
        # Search for relevant schema elements
        with timer("search_embeddings"):
            results = self.embedding_manager.search(query, top_k=5)
        
        # Extract relevant tables
        tables = []
        relationships = []
        similar_queries = []
        performance_notes = []
        
        with timer("process_search_results"):
            for result in results:
                if result["type"] == "table":
                    # Get full table info
                    table_info = self.db.get_table_schema(result["name"])
                    if table_info:
                        tables.append(table_info)
                        
                        # Get relationships for this table
                        table_rels = self.db.get_table_relationships(result["name"])
                        relationships.extend(table_rels)
                
                elif result["type"] == "column":
                    # Get the table this column belongs to
                    table_info = self.db.get_table_schema(result["table"])
                    if table_info and table_info not in tables:
                        tables.append(table_info)
                        
                        # Get relationships for this table
                        table_rels = self.db.get_table_relationships(result["table"])
                        relationships.extend(table_rels)
                
                elif result["type"] == "relationship":
                    relationships.append({
                        "from_table": result["from_table"],
                        "from_column": result["from_column"],
                        "to_table": result["to_table"],
                        "to_column": result["to_column"],
                        "relationship": result["relationship_type"],
                        "description": result["description"]
                    })
                    
                    # Get info for both tables in the relationship
                    for table_name in [result["from_table"], result["to_table"]]:
                        table_info = self.db.get_table_schema(table_name)
                        if table_info and table_info not in tables:
                            tables.append(table_info)
                
                elif result["type"] == "query":
                    similar_queries.append({
                        "name": result["name"],
                        "description": result["description"],
                        "sql": result["sql"]
                    })
        
        # Deduplicate relationships
        unique_relationships = []
        seen = set()
        for rel in relationships:
            key = f"{rel['from_table']}.{rel['from_column']}->{rel['to_table']}.{rel['to_column']}"
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return {
            "tables": tables,
            "relationships": unique_relationships,
            "similar_queries": similar_queries,
            "performance_notes": performance_notes
        }
    
    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format retrieved context into a string suitable for the LLM prompt.
        
        Args:
            context: Retrieved context dictionary
            
        Returns:
            Formatted context string
        """
        formatted = []
        
        # Add tables
        if context["tables"]:
            formatted.append("## Tables")
            for table in context["tables"]:
                formatted.append(f"### {table['name']}")
                formatted.append(f"Description: {table.get('description', 'No description')}")
                
                # Add columns
                formatted.append("Columns:")
                for column in table["columns"]:
                    constraints = column.get("constraints", "")
                    formatted.append(f"- {column['name']} ({column['type']}) {constraints}: {column.get('description', '')}")
                
                # Add indexes if available
                if "indexes" in table and table["indexes"]:
                    formatted.append("Indexes:")
                    for idx in table["indexes"]:
                        unique = "UNIQUE " if idx.get("unique", False) else ""
                        formatted.append(f"- {unique}INDEX {idx['name']} on ({', '.join(idx['columns'])})")
                
                formatted.append("")  # Empty line for readability
        
        # Add relationships
        if context["relationships"]:
            formatted.append("## Relationships")
            for rel in context["relationships"]:
                rel_type = rel.get("relationship_type", "").replace("-", " to ")
                formatted.append(f"- {rel.get('from_table')}.{rel.get('from_column')} → {rel.get('to_table')}.{rel.get('to_column')} ({rel_type}): {rel.get('description', '')}")
            formatted.append("")
        
        # Add similar queries
        if context["similar_queries"]:
            formatted.append("## Similar Queries")
            for query in context["similar_queries"]:
                formatted.append(f"### {query['name']}")
                formatted.append(f"Description: {query['description']}")
                formatted.append("```sql")
                formatted.append(query["sql"])
                formatted.append("```")
                formatted.append("")
        
        # Add performance notes
        if context["performance_notes"]:
            formatted.append("## Performance Notes")
            for note in context["performance_notes"]:
                formatted.append(f"- {note}")
            formatted.append("")
        
        return "\n".join(formatted)

def get_knowledge_base() -> KnowledgeBase:
    """
    Get the knowledge base instance.
    Creates the instance on first call.
    
    Returns:
        KnowledgeBase instance
    """
    if not hasattr(get_knowledge_base, 'knowledge_base'):
        get_knowledge_base.knowledge_base = KnowledgeBase()
    return get_knowledge_base.knowledge_base