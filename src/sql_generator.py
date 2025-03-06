#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQL Generator for the RAG-powered SQL Assistant.

This module combines the Knowledge Base and LLM components to generate 
SQL queries from natural language questions, implementing the full
RAG pipeline.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import re
import sqlite3

from .knowledge_base import get_knowledge_base
from .llm import get_llm
from .database import get_database

class SQLGenerator:
    """
    Main SQL Generator class implementing the RAG pipeline.
    """
    
    def __init__(self):
        """Initialize the SQL Generator."""
        self.knowledge_base = get_knowledge_base()
        self.llm = get_llm()
        self.db = get_database()
        
        # Ensure knowledge base is set up
        self.knowledge_base.setup()
    
    def generate_sql(self, query: str) -> Dict[str, Any]:
        """
        Generate an SQL query from a natural language question using RAG.
        
        Args:
            query: Natural language question
            
        Returns:
            Dictionary with generated SQL, explanation, and performance info
        """
        # Step 1: Retrieve relevant context from knowledge base
        print(f"🔍 Retrieving context for query: '{query}'")
        context = self.knowledge_base.retrieve_context(query)
        
        # Step 2: Format context for prompt
        formatted_context = self.knowledge_base.format_context_for_prompt(context)
        
        # Step 3: Generate SQL using LLM with RAG context
        print("🧠 Generating SQL with RAG...")
        result = self.llm.generate_sql(query, formatted_context)
        
        # Step 4: Analyze query performance
        sql_query = result["sql"]
        performance_info = {}
        
        if sql_query:
            try:
                print("⚙️ Analyzing query performance...")
                performance_info = self.analyze_query(sql_query)
            except Exception as e:
                print(f"Error analyzing query: {e}")
                performance_info = {"error": str(e)}
        
        # Combine results
        result.update({
            "original_query": query,
            "context_summary": self._summarize_context(context),
            "performance_analysis": performance_info
        })
        
        return result
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the retrieved context.
        
        Args:
            context: Retrieved context dictionary
            
        Returns:
            Summary dictionary
        """
        return {
            "tables": [table["name"] for table in context["tables"]],
            "relationships": len(context["relationships"]),
            "similar_queries": len(context["similar_queries"]),
            "performance_notes": len(context["performance_notes"])
        }
    
    def analyze_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze the generated SQL query for performance issues.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Dictionary with performance analysis
        """
        try:
            # Get query explain plan
            explain_results = self.db.execute_explain(sql_query)
            
            # Analyze query
            analysis = self.db.analyze_query_performance(sql_query)
            
            # Validate query is syntactically correct
            self.db.execute_query("EXPLAIN " + sql_query)
            analysis["is_valid"] = True
            
            return analysis
        except sqlite3.Error as e:
            return {
                "is_valid": False,
                "error": str(e),
                "warnings": ["Query contains syntax errors"]
            }
    
    def execute_query(self, sql_query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Execute the generated SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            limit: Maximum number of rows to return
            
        Returns:
            Dictionary with query results and metadata
        """
        try:
            # Add LIMIT clause if not present
            if "LIMIT" not in sql_query.upper():
                sql_query = f"{sql_query} LIMIT {limit}"
            
            # Execute query
            results = self.db.execute_query(sql_query)
            
            return {
                "success": True,
                "row_count": len(results),
                "results": results[:limit],
                "sql": sql_query
            }
        except sqlite3.Error as e:
            return {
                "success": False,
                "error": str(e),
                "sql": sql_query
            }

# Singleton instance for easy import
sql_generator = SQLGenerator()

def get_sql_generator() -> SQLGenerator:
    """
    Get the SQL generator instance.
    
    Returns:
        SQLGenerator instance
    """
    return sql_generator