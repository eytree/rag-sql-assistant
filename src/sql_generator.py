#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQL Generator for the RAG-powered SQL Assistant.

This module combines the Knowledge Base and LLM components to generate 
SQL queries from natural language questions, implementing the full
RAG pipeline.
"""

from typing import Dict, Any
import sqlite3
import logging

from .knowledge_base import get_knowledge_base
from .llm import get_llm
from .database import get_database
from .logging_config import register_logger

# Register this module's logger
LOGGER_NAME = 'sql_generator'
register_logger(LOGGER_NAME)
logger = logging.getLogger(LOGGER_NAME)

class SQLGenerator:
    """
    Main SQL Generator class implementing the RAG pipeline.
    """
    
    def __init__(self):
        """Initialize the SQL Generator."""
        logger.info("Initializing SQL Generator")
        self.knowledge_base = get_knowledge_base()
        self.llm = get_llm()
        self.db = get_database()
        
        # Ensure knowledge base is set up
        self.knowledge_base.setup()
        logger.debug("SQL Generator initialized successfully")
    
    def generate_sql(self, query: str, status_callback=None) -> Dict[str, Any]:
        """
        Generate an SQL query from a natural language question using RAG.
        
        Args:
            query: Natural language question
            status_callback: Optional callback function to update status
            
        Returns:
            Dictionary with generated SQL, explanation, and performance info
        """
        logger.info("Generating SQL for query: %s", query)
        
        # Step 1: Retrieve relevant context from knowledge base
        msg = "Step 1: Retrieving context from knowledge base..."
        if status_callback:
            status_callback(msg)
        logger.info(msg)
        context = self.knowledge_base.retrieve_context(query)
        logger.debug("Retrieved context: %s", context)
        
        # Step 2: Format context for prompt
        msg = "Step 2: Formatting context for prompt..."
        if status_callback:
            status_callback(msg)
        logger.info(msg)
        formatted_context = self.knowledge_base.format_context_for_prompt(context)
        logger.debug("Formatted context: %s", formatted_context)
        
        # Step 3: Generate SQL using LLM with RAG context
        msg = "Step 3: Generating SQL with language model..."
        if status_callback:
            status_callback(msg)
        logger.info(msg)
        result = self.llm.generate_sql(query, formatted_context)
        logger.debug("LLM result: %s", result)
        
        # Step 4: Analyze query performance
        sql_query = result["sql"]
        performance_info = {}
        
        if sql_query:
            try:
                msg = "Step 4: Analyzing query performance..."
                if status_callback:
                    status_callback(msg)
                logger.info(msg)
                performance_info = self.analyze_query(sql_query)
                logger.debug("Performance analysis: %s", performance_info)
            except Exception as e:
                logger.error("Error analyzing query: %s", str(e))
                performance_info = {"error": str(e)}
        
        # Combine results
        result.update({
            "original_query": query,
            "context_summary": self._summarize_context(context),
            "performance_analysis": performance_info
        })
        
        msg = "SQL generation completed"
        if status_callback:
            status_callback(msg)
        logger.info(msg)
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
        logger.info("Executing query: %s", sql_query)
        try:
            # Execute query
            results = self.db.execute_query(sql_query)
            logger.info("Query executed successfully. Retrieved %d rows", len(results))
            
            return {
                "success": True,
                "row_count": len(results),
                "results": results[:limit],
                "sql": sql_query
            }
        except sqlite3.Error as e:
            logger.error("Error executing query: %s", str(e))
            return {
                "success": False,
                "error": str(e),
                "sql": sql_query
            }

def get_sql_generator() -> SQLGenerator:
    """
    Get the SQL generator instance.
    Creates the instance on first call.
    
    Returns:
        SQLGenerator instance
    """
    if not hasattr(get_sql_generator, 'sql_generator'):
        get_sql_generator.sql_generator = SQLGenerator()
    return get_sql_generator.sql_generator