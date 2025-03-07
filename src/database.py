#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database connection and operations for the RAG-powered SQL Assistant.

This module provides functionality to interact with the SQLite database,
including connection management, query execution, and schema information retrieval.
"""

import json
import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from .logging_config import register_logger

# Register this module's logger
LOGGER_NAME = 'database'
register_logger(LOGGER_NAME)
logger = logging.getLogger(LOGGER_NAME)

# Database path
DB_PATH = Path("data/trading.db")

class Database:
    """Database connection and operations class"""
    
    def __init__(self, db_path: Path = DB_PATH):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        logger.info(f"Initializing Database with path: {db_path}")
        
        # Load schema information once
        schema_path = Path("data/schema.json")
        if schema_path.exists():
            logger.info("Loading schema from %s", schema_path)
            with open(schema_path, "r") as f:
                self.schema = json.load(f)
            logger.debug("Schema loaded successfully")
        else:
            logger.warning("Schema file not found at %s", schema_path)
            self.schema = None
    
    def connect(self) -> None:
        """Establish database connection"""
        if not self.conn:
            if not self.db_path.exists():
                logger.error(f"Database file not found: {self.db_path}")
                raise FileNotFoundError(f"Database file not found: {self.db_path}")
            
            logger.info("Establishing new database connection")
            self.conn = sqlite3.connect(self.db_path)
            # Configure connection to return rows as dictionaries
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.debug("Database connection established successfully")
    
    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            logger.info("Closing database connection")
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.debug("Database connection closed")
    
    def execute_query(self, query: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query to execute
            params: Parameters for the query (for parameterized queries)
            
        Returns:
            List of dictionaries representing the query results
        """
        try:
            logger.info("Executing query: %s", query)
            if params:
                logger.debug("Query parameters: %s", params)
            
            self.connect()
            
            # Split the query into statements and take the last one
            statements = [s.strip() for s in query.split(';') if s.strip()]
            if not statements:
                logger.warning("No valid SQL statements found in query")
                return []
            
            # Use the last statement (ignoring comments)
            actual_query = statements[-1]
            logger.debug("Using query: %s", actual_query)
            
            if params:
                self.cursor.execute(actual_query, params)
            else:
                self.cursor.execute(actual_query)
            
            # Convert results to dictionaries
            columns = [col[0] for col in self.cursor.description] if self.cursor.description else []
            logger.debug("Query columns: %s", columns)
            
            results = []
            for row in self.cursor.fetchall():
                results.append({columns[i]: row[i] for i in range(len(columns))})
            
            logger.info("Query executed successfully. Retrieved %d rows", len(results))
            return results
            
        except sqlite3.Error as e:
            logger.error("Database error: %s", str(e))
            logger.error("Failed query: %s", query)
            if params:
                logger.error("Failed query parameters: %s", params)
            return []
    
    def execute_explain(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute EXPLAIN QUERY PLAN for a SQL query to analyze performance.
        
        Args:
            query: SQL query to explain
            
        Returns:
            List of dictionaries representing the explanation plan
        """
        explain_query = f"EXPLAIN QUERY PLAN {query}"
        return self.execute_query(explain_query)
    
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """
        Analyze query performance and provide insights.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            Dictionary with performance analysis results
        """
        explain_results = self.execute_explain(query)
        
        analysis = {
            "plan": explain_results,
            "warnings": [],
            "suggestions": []
        }
        
        # Analyze the explain plan
        for step in explain_results:
            detail = step.get("detail", "")
            
            # Check for full table scans
            if "SCAN TABLE" in detail and "USING INDEX" not in detail:
                table_name = detail.split("SCAN TABLE")[1].split()[0]
                analysis["warnings"].append(f"Full table scan on {table_name}")
                
                # Suggest indexes based on schema information
                if self.schema:
                    for table in self.schema["tables"]:
                        if table["name"] == table_name:
                            for relationship in self.schema.get("relationships", []):
                                if relationship["from_table"] == table_name:
                                    analysis["suggestions"].append(
                                        f"Consider adding an index on {table_name}.{relationship['from_column']}"
                                    )
            
            # Check for nested loops
            if "USING TEMP" in detail:
                analysis["warnings"].append("Query uses temporary tables which can be slow")
                analysis["suggestions"].append("Consider rewriting the query to avoid temporary tables")
        
        # Add general performance notes from schema if available
        if self.schema and "performance_notes" in self.schema:
            relevant_notes = []
            query_lower = query.lower()
            
            for note in self.schema["performance_notes"]:
                # Add performance notes that are relevant to the tables in the query
                for table in self.schema["tables"]:
                    if table["name"].lower() in query_lower and table["name"].lower() in note.lower():
                        relevant_notes.append(note)
                        break
            
            analysis["performance_notes"] = relevant_notes
        
        return analysis
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema information for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table schema information or None if table not found
        """
        if not self.schema:
            return None
        
        for table in self.schema["tables"]:
            if table["name"].lower() == table_name.lower():
                return table
        
        return None
    
    def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of relationships involving the specified table
        """
        if not self.schema or "relationships" not in self.schema:
            return []
        
        relationships = []
        for rel in self.schema["relationships"]:
            if rel["from_table"].lower() == table_name.lower() or rel["to_table"].lower() == table_name.lower():
                relationships.append(rel)
        
        return relationships
    
    def get_common_queries(self) -> List[Dict[str, Any]]:
        """
        Get common predefined queries from the schema.
        
        Returns:
            List of common queries with descriptions
        """
        if not self.schema or "common_queries" not in self.schema:
            return []
        
        return self.schema["common_queries"]
    
    def get_all_tables(self) -> List[Dict[str, Any]]:
        """
        Get information about all tables in the database.
        
        Returns:
            List of dictionaries with table information
        """
        if not self.schema:
            return []
        
        return self.schema["tables"]
    
    def validate_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        self.connect()
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return bool(self.cursor.fetchone())
    
    def validate_column_exists(self, table_name: str, column_name: str) -> bool:
        """
        Check if a column exists in a table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            
        Returns:
            True if the column exists, False otherwise
        """
        table_info = self.get_table_schema(table_name)
        if not table_info:
            return False
        
        for column in table_info["columns"]:
            if column["name"].lower() == column_name.lower():
                return True
        
        return False

def get_database() -> Database:
    """
    Get the database instance.
    Creates the instance on first call.
    
    Returns:
        Database instance
    """
    if not hasattr(get_database, 'db'):
        get_database.db = Database()
    return get_database.db