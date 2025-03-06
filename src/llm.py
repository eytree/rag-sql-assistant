#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local LLM integration for the RAG-powered SQL Assistant.

This module provides functionality to interact with a locally-run LLM
using llama.cpp. It handles prompt construction, response parsing, and
model management.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import re
from dotenv import load_dotenv

# Import llama_cpp conditionally to avoid errors if not installed
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Load environment variables
load_dotenv()

# Default model path (can be overridden with environment variable)
DEFAULT_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "./models/llama-3-8b-instruct.gguf")

class LocalLLM:
    """
    Interface for interacting with a locally-run LLM using llama.cpp.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the local LLM interface.
        
        Args:
            model_path: Path to the local model file (.gguf format)
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        
        # Check if model exists
        if not Path(self.model_path).exists():
            print(f"Warning: Model file not found at {self.model_path}")
            print("You can download a compatible model from huggingface.co or specify a different path.")
            print("Using model path from environment variable: LLM_MODEL_PATH")
    
    def load_model(self, n_ctx: int = 8192, n_gpu_layers: int = -1):
        """
        Load the LLM model if not already loaded.
        
        Args:
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
        if not LLAMA_AVAILABLE:
            raise ImportError(
                "llama_cpp package is not installed. "
                "Install it with 'pip install llama-cpp-python'"
            )
        
        if self.model is None:
            print(f"Loading LLM from {self.model_path}")
            try:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 2048, 
        temperature: float = 0.1,
        top_p: float = 0.95,
        stop: List[str] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (lower = more deterministic)
            top_p: Top-p sampling parameter
            stop: List of strings to stop generation at
            
        Returns:
            Generated text response
        """
        if self.model is None:
            self.load_model()
        
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or []
        )
        
        return response["choices"][0]["text"]
    
    def create_sql_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for SQL generation with RAG context.
        
        Args:
            query: Natural language query
            context: Retrieved context for RAG
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are a helpful SQL assistant for a trading database. Your task is to convert natural language questions into valid SQLite SQL queries.

Below is information about the database schema:

{context}

Based on this schema, please write an SQL query that answers the following question:

User question: {query}

Please respond with:
1. A valid SQL query that answers the question.
2. A brief explanation of how the query works.
3. Any potential performance issues that might arise with the query.
4. Suggestions for optimizing the query if needed.

SQL Query:
"""
        return prompt
    
    def extract_sql_from_response(self, response: str) -> Tuple[str, str, str]:
        """
        Extract SQL query and explanations from the model's response.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Tuple of (sql_query, explanation, performance_notes)
        """
        # Default values
        sql_query = ""
        explanation = ""
        performance_notes = ""
        
        # Extract SQL query - look for code blocks or SQL: sections
        sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # Try alternative patterns
            sql_match = re.search(r"SQL Query:\n(.*?)(?:\n\n|$)", response, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
        
        # Extract explanation
        explanation_match = re.search(r"Explanation:(.*?)(?:\n\n|Performance|$)", response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        # Extract performance notes
        performance_match = re.search(r"Performance (?:Issues|Notes|Considerations):(.*?)(?:\n\n|Suggestions|$)", response, re.DOTALL)
        if performance_match:
            performance_notes = performance_match.group(1).strip()
        
        # If no performance notes found, try to find optimization suggestions
        if not performance_notes:
            optimization_match = re.search(r"Optimization (?:Suggestions|Notes|Tips):(.*?)(?:\n\n|$)", response, re.DOTALL)
            if optimization_match:
                performance_notes = optimization_match.group(1).strip()
        
        return sql_query, explanation, performance_notes
    
    def generate_sql(self, query: str, context: str) -> Dict[str, str]:
        """
        Generate an SQL query from a natural language query using RAG.
        
        Args:
            query: Natural language query
            context: Retrieved context for RAG
            
        Returns:
            Dictionary with generated SQL and explanations
        """
        prompt = self.create_sql_prompt(query, context)
        
        # Generate response
        response = self.generate(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["User question:"]
        )
        
        # Extract SQL and explanations
        sql_query, explanation, performance_notes = self.extract_sql_from_response(response)
        
        return {
            "sql": sql_query,
            "explanation": explanation,
            "performance_notes": performance_notes,
            "full_response": response
        }

# Singleton instance for easy import
llm = LocalLLM()

def get_llm() -> LocalLLM:
    """
    Get the LLM instance.
    
    Returns:
        LocalLLM instance
    """
    return llm