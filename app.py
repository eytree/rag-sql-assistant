#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point for the RAG-powered SQL Assistant.

This script provides a command-line interface for interacting with the system.
Users can ask natural language questions about the trading database, and the 
system will generate SQL queries with explanations and performance insights.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional
import sqlite3
import json

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint

# Ensure the src directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sql_generator import get_sql_generator
from src.database import get_database
from src.logging_config import LogLevel, setup_logging

# Create Typer app
app = typer.Typer(help="RAG-powered SQL Assistant")
console = Console()

def check_database():
    """Check if the database exists and has been set up."""
    db_path = Path("data/trading.db")
    if not db_path.exists():
        console.print("[bold red]Database not found![/bold red]")
        console.print("Please run the setup script first:")
        console.print("[bold]python setup_db.py[/bold]")
        sys.exit(1)

def display_sql_result(result):
    """Display the generated SQL query and explanation."""
    # Display original query
    console.print(Panel(f"[bold blue]{result['original_query']}[/bold blue]", title="Your Question"))
    
    # Display context summary
    ctx = result["context_summary"]
    console.print(Panel(
        f"Retrieved context from [bold]{len(ctx['tables'])}[/bold] tables, "
        f"[bold]{ctx['relationships']}[/bold] relationships, and "
        f"[bold]{ctx['similar_queries']}[/bold] similar queries.",
        title="Context Summary"
    ))
    
    # Display generated SQL
    if result["sql"]:
        sql_syntax = Syntax(result["sql"], "sql", theme="monokai", line_numbers=True)
        console.print(Panel(sql_syntax, title="Generated SQL Query"))
        
        # Display explanation
        if result["explanation"]:
            console.print(Panel(Markdown(result["explanation"]), title="Explanation"))
    else:
        console.print("[bold red]No SQL query could be generated.[/bold red]")
    
    # Display performance analysis
    if "performance_analysis" in result and result["performance_analysis"]:
        perf = result["performance_analysis"]
        if "warnings" in perf and perf["warnings"]:
            console.print("[bold yellow]Performance Warnings:[/bold yellow]")
            for warning in perf["warnings"]:
                console.print(f"• {warning}")
        
        if "suggestions" in perf and perf["suggestions"]:
            console.print("[bold green]Optimization Suggestions:[/bold green]")
            for suggestion in perf["suggestions"]:
                console.print(f"• {suggestion}")
    
    # Display model's performance notes
    if result["performance_notes"]:
        console.print(Panel(Markdown(result["performance_notes"]), title="Performance Notes"))

def display_query_results(data):
    """Display the results of executing the SQL query."""
    if not data["success"]:
        console.print(f"[bold red]Error executing query:[/bold red] {data['error']}")
        return
    
    if not data["results"]:
        console.print("[yellow]Query executed successfully, but returned no results.[/yellow]")
        return
    
    # Create table
    table = Table(title=f"Query Results ({data['row_count']} rows)")
    
    # Add columns
    for key in data["results"][0].keys():
        table.add_column(key, style="cyan")
    
    # Add rows
    for row in data["results"]:
        table.add_row(*[str(val) for val in row.values()])
    
    console.print(table)

@app.callback()
def main(log_level: LogLevel = typer.Option(
    LogLevel.NONE,
    "--log-level", "-l",
    help="Set logging level (none, info, debug)"
)):
    """Initialize the application with optional logging level."""
    setup_logging(log_level)

@app.command()
def generate(query: Optional[str] = typer.Argument(None, help="Natural language query")):
    """Generate an SQL query from natural language."""
    check_database()
    
    # Get query from argument or prompt
    if not query:
        query = Prompt.ask("[bold]Enter your question about the trading database[/bold]")
    
    with console.status("[bold green]Processing...[/bold green]"):
        # Initialize SQL generator
        sql_generator = get_sql_generator()
        
        # Generate SQL
        result = sql_generator.generate_sql(query)
    
    # Display result
    display_sql_result(result)
    
    # Ask if user wants to execute the query
    if result["sql"]:
        execute = Prompt.ask(
            "\n[bold]Would you like to execute this query?[/bold]",
            choices=["y", "n"],
            default="y"
        )
        
        if execute.lower() == "y":
            with console.status("[bold green]Executing query...[/bold green]"):
                execution_result = sql_generator.execute_query(result["sql"])
            
            display_query_results(execution_result)

@app.command()
def interactive():
    """Start an interactive session with the SQL Assistant."""
    check_database()
    
    console.print("[bold green]RAG-powered SQL Assistant[/bold green]")
    console.print("Ask questions in natural language about the trading database.")
    console.print("Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to end the session.\n")
    
    # Initialize SQL generator
    sql_generator = get_sql_generator()
    
    while True:
        query = Prompt.ask("[bold]Enter your question[/bold]")
        
        if query.lower() in ("exit", "quit"):
            break
        
        with console.status("[bold green]Processing...[/bold green]"):
            # Generate SQL
            result = sql_generator.generate_sql(query)
        
        # Display result
        display_sql_result(result)
        
        # Ask if user wants to execute the query
        if result["sql"]:
            execute = Prompt.ask(
                "\n[bold]Would you like to execute this query?[/bold]",
                choices=["y", "n"],
                default="y"
            )
            
            if execute.lower() == "y":
                with console.status("[bold green]Executing query...[/bold green]"):
                    execution_result = sql_generator.execute_query(result["sql"])
                
                display_query_results(execution_result)
        
        console.print("\n" + "-" * 80 + "\n")

@app.command()
def info():
    """Display information about the database schema."""
    check_database()
    
    db = get_database()
    
    console.print("[bold green]Trading Database Schema Information[/bold green]\n")
    
    tables = db.get_all_tables()
    
    # Display tables
    table = Table(title="Database Tables")
    table.add_column("Table Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Columns", style="magenta")
    
    for table_info in tables:
        table.add_row(
            table_info["name"],
            table_info.get("description", ""),
            str(len(table_info["columns"]))
        )
    
    console.print(table)
    
    # Ask if user wants to see detailed info for a specific table
    show_details = Prompt.ask(
        "\n[bold]Would you like to see detailed information for a specific table?[/bold]",
        choices=["y", "n"],
        default="y"
    )
    
    if show_details.lower() == "y":
        table_name = Prompt.ask(
            "[bold]Enter table name[/bold]",
            choices=[t["name"] for t in tables]
        )
        
        # Get table info
        table_info = db.get_table_schema(table_name)
        
        if table_info:
            console.print(f"\n[bold]Table:[/bold] {table_info['name']}")
            console.print(f"[bold]Description:[/bold] {table_info.get('description', '')}\n")
            
            # Display columns
            columns_table = Table(title=f"Columns in {table_info['name']}")
            columns_table.add_column("Column Name", style="cyan")
            columns_table.add_column("Type", style="yellow")
            columns_table.add_column("Constraints", style="green")
            columns_table.add_column("Description", style="magenta")
            
            for column in table_info["columns"]:
                columns_table.add_row(
                    column["name"],
                    column["type"],
                    column.get("constraints", ""),
                    column.get("description", "")
                )
            
            console.print(columns_table)
            
            # Display indexes
            if "indexes" in table_info and table_info["indexes"]:
                indexes_table = Table(title=f"Indexes in {table_info['name']}")
                indexes_table.add_column("Index Name", style="cyan")
                indexes_table.add_column("Columns", style="yellow")
                indexes_table.add_column("Unique", style="green")
                
                for idx in table_info["indexes"]:
                    indexes_table.add_row(
                        idx["name"],
                        ", ".join(idx["columns"]),
                        "Yes" if idx.get("unique", False) else "No"
                    )
                
                console.print(indexes_table)
            
            # Display relationships
            relationships = db.get_table_relationships(table_name)
            
            if relationships:
                console.print(f"\n[bold]Relationships for {table_name}:[/bold]")
                
                for rel in relationships:
                    if rel["from_table"] == table_name:
                        console.print(f"→ {rel['relationship']}: {rel['from_table']}.{rel['from_column']} to {rel['to_table']}.{rel['to_column']}")
                    else:
                        console.print(f"← {rel['relationship']}: {rel['to_table']}.{rel['to_column']} from {rel['from_table']}.{rel['from_column']}")

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[bold]Exiting...[/bold]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)