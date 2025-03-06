# Example Queries for the RAG-powered SQL Assistant

This document contains example natural language queries to demonstrate the capabilities of the RAG-powered SQL Assistant. You can use these as a starting point to explore the system's capabilities.

## Basic Queries

1. "Show me all users in the system"
   - This simple query will list all users from the users table.

2. "List all active instruments on the NASDAQ exchange"
   - Retrieves instruments filtered by exchange and active status.

3. "What are the technology sector stocks in our database?"
   - Demonstrates filtering by sector.

4. "Show me all portfolios belonging to user 'john_doe'"
   - Demonstrates simple join operations.

## Intermediate Queries

5. "What were the top 5 most expensive trades in the past six months?"
   - Demonstrates sorting, limiting, and date filtering.

6. "Calculate the total value of all trades for each user"
   - Demonstrates aggregation with GROUP BY.

7. "Show me all trades of Apple stock, including the user's name who made the trade"
   - Demonstrates multi-table joins and filtering.

8. "What's the average closing price of Bitcoin in the last 30 days?"
   - Demonstrates date calculations and aggregation.

## Advanced Queries

9. "Show me the portfolio performance of user 'john_doe' over the last quarter, comparing starting and ending values"
   - Demonstrates complex joins, date calculations, and analytics.

10. "Find all users who have made more than 10 trades in the technology sector in the last month"
    - Demonstrates subqueries, joins across multiple tables, and complex conditions.

11. "Calculate the sector allocation for portfolio 1, showing the percentage of portfolio value in each sector"
    - Demonstrates window functions, complex calculations, and multiple joins.

12. "Which instruments have shown the highest price volatility in the past 3 months?"
    - Demonstrates statistical calculations and complex time-series analysis.

## Performance-Challenging Queries

13. "Show me all users who have traded both Apple and Microsoft stock"
    - Tests handling of intersections and set operations.

14. "List all pairs of users who have traded the same instruments"
    - Tests handling of self-joins and complex relationship detection.

15. "Calculate the correlation between AAPL and MSFT stock prices over the past year"
    - Tests handling of complex time-series analysis and statistical calculations.

16. "Show me a complete trade history for all instruments in portfolio 2, including daily price changes after each trade"
    - Tests handling of large result sets and complex temporal relationships.

## Potential Issues and RAG Benefits

When running these queries, pay attention to:

1. **Schema Understanding** - How well does the system understand the relationships between tables?
2. **Performance Insights** - Does the system provide helpful performance warnings and optimization suggestions?
3. **Query Accuracy** - Are the generated queries correct for the requested information?
4. **RAG Context** - How does the system leverage the retrieved context to improve query generation?

The RAG approach should help the system:

- Correctly identify relevant tables and joins
- Include appropriate indexes in the query plan
- Provide performance insights based on the schema information
- Suggest optimizations for potentially slow queries
- Handle ambiguities in the natural language questions