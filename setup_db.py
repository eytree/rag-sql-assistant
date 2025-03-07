#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database setup script for the RAG-powered SQL Assistant.

This script creates the SQLite database and tables based on the schema defined in schema.json.
It also populates the database with sample data for demonstration purposes.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
import random
from pathlib import Path

# Create necessary directories
Path("data").mkdir(exist_ok=True)
DB_PATH = "data/trading.db"

def create_tables(conn, schema_data):
    """
    Create database tables based on the schema definition.
    
    Args:
        conn: SQLite connection object
        schema_data: Loaded schema JSON data
    """
    cursor = conn.cursor()
    
    print("Creating tables...")
    for table in schema_data["tables"]:
        # Build the CREATE TABLE statement
        columns = []
        for column in table["columns"]:
            column_def = f"{column['name']} {column['type']}"
            if column["constraints"]:
                column_def += f" {column['constraints']}"
            columns.append(column_def)
        
        create_stmt = f"CREATE TABLE IF NOT EXISTS {table['name']} (\n  "
        create_stmt += ",\n  ".join(columns)
        create_stmt += "\n);"
        
        cursor.execute(create_stmt)
        print(f"  - Created table: {table['name']}")
        
        # Create indexes for the table
        for index in table.get("indexes", []):
            idx_columns = ", ".join(index["columns"])
            unique = "UNIQUE " if index.get("unique", False) else ""
            idx_stmt = f"CREATE {unique}INDEX IF NOT EXISTS {index['name']} ON {table['name']} ({idx_columns});"
            cursor.execute(idx_stmt)
            print(f"    - Created index: {index['name']}")
    
    conn.commit()

def generate_sample_data(conn, schema_data):
    """
    Generate and insert sample data into the database.
    
    Args:
        conn: SQLite connection object
        schema_data: Loaded schema JSON data
    """
    cursor = conn.cursor()
    
    print("\nGenerating sample data...")
    
    # Sample data for users
    users = [
        (1, "john_doe", "john.doe@example.com", "John Doe", "2022-01-15", "retail"),
        (2, "jane_smith", "jane.smith@example.com", "Jane Smith", "2022-02-20", "retail"),
        (3, "bob_johnson", "bob.johnson@example.com", "Bob Johnson", "2022-03-10", "retail"),
        (4, "institutional_user1", "inst1@hedgefund.com", "Hedge Fund One", "2022-01-05", "institutional"),
        (5, "institutional_user2", "inst2@bank.com", "Investment Bank", "2022-02-10", "institutional"),
        (6, "admin_user", "admin@tradingsystem.com", "System Admin", "2022-01-01", "admin")
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO users (user_id, username, email, full_name, created_at, account_type) VALUES (?, ?, ?, ?, ?, ?)",
        users
    )
    print(f"  - Inserted {len(users)} users")
    
    # Sample data for instruments
    instruments = [
        (1, "AAPL", "Apple Inc.", "stock", "NASDAQ", "Technology", "US", True),
        (2, "MSFT", "Microsoft Corporation", "stock", "NASDAQ", "Technology", "US", True),
        (3, "GOOGL", "Alphabet Inc.", "stock", "NASDAQ", "Technology", "US", True),
        (4, "AMZN", "Amazon.com Inc.", "stock", "NASDAQ", "Consumer Cyclical", "US", True),
        (5, "JPM", "JPMorgan Chase & Co.", "stock", "NYSE", "Financial Services", "US", True),
        (6, "JNJ", "Johnson & Johnson", "stock", "NYSE", "Healthcare", "US", True),
        (7, "V", "Visa Inc.", "stock", "NYSE", "Financial Services", "US", True),
        (8, "PG", "Procter & Gamble Co.", "stock", "NYSE", "Consumer Defensive", "US", True),
        (9, "TSLA", "Tesla Inc.", "stock", "NASDAQ", "Consumer Cyclical", "US", True),
        (10, "NVDA", "NVIDIA Corporation", "stock", "NASDAQ", "Technology", "US", True),
        (11, "BTC-USD", "Bitcoin", "crypto", "CRYPTO", "Cryptocurrency", "Global", True),
        (12, "ETH-USD", "Ethereum", "crypto", "CRYPTO", "Cryptocurrency", "Global", True),
        (13, "SPY", "SPDR S&P 500 ETF", "etf", "NYSE", "Index Fund", "US", True),
        (14, "QQQ", "Invesco QQQ Trust", "etf", "NASDAQ", "Index Fund", "US", True),
        (15, "VTI", "Vanguard Total Stock Market ETF", "etf", "NYSE", "Index Fund", "US", True)
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO instruments (instrument_id, symbol, name, instrument_type, exchange, sector, country, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        instruments
    )
    print(f"  - Inserted {len(instruments)} instruments")
    
    # Sample data for portfolios
    portfolios = [
        (1, 1, "Retirement", "Long-term retirement savings", "2022-02-01", "balanced"),
        (2, 1, "Speculative", "High-risk investments", "2022-02-01", "aggressive"),
        (3, 2, "Savings", "Conservative investments", "2022-03-15", "conservative"),
        (4, 2, "Growth", "Growth-focused investments", "2022-03-15", "aggressive"),
        (5, 3, "Main Portfolio", "All-purpose investment portfolio", "2022-04-10", "balanced"),
        (6, 4, "Fund Alpha", "Institutional high-volume strategy", "2022-01-10", "aggressive"),
        (7, 5, "Core Holdings", "Banking core portfolio", "2022-02-15", "conservative")
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO portfolios (portfolio_id, user_id, name, description, created_at, portfolio_type) VALUES (?, ?, ?, ?, ?, ?)",
        portfolios
    )
    print(f"  - Inserted {len(portfolios)} portfolios")
    
    # Generate price history data
    price_history = []
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # One year of data
    
    price_id = 1
    for instrument_id in range(1, 16):  # For all 15 instruments
        # Generate a starting price appropriate for the instrument
        if instrument_id == 11:  # Bitcoin
            base_price = 30000.0
        elif instrument_id == 12:  # Ethereum
            base_price = 2000.0
        elif instrument_id in [1, 2, 3, 9, 10]:  # Tech stocks
            base_price = random.uniform(100.0, 300.0)
        elif instrument_id in [13, 14, 15]:  # ETFs
            base_price = random.uniform(200.0, 400.0)
        else:  # Other stocks
            base_price = random.uniform(50.0, 150.0)
        
        current_date = start_date
        current_price = base_price
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Only generate data for weekdays (not weekends)
                # Generate daily price movement
                daily_change = random.uniform(-0.03, 0.03)  # -3% to +3% daily movement
                current_price = max(0.01, current_price * (1 + daily_change))
                
                # Calculate daily high, low, open
                daily_high = current_price * random.uniform(1.0, 1.02)
                daily_low = current_price * random.uniform(0.98, 1.0)
                daily_open = random.uniform(daily_low, daily_high)
                
                # Generate trading volume
                if instrument_id in [1, 2, 3, 4, 9, 10]:  # High volume for popular stocks
                    volume = int(random.uniform(1000000, 10000000))
                elif instrument_id in [11, 12]:  # Crypto
                    volume = int(random.uniform(10000, 100000))
                else:  # Other instruments
                    volume = int(random.uniform(100000, 1000000))
                
                price_history.append((
                    price_id,
                    instrument_id,
                    current_date.strftime("%Y-%m-%d"),
                    daily_open,
                    current_price,  # close price
                    daily_high,
                    daily_low,
                    volume
                ))
                
                price_id += 1
            
            current_date += timedelta(days=1)
    
    # Insert in batches to avoid SQLite limits
    batch_size = 1000
    for i in range(0, len(price_history), batch_size):
        batch = price_history[i:i+batch_size]
        cursor.executemany(
            "INSERT OR IGNORE INTO price_history (price_id, instrument_id, date, open_price, close_price, high_price, low_price, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            batch
        )
    
    print(f"  - Inserted {len(price_history)} price history records")
    
    # Generate trades
    trades = []
    trade_id = 1
    
    # For each user, generate some trades
    for user_id in range(1, 7):
        # Different numbers of trades for different user types
        num_trades = random.randint(20, 50) if user_id <= 3 else random.randint(50, 100)
        
        for _ in range(num_trades):
            # Select a random instrument
            instrument_id = random.randint(1, 15)
            
            # Select a portfolio if available for this user
            portfolio_candidates = [p[0] for p in portfolios if p[1] == user_id]
            portfolio_id = random.choice(portfolio_candidates) if portfolio_candidates else None
            
            # Generate trade details
            trade_type = random.choice(["buy", "sell"])
            quantity = random.randint(1, 100) * 10
            
            # Get a random date for the trade
            days_ago = random.randint(0, 364)
            trade_date = end_date - timedelta(days=days_ago)
            
            # Find the closing price for that date
            cursor.execute(
                "SELECT close_price FROM price_history WHERE instrument_id = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (instrument_id, trade_date)
            )
            result = cursor.fetchone()
            if result:
                price = result[0]
                total_amount = price * quantity
                
                # Add 1-3 days for settlement
                settlement_date = trade_date + timedelta(days=random.randint(1, 3))
                
                # Commission fee
                commission_fee = round(min(max(total_amount * 0.001, 1.0), 20.0), 2)
                
                trades.append((
                    trade_id,
                    user_id,
                    portfolio_id,
                    instrument_id,
                    trade_type,
                    quantity,
                    price,
                    total_amount,
                    trade_date.strftime("%Y-%m-%d %H:%M:%S"),
                    settlement_date.strftime("%Y-%m-%d"),
                    commission_fee,
                    "completed"
                ))
                
                trade_id += 1
    
    # Insert trades in batches
    for i in range(0, len(trades), batch_size):
        batch = trades[i:i+batch_size]
        cursor.executemany(
            "INSERT OR IGNORE INTO trades (trade_id, user_id, portfolio_id, instrument_id, trade_type, quantity, price, total_amount, trade_date, settlement_date, commission_fee, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch
        )
    
    print(f"  - Inserted {len(trades)} trades")
    
    # Generate portfolio holdings
    holdings = []
    holding_id = 1
    
    # Calculate holdings based on completed trades
    for portfolio_id in [p[0] for p in portfolios]:
        # Get all trades for this portfolio
        cursor.execute(
            "SELECT instrument_id, trade_type, quantity, price FROM trades WHERE portfolio_id = ? AND status = 'completed'",
            (portfolio_id,)
        )
        
        portfolio_trades = cursor.fetchall()
        
        # Calculate net holdings
        holdings_dict = {}
        for instr_id, trade_type, qty, price in portfolio_trades:
            if instr_id not in holdings_dict:
                holdings_dict[instr_id] = {"quantity": 0, "total_cost": 0.0}
                
            if trade_type == "buy":
                holdings_dict[instr_id]["total_cost"] += price * qty
                holdings_dict[instr_id]["quantity"] += qty
            else:  # sell
                # For simplicity, we're not handling the case where quantity goes negative
                if holdings_dict[instr_id]["quantity"] >= qty:
                    # Reduce quantity but keep track of average cost
                    old_qty = holdings_dict[instr_id]["quantity"]
                    old_cost = holdings_dict[instr_id]["total_cost"]
                    
                    # Remove this portion from cost basis (simplified method)
                    cost_reduction = (qty / old_qty) * old_cost
                    
                    holdings_dict[instr_id]["quantity"] -= qty
                    holdings_dict[instr_id]["total_cost"] -= cost_reduction
        
        # Create holdings records for instruments with positive quantities
        for instr_id, data in holdings_dict.items():
            if data["quantity"] > 0:
                avg_cost = data["total_cost"] / data["quantity"]
                
                holdings.append((
                    holding_id,
                    portfolio_id,
                    instr_id,
                    data["quantity"],
                    avg_cost,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                
                holding_id += 1
    
    cursor.executemany(
        "INSERT OR IGNORE INTO portfolio_holdings (holding_id, portfolio_id, instrument_id, quantity, average_cost, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
        holdings
    )
    
    print(f"  - Inserted {len(holdings)} portfolio holdings")
    
    conn.commit()

def main():
    """Main function to set up the database"""
    # Check if database already exists
    if os.path.exists(DB_PATH):
        user_input = input(f"Database already exists at {DB_PATH}. Overwrite? (y/n): ")
        if user_input.lower() != 'y':
            print("Database setup cancelled.")
            return
        os.remove(DB_PATH)
    
    # Load schema data
    with open("data/schema.json", "r") as f:
        schema_data = json.load(f)
    
    # Create database connection
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Create tables
        create_tables(conn, schema_data)
        
        # Generate and insert sample data
        generate_sample_data(conn, schema_data)
        
        print("\nDatabase setup completed successfully!")
        print(f"Database created at: {os.path.abspath(DB_PATH)}")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
    