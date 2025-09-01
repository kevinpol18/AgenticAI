from contextlib import contextmanager
import json
import uuid
import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from datetime import datetime, timedelta
from sqlalchemy.sql import text
from typing import Dict, List, Optional, Union, Tuple, Set, Any
from sqlalchemy import create_engine, Engine
from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from enum import Enum
import functools
from difflib import SequenceMatcher
import re


# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# fmt: off
# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",               "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]
# fmt: on

# Given below are some utility functions you can use to implement your multi-agent system


def generate_sample_inventory(
    paper_supplies: list, coverage: float = 0.4, seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)), size=num_items, replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(200, 800),  # Realistic stock range
                "min_stock_level": np.random.randint(
                    50, 150
                ),  # Reasonable threshold for reordering
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.
    """
    try:
        # Define a consistent starting date BEFORE any use
        initial_date = datetime(2025, 1, 1).isoformat()

        # 1) Create 'transactions' table with an explicit schema
        with db_engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS transactions"))
            conn.execute(
                text(
                    """
                    CREATE TABLE transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_name TEXT,
                        transaction_type TEXT CHECK (transaction_type IN ('stock_orders','sales')),
                        units INTEGER,
                        price REAL,
                        transaction_date TEXT
                    )
                    """
                )
            )

        # 2) Load and initialize 'quote_requests' table
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # 3) Load and transform 'quotes' table
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date  # uses the defined initial_date

        # Unpack metadata fields if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        quotes_df = quotes_df[
            ["request_id", "total_amount", "quote_explanation", "order_date", "job_type", "order_size", "event_type"]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # 4) Generate inventory and seed stock
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed, coverage=0.8)

        # Seed initial transactions: use None (not null)
        initial_transactions = []
        initial_transactions.append(
            {
                "item_name": None,             # starting cash row
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": int(item["current_stock"]),
                    "price": float(item["current_stock"] * item["unit_price"]),
                    "transaction_date": initial_date,
                }
            )

        # Insert initial transactions with a single connection
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO transactions (item_name, transaction_type, units, price, transaction_date)
                    VALUES (:item_name, :transaction_type, :units, :price, :transaction_date)
                    """
                ),
                initial_transactions,
            )

        # Save inventory table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        log(f"Error initializing database: {e}", LogLevel.ERROR)
        raise


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        log(
            f"[INFO] Creating transaction: {item_name}, {transaction_type}, {quantity}, {price}, {date_str}",
            LogLevel.INFO,
        )

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                }
            ]
        )

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        log(f"Error creating transaction: {e}", LogLevel.ERROR)
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    log(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'",
        LogLevel.DEBUG,
    )

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        log(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.",
            LogLevel.WARNING,
        )
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        log(f"Error getting cash balance: {e}", LogLevel.ERROR)
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        # this has been modified from the original code since
        # creating a dict directly from a row fails due to
        # metadata from SQLAlchemy result objects.
        return [row._asdict() for row in result]


########################
########################
########################
# Progress Manager for CLI Output & print utils
########################
########################
########################


class LogLevel(Enum):
    OFF = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


def log(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """
    Log a message to the console with the specified log level.

    Args:
        message (str): The message to log.
        level (LogLevel, optional): The log level for the message. Defaults to LogLevel.INFO.
    """
    if level.value <= log_level.value:
        print(f"[{level.name}] {message}")


class StatusContext:
    """Helper class for manual status control"""

    def __init__(self, task_name: str, start_time: float):
        self.task_name = task_name
        self.start_time = start_time
        self.is_failed = False

    def fail(self, reason: str = "Operation failed"):
        """Manually mark the task as failed"""
        self.is_failed = True
        duration = time.time() - self.start_time
        duration_str = (
            f"{duration:.1f}s" if duration >= 1 else f"{int(duration * 1000)}ms"
        )
        print(f"❌ Failed: {self.task_name} ({duration_str}) - {reason}")

    def complete(self, message: str = None):
        """Manually mark the task as completed"""
        duration = time.time() - self.start_time
        duration_str = (
            f"{duration:.1f}s" if duration >= 1 else f"{int(duration * 1000)}ms"
        )
        result_msg = f" - {message}" if message else ""
        print(f"✅ Completed: {self.task_name} ({duration_str}){result_msg}")


@contextmanager
def status(task_name: str):
    """
    Simple context manager that prints running/completed/failed status

    Usage:
        # Auto completion (existing behavior)
        with status("OrderAgent - process_order"):
            your_function()

        # Manual control
        with status("OrderAgent - process_order") as s:
            result = your_function()
            if not result.success:
                s.fail("Order extraction failed")
                return
            s.complete("Found 3 items")
    """
    print(f"🔄 Running: {task_name}")
    start_time = time.time()
    status_obj = StatusContext(task_name, start_time)

    try:
        yield status_obj

        # Only auto-complete if user didn't manually call complete() or fail()
        # We check this by seeing if the duration would be very recent
        current_time = time.time()
        if current_time - start_time > 0.001:  # More than 1ms has passed
            duration = current_time - start_time
            if duration < 1:
                duration_str = f"{int(duration * 1000)}ms"
            else:
                duration_str = f"{duration:.1f}s"
            if not status_obj.is_failed:
                # If it wasn't marked as failed, we assume it completed successfully
                print(f"✅ Completed: {task_name} ({duration_str})")

    except Exception as e:
        duration = time.time() - start_time
        duration_str = (
            f"{duration:.1f}s" if duration >= 1 else f"{int(duration * 1000)}ms"
        )
        print(f"❌ Failed: {task_name} ({duration_str}) - {str(e)}")
        raise  # Re-raise the exception


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
#
# Entity Definitions
#


class InventoryItem(BaseModel):
    """
    Canonical representation of a SKU in the inventory catalog.

    Purpose
    - Holds static attributes for an item, independent of the current stock level snapshot.
    - Used by matching, quoting, and inventory checks to determine pricing and minimum buffer rules.

    Important notes
    - unit_price is expressed per base unit:
      - Paper types are priced per sheet.
      - Product types are priced per physical unit (e.g., per plate, per cup, per roll).
    - min_stock_level is a replenishment threshold; it does not reflect current on-hand stock.
    """

    item_name: str = Field(
        ...,
        description="Canonical display name of the SKU as stored in the inventory table (e.g., 'A4 paper')."
    )
    category: str = Field(
        ...,
        description="Functional category for the SKU (e.g., 'paper', 'product', 'specialty', 'large_format')."
    )
    unit_price: float = Field(
        ...,
        description="Price per base unit for this SKU. Paper is per sheet; products are per unit."
    )
    min_stock_level: int = Field(
        ...,
        description="Target minimum on-hand quantity. If fulfilling an order would drop below this, create a restock."
    )


class OrderItem(BaseModel):
    """
    A single line item requested by the customer, normalized to base units.

    Purpose
    - Represents the exact quantity and per-unit price to be used for inventory checks and quoting.
    - Quantities must be expressed in base units (e.g., sheets or product units); conversions (reams/packs/boxes) are handled before this stage.
    """

    item_name: str = Field(
        ...,
        description="Canonical inventory item name selected by the matcher."
    )
    quantity: int = Field(
        ...,
        description="Requested quantity in base units (e.g., 'sheets' for paper, 'units' for products)."
    )
    unit_price: float = Field(
        ...,
        description="Per-unit price used for quoting this item (typically from inventory)."
    )


class Order(BaseModel):
    """
    A complete, normalized customer order.

    Purpose
    - Carries a unique ID, the request and expected delivery dates (ISO YYYY-MM-DD), and normalized items.
    - Serves as the single source of truth for inventory validation and quoting.
    """

    id: str = Field(
        ...,
        description="Unique identifier for the order, typically a UUID hex string."
    )
    request_date: str = Field(
        ...,
        description="ISO date string YYYY-MM-DD when the customer placed the request."
    )
    expected_delivery_date: str = Field(
        ...,
        description="ISO date string YYYY-MM-DD when the customer expects delivery."
    )
    items: List[OrderItem] = Field(
        ...,
        description="Flattened list of items with quantities in base units."
    )


class OrderResult(BaseModel):
    """
    Structured output from the Order (or Extraction) Agent.

    Purpose
    - Communicates whether the system successfully extracted a valid Order from the input.
    - If unsuccessful, agent_error provides a human-readable reason for failure.
    """

    is_success: bool = Field(
        ...,
        description="True if the order was successfully extracted; False otherwise."
    )
    order: Optional[Order] = Field(
        None,
        description="The normalized Order if extraction succeeded; otherwise None."
    )
    agent_error: Optional[str] = Field(
        None,
        description="Human-readable error message describing why extraction failed."
    )


class StockOrderItem(BaseModel):
    """
    A single line item for replenishment (purchase from supplier).

    Purpose
    - Created when on-hand stock is insufficient to fulfill an Order or would fall below minimum stock thresholds.
    - delivery_date reflects the supplier delivery ETA (not the customer’s expected date).
    """

    item_name: str = Field(
        ...,
        description="Canonical inventory item name that needs to be reordered."
    )
    quantity: int = Field(
        ...,
        description="Quantity to purchase from supplier, in base units."
    )
    unit_price: float = Field(
        ...,
        description="Per-unit purchase price at which the stock order will be recorded."
    )
    delivery_date: datetime = Field(
        ...,
        description="Estimated date when the supplier will deliver the replenishment."
    )


class StockOrder(BaseModel):
    """
    A collection of replenishment items required to maintain service levels.

    Purpose
    - Aggregates StockOrderItem lines created during inventory checks.
    - May be empty if no restocking is needed.
    """

    items: List[StockOrderItem] = Field(
        ...,
        description="List of replenishment lines; can be an empty list when no restock is required."
    )


class InventoryResult(BaseModel):
    """
    Outcome of the inventory validation step.

    Purpose
    - Reports whether the order is feasible within stock, restock lead times, and cash constraints.
    - If feasible, returns a StockOrder (possibly empty).
    - If infeasible, agent_error conveys a user-friendly failure reason.
    """

    is_success: bool = Field(
        ...,
        description="True if inventory checks passed; False if constraints were violated."
    )
    stock_order: Optional[StockOrder] = Field(
        None,
        description="Replenishment plan if restocking is necessary; None when not applicable."
    )
    agent_error: Optional[str] = Field(
        None,
        description="User-friendly reason why inventory validation failed."
    )


class DiscountPolicyType(str, Enum):
    """
    Types of discount policies that can be applied when generating quotes.

    Notes
    - NO_DISCOUNT: No reductions applied.
    - PERCENTAGE: Apply a percentage discount to item-level prices.
    - ROUND_DOWN: Round down to a 'friendly' final price (never exceed 10% total reduction).
    """

    NO_DISCOUNT = "No discount policy applied."
    PERCENTAGE = "Discount the amount by a percentage value."
    ROUND_DOWN = "Round down the amount to a specific precision. But never round down more than 10%."


class DiscountPolicy(BaseModel):
    """
    A selected discount policy for a given quote.

    Purpose
    - Captures the discount type and a human-readable description of how it was applied.
    """

    policy: DiscountPolicyType = Field(
        ...,
        description="Chosen discount type (e.g., NO_DISCOUNT, PERCENTAGE, ROUND_DOWN)."
    )
    policy_description: str = Field(
        ...,
        description="Human-readable description of how the discount was applied."
    )


class QuoteItem(BaseModel):
    """
    Per-item pricing line in the generated quote.

    Purpose
    - Records the canonical item, the final quoted quantity, and the discounted per-unit price.
    - The sum over all items must equal the discounted_total_amount in the Quote object.
    """

    item_name: str = Field(
        ...,
        description="Canonical inventory item name as shown on the quote."
    )
    quantity: int = Field(
        ...,
        description="Quoted quantity in base units."
    )
    discounted_price: float = Field(
        ...,
        description="Per-unit price after applying applicable discounts."
    )


class Quote(BaseModel):
    """
    A customer-facing price proposal.

    Purpose
    - Bundles the order context, itemized quoted prices, aggregate totals, and the text used to communicate the offer.
    - Implements the convention that discounted_total_amount equals the sum(item.quantity * item.discounted_price).
    """

    order: Order = Field(
        ...,
        description="The normalized Order being priced."
    )
    quote_items: List[QuoteItem] = Field(
        ...,
        description="Itemized pricing lines with discounted per-unit prices."
    )
    customer_quote: str = Field(
        ...,
        description="Natural-language quote text to present to the customer."
    )
    total_amount: float = Field(
        ...,
        description="Undiscounted total of the order (sum of base unit prices × quantities)."
    )
    discounted_total_amount: float = Field(
        ...,
        description="Final total after discounts, typically rounded to a friendly whole-dollar amount."
    )
    discount_amount: float = Field(
        ...,
        description="Absolute amount discounted from the original total."
    )
    discount_policy: Optional[DiscountPolicy] = Field(
        None,
        description="Applied discount policy; None if no discount was applied."
    )


class Transaction(BaseModel):
    """
    A financial movement recorded in the ledger.

    Purpose
    - Records stock purchases and customer sales.
    - Supports financial reporting such as cash balance and top sellers.

    Notes
    - transaction_type must be either 'stock_orders' (purchase) or 'sales' (revenue).
    - price records the total transaction amount (not a per-unit price).
    - date should be stored as an ISO string (YYYY-MM-DD or full ISO timestamp).
    """

    item_name: str = Field(
        ...,
        description="Canonical item name associated with the transaction; may be None for initialization rows."
    )
    transaction_type: str = Field(
        ...,
        description="Transaction kind: 'stock_orders' for purchases, 'sales' for customer revenue."
    )
    quantity: int = Field(
        ...,
        description="Number of units involved in the transaction."
    )
    price: float = Field(
        ...,
        description="Total transaction amount (quantity × per-unit price)."
    )
    date: str = Field(
        ...,
        description="ISO date or datetime string when the transaction occurred."
    )


class TransactionResult(BaseModel):
    """
    Outcome of writing stock and sales transactions.

    Purpose
    - Communicates the database IDs of created rows for traceability and auditing.
    """

    sale_transaction_ids: List[int] = Field(
        ...,
        description="IDs of created 'sales' transactions, in the order they were written."
    )
    stock_order_transaction_ids: List[int] = Field(
        ...,
        description="IDs of created 'stock_orders' transactions, in the order they were written."
    )


class QuoteResult(BaseModel):
    """
    Composite result combining the normalized Order and the generated Quote.

    Purpose
    - Serves as the data contract returned by the Orchestration Agent.
    - __str__ implements the project's compact, comma-separated presentation format.
    """

    order: Order = Field(
        ...,
        description="Normalized customer Order that was priced."
    )
    quote: Quote = Field(
        ...,
        description="Generated Quote containing itemized pricing, totals, and text."
    )

    def __str__(self) -> str:
        """
        Compact, comma-separated representation of the quote outcome:

        Format
        "<discounted_total_amount as integer>, <customer_quote>, <order_details JSON>"

        Notes
        - discounted_total_amount is rounded to the nearest integer for a 'friendly' display.
        - order_details contains request_date, expected_delivery_date, and the item lines.
        """
        total = int(round(self.quote.discounted_total_amount))
        order_details = {
            "request_date": self.order.request_date,
            "expected_delivery_date": self.order.expected_delivery_date,
            "items": [i.model_dump() for i in self.order.items],
        }
        return f"{total}, {self.quote.customer_quote}, {json.dumps(order_details)}"

# Enhanced parsing/matching entities

class RawOrderLine(BaseModel):
    text: str = Field(..., description="Raw line as mentioned by the customer")
    quantity_value: float = Field(..., description="Numeric quantity as provided")
    quantity_unit: str = Field(..., description="Unit as provided by customer: units|reams|pack|box|sheets|...")
    item_hint: Optional[str] = Field(None, description="Optional explicit item hint or alias")


class ParsedRequest(BaseModel):
    request_date: str = Field(..., description="ISO date YYYY-MM-DD")
    expected_delivery_date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD if specified")
    raw_items: List[RawOrderLine] = Field(..., description="Raw request items without unit conversion")
    notes: Optional[str] = Field(None, description="Optional notes or constraints")


class NormalizedRequestedItem(BaseModel):
    item_name: str = Field(..., description="Canonical inventory name")
    quantity_units: int = Field(..., description="Quantity converted to base units")
    unit_price: float = Field(..., description="Unit price pulled from inventory")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0..1")
    source_line: str = Field(..., description="Original raw line text")


class MatchingOutput(BaseModel):
    items: List[NormalizedRequestedItem] = Field(default_factory=list)
    unknown_items: List[str] = Field(default_factory=list)


#
# Agent Definitions
#


class OrderAgent:
    """
    Order Agent responsible for extracting order details from customer requests.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "order_agent"
        self.agent_name = "Order Agent"
        self.agent = Agent(
            model=OpenAIModel(OPENAI_DEFAULT_MODEL, provider=model_provider),
            output_type=OrderResult,
            system_prompt=self._get_system_prompt(),
            tools=[get_order_id_tool, get_all_inventory_items_tool],
        )

    def process_quote_request(self, request_text: str) -> OrderResult:
        log("Agent::process_quote_request")
        order_output = None
        try:
            agent_response = self.agent.run_sync(request_text)
            order_output = agent_response.output
            if order_output.is_success:
                log("Order extraction successful", LogLevel.INFO)
                return order_output
            else:
                log(f"Order extraction failed: {order_output.agent_error}", LogLevel.ERROR)
                log("Extracted order:", LogLevel.ERROR)
                log(order_output.model_dump_json(indent=2), LogLevel.ERROR)
        except Exception as e:
            log("Error processing quote requests", LogLevel.ERROR)
            print(e)
            order_output = OrderResult(
                is_success=False,
                order=None,
                agent_error=f"Error: {str(e)}",
            )
        return order_output

    @classmethod
    def _get_system_prompt(cls) -> str:
        """
        Returns the system prompt for the Order Agent.
        """
        return """
            You are an sales agent working in the sales department of a paper company.

            Your task is to process incoming order requests from customers and extract the date of the request and relevant order details from the request. You will receive a customer request as input, which may contain various details about the order, such as item names, quantities, and any special instructions. For each item requested by the customer, find the respective item in the inventory. Names of items might not match exactly, so you need to map them to the inventory items. If an item is not found in the inventory, you should return an error message indicating that the item is not available.

            If no expected delivery date is specified by the customer set the expected delivery date to 14 days after the request date.

            Your response should include the following fields:
            - `is_success`: A boolean indicating whether the order extraction was successful.
            - `order`: An Order object containing the extracted order details, including:
                - `request_date`: The date of the request in ISO format (YYYY-MM-DD).
                - `expected_delivery_date`: The expected delivery date for the order in ISO format (YYYY-MM-DD).
                - `items`: A list of OrderItem objects, each containing:
                    - `item_name`: The name of the item being ordered.
                    - `quantity`: The number of units to order.
                    - `unit_price`: The price per unit of the item being ordered.

            Provide quantities always as units (paper sheets, etc.). This means that you need to convert any other units (e.g., reams, packs) into units.
            - `reams`: A ream is 500 units.
            - `pack`: A pack is 100 units.
            - `box`: A box is 5000 units.
            
            Apply the following rules when extracting the order to map the items to the inventory:
            * Printer paper: standard copy paper
            * A3 paper: poster paper

            If the request does not contain sufficient information to extract an order, set `is_success` to False and provide an appropriate error message in the `agent_error` field.

            Think step by step.
            """


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCode(Enum):
    ITEM_NOT_FOUND = "ITEM_NOT_FOUND"
    STOCK_SHORTAGE = "STOCK_SHORTAGE"
    FUNDS_INSUFFICIENT = "FUNDS_INSUFFICIENT"
    DELIVERY_WINDOW_MISS = "DELIVERY_WINDOW_MISS"
    UNKNOWN = "UNKNOWN"


class FulfillmentError(Exception):
    """
    Base error for order fulfillment. Carries machine-readable code, severity,
    user-safe message, developer message, and context for debugging/telemetry.
    """

    def __init__(
        self,
        code: ErrorCode,
        user_message: str,
        dev_message: Optional[str] = None,
        severity: Severity = Severity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(dev_message or user_message)
        self.code = code
        self.user_message = user_message
        self.dev_message = dev_message or user_message
        self.severity = severity
        self.context = context or {}

    def add_context(self, **kwargs) -> "FulfillmentError":
        self.context.update(kwargs)
        return self

    def to_dict(self) -> Dict[str, Any]:
        def iso(v):
            if isinstance(v, datetime):
                return v.isoformat()
            return v

        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "user_message": self.user_message,
            "dev_message": self.dev_message,
            "context": {k: iso(v) for k, v in self.context.items()},
        }

    def to_user_message(self) -> str:
        return self.user_message

    def __str__(self) -> str:
        return f"{self.code.value}: {self.dev_message} | ctx={self.context}"


class ItemNotFoundError(FulfillmentError):
    def __init__(
        self,
        item_name: str,
        suggestions: Optional[List[str]] = None,
        dev_message: Optional[str] = None,
    ):
        user_msg = f"Item '{item_name}' is not in inventory."
        super().__init__(
            code=ErrorCode.ITEM_NOT_FOUND,
            user_message=user_msg,
            dev_message=dev_message or user_msg,
            severity=Severity.MEDIUM,
            context={"item_name": item_name, "suggestions": suggestions or []},
        )


class StockShortageError(FulfillmentError):
    def __init__(
        self,
        item_name: str,
        requested_qty: int,
        available_qty: int,
        earliest_restock: Optional[datetime] = None,
        dev_message: Optional[str] = None,
    ):
        user_msg = (
            f"Insufficient stock for '{item_name}': requested {requested_qty}, "
            f"available {available_qty}."
        )
        ctx = {
            "item_name": item_name,
            "requested_qty": requested_qty,
            "available_qty": available_qty,
        }
        if earliest_restock:
            ctx["earliest_restock"] = earliest_restock
        super().__init__(
            code=ErrorCode.STOCK_SHORTAGE,
            user_message=user_msg,
            dev_message=dev_message or user_msg,
            severity=Severity.HIGH,
            context=ctx,
        )


class InsufficientFundsError(FulfillmentError):
    def __init__(
        self,
        required_amount: float,
        available_cash: float,
        dev_message: Optional[str] = None,
    ):
        user_msg = (
            "The order exceeds the current maximum order amount. "
            "Please contact support or reduce the order."
        )
        super().__init__(
            code=ErrorCode.FUNDS_INSUFFICIENT,
            user_message=user_msg,
            dev_message=dev_message
            or f"Insufficient funds: required={required_amount}, available={available_cash}",
            severity=Severity.HIGH,
            context={
                "required_amount": required_amount,
                "available_cash": available_cash,
            },
        )


class DeliveryWindowMissError(FulfillmentError):
    def __init__(
        self,
        item_name: str,
        restock_date: datetime,
        expected_delivery: datetime,
        dev_message: Optional[str] = None,
    ):
        user_msg = (
            f"Restock for '{item_name}' arrives on {restock_date.date()}, "
            f"which is after the expected delivery {expected_delivery.date()}."
        )
        super().__init__(
            code=ErrorCode.DELIVERY_WINDOW_MISS,
            user_message=user_msg,
            dev_message=dev_message
            or f"Restock misses delivery window for {item_name}: restock={restock_date}, expected={expected_delivery}",
            severity=Severity.MEDIUM,
            context={
                "item_name": item_name,
                "restock_date": restock_date,
                "expected_delivery": expected_delivery,
            },
        )

class InventoryAgent:
    """
    Inventory Agent responsible for managing and querying inventory data.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "inventory_agent"
        self.agent_name = "Inventory Agent"
        self.agent = Agent(
            model=OpenAIModel(OPENAI_DEFAULT_MODEL, provider=model_provider),
            output_type=InventoryResult,
            system_prompt=InventoryAgent._get_system_prompt(),
            tools=[
                get_cash_balance_tool,
                get_supplier_delivery_date_tool,
                get_all_inventory_items_tool,
                get_stock_level_tool,
            ],
        )

    # LLM-based solution
    def process_order_llm(self, order: Order) -> InventoryResult:
        try:
            user_message = order.model_dump_json(indent=2)
            agent_result = self.agent.run_sync(user_message)
            return agent_result.output
        except Exception as e:
            log(f"Error processing order with LLM: {e}", LogLevel.ERROR)
            return InventoryResult(
                is_success=False,
                stock_order=None,
                agent_error=f"Error processing order with LLM: {str(e)}",
            )

    # Deterministic solution with enhanced error classes
    def process_order_direct(self, order: Order) -> InventoryResult:
        log("InventoryAgent::process_order", LogLevel.INFO)

        order_date = datetime.fromisoformat(order.request_date)
        expected_delivery_date = datetime.fromisoformat(order.expected_delivery_date)

        try:
            stock_orders: List[StockOrderItem] = []
            for item in order.items:
                stock_order_item = self._process_order_item(
                    item=item,
                    order_date=order_date,
                    expected_order_delivery_date=expected_delivery_date,
                )
                if stock_order_item is not None:
                    stock_orders.append(stock_order_item)

            stock_order = StockOrder(items=stock_orders)

            # Use order date as-of for cash balance
            self._check_against_cash_balance(stock_order, order_date)

            # Validate delivery window
            self._check_latest_delivery_date(
                expected_delivery_date=expected_delivery_date,
                stock=stock_order,
            )

            return InventoryResult(is_success=True, stock_order=stock_order)

        except ItemNotFoundError as e:
            log(str(e), LogLevel.ERROR)
            agent_error_message = e.to_user_message()

        except StockShortageError as e:
            log(str(e), LogLevel.ERROR)
            # Optionally enrich user message with earliest restock date if provided
            extra = ""
            try:
                er = e.context.get("earliest_restock")
                if er:
                    if isinstance(er, datetime):
                        extra = f" Earliest restock date is {er.date()}."
                    else:
                        extra = f" Earliest restock date is {er}."
            except Exception:
                pass
            agent_error_message = e.to_user_message() + extra

        except InsufficientFundsError as e:
            log(str(e), LogLevel.ERROR)
            agent_error_message = e.to_user_message()

        except DeliveryWindowMissError as e:
            log(str(e), LogLevel.ERROR)
            agent_error_message = e.to_user_message()

        except FulfillmentError as e:
            # Generic fulfillment error catch-all
            log(str(e), LogLevel.ERROR)
            agent_error_message = e.to_user_message()

        except Exception as e:
            log(f"Error processing order: {e}", LogLevel.ERROR)
            agent_error_message = f"Error processing order. An exception was raised during processing: {str(e)}"

        return InventoryResult(
            is_success=False,
            stock_order=None,
            agent_error=agent_error_message,
        )

    def _process_order_item(
        self,
        item: OrderItem,
        order_date: datetime,
        expected_order_delivery_date: datetime,
    ) -> Optional[StockOrderItem]:
        # static inventory data
        inventory_item = get_inventory_item(item.item_name)
        # "realtime" inventory data with accurate stock level
        stock_level_data = get_stock_level(item.item_name, order_date)

        if inventory_item is None or stock_level_data.empty:
            # Provide suggestions if possible
            suggestions = suggest_substitutes(item.item_name, k=3)
            log(f"Item {item.item_name} not found in inventory. Suggestions: {suggestions}", LogLevel.ERROR)
            raise ItemNotFoundError(item_name=item.item_name, suggestions=suggestions)

        stock_level = stock_level_data["current_stock"].iloc[0]
        stock_min_level = inventory_item.min_stock_level

        # insufficient stock
        if stock_level < item.quantity:
            restock_delivery_date = datetime.fromisoformat(
                get_supplier_delivery_date(order_date.date().isoformat(), item.quantity)
            )
            # assume at least 1 day for shipping to customer after restock
            if restock_delivery_date + timedelta(days=1) <= expected_order_delivery_date:
                # acceptable timeline → create stock order
                return StockOrderItem(
                    item_name=item.item_name,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    delivery_date=restock_delivery_date,
                )
            else:
                # not acceptable timeline → raise detailed shortage with earliest restock
                raise StockShortageError(
                    item_name=item.item_name,
                    requested_qty=item.quantity,
                    available_qty=int(stock_level),
                    earliest_restock=restock_delivery_date,
                )

        # stock sufficient, but below minimum after fulfillment
        if stock_level - item.quantity < stock_min_level:
            log(
                f"Stock for item {item.item_name} is below minimum level: issuing stock order.",
                LogLevel.INFO,
            )
            delivery_date_str = get_supplier_delivery_date(order_date.date().isoformat(), item.quantity)
            delivery_date = datetime.fromisoformat(delivery_date_str)
            return StockOrderItem(
                item_name=item.item_name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                delivery_date=delivery_date,
            )

        # no stock order necessary
        return None

    def _check_against_cash_balance(self, stock_order: StockOrder, as_of_date: datetime) -> bool:
        cash_balance = get_cash_balance(as_of_date)
        total_order_cost = sum(item.quantity * item.unit_price for item in stock_order.items if item)
        if cash_balance < total_order_cost:
            log(
                f"Insufficient funds: required {total_order_cost}, available {cash_balance}.",
                LogLevel.ERROR,
            )
            raise InsufficientFundsError(
                required_amount=float(total_order_cost),
                available_cash=float(cash_balance),
            )
        return True

    def _check_latest_delivery_date(self, expected_delivery_date: datetime, stock: StockOrder) -> bool:
        for item in stock.items:
            if item.delivery_date > expected_delivery_date:
                log(
                    f"Delivery window miss: {item.item_name} restock {item.delivery_date} > expected {expected_delivery_date}",
                    LogLevel.ERROR,
                )
                raise DeliveryWindowMissError(
                    item_name=item.item_name,
                    restock_date=item.delivery_date,
                    expected_delivery=expected_delivery_date,
                )
        return True

    @classmethod
    def _get_system_prompt(cls) -> str:
        """
        Inventory Agent System Prompt (LLM path)
        """
        return """
# Inventory Agent System Prompt

You are an Inventory Agent working in the sales department of a paper company. Your primary responsibility is to process incoming customer orders by analyzing stock levels and determining if restocking is required.

Process the order systematically and return the appropriate `StockOrder` result or raise the specified exceptions when validation fails.
        """


class QuotingAgent:
    """
    Quoting Agent responsible for generating quotes based on customer requests and inventory data.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "quoting_agent"
        self.agent_name = "Quoting Agent"
        self.agent = Agent(
            model=OpenAIModel(OPENAI_DEFAULT_MODEL, provider=model_provider),
            output_type=Quote,
            system_prompt=QuotingAgent.get_system_prompt(),
            tools=[search_quote_history_tool],
        )

    def generate_quote(self, order: Order, customer_request: str) -> Optional[Quote]:
        log(f"{self.agent_name}::generate_quote", LogLevel.INFO)
        order_total = sum([item.unit_price * item.quantity for item in order.items])
        message = (
            f"Generate a quote for the following order. "
            f"The order total without any discount is: {order_total}. "
            f"### Order Details: {order.model_dump_json()}\n"
            f"### Original Customer Request: {customer_request}."
        )
        try:
            response = self.agent.run_sync(message)
            return response.output
        except Exception as e:
            log(f"Error generating quote: {e}", LogLevel.ERROR)
            return None

    @classmethod
    def get_system_prompt(cls) -> str:
        return """
            You are a quoting agent working in the sales department of a paper company. Your company prides itself on providing competitive pricing and excellent customer service, and it offers various discount policies to its customers.

            Your task is to generate a quote based on the order details provided by the customer.

            Instructions:
            1. Search for similar quotes in the database of past quotes to determine the best pricing or discount strategy. Use the original customer request to search for similar quotes.
            2. Apply any applicable discount policies to the quote.
            3. Calculate the total amount for the quote, including any discounts applied.
            4. Ensure that the total amount is rounded to a friendly dollar value (integer, no fraction).
            5. Ensure that the discount is applied on the individual item level so the sum of item totals matches the discounted total amount.
            6. Generate a customer quote text that summarizes the order and pricing details.

            Your response should include the following fields:
            - `order`: An Order object containing the order details.
            - `quote_items`: List of QuoteItem with item_name, quantity, discounted_price.
            - `customer_quote`: The customer-facing quote text.
            - `total_amount`: Total before discounts.
            - `discounted_total_amount`: Total after discounts, rounded to a full dollar amount.
            - `discount_amount`: The total amount discounted.
            - `discount_policy`: Optional DiscountPolicy object describing the policy applied.
        """


class TransactionAgent:
    """
    Transaction Agent responsible for managing financial transactions related to orders and inventory.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "transaction_agent"
        self.agent_name = "Transaction Agent"

    def process_transactions(self, order: Order, quote: Quote, stock_order: StockOrder) -> TransactionResult:
        log("Agent::process_transactions", LogLevel.INFO)

        order_request_date = order.request_date

        # First, create stock order transactions
        stock_order_transaction_ids = []
        try:
            for stock_item in stock_order.items:
                id = create_transaction(
                    item_name=stock_item.item_name,
                    transaction_type="stock_orders",
                    quantity=stock_item.quantity,
                    price=stock_item.unit_price * stock_item.quantity,
                    date=order_request_date,
                )
                stock_order_transaction_ids.append(id)
        except Exception as e:
            log(f"Error processing stock order transactions: {e}", LogLevel.ERROR)
            raise Exception("An error occurred while processing stock order transactions.")

        log(f"Stock order transactions created: {len(stock_order_transaction_ids)}", LogLevel.INFO)

        # Second, create sales transactions based on the quote
        sales_transaction_ids = []
        try:
            for quote_item in quote.quote_items:
                id = create_transaction(
                    item_name=quote_item.item_name,
                    transaction_type="sales",
                    quantity=quote_item.quantity,
                    price=quote_item.discounted_price * quote_item.quantity,
                    date=order_request_date,
                )
                sales_transaction_ids.append(id)
        except Exception as e:
            log(f"Error processing sales transactions: {e}", LogLevel.ERROR)
            raise Exception("An error occurred while processing sales transactions.")

        log(f"Sales transactions created: {len(sales_transaction_ids)}", LogLevel.INFO)

        result = TransactionResult(
            sale_transaction_ids=sales_transaction_ids,
            stock_order_transaction_ids=stock_order_transaction_ids,
        )
        return result


class WorkflowError(Enum):
    ORDER_ERROR = "Failed to extract order details"
    INVENTORY_ERROR = "Inventory check failed"
    QUOTING_ERROR = "Quote generation failed"
    TRANSACTION_ERROR = "Transaction processing failed"


class OrchestrationAgent:
    """
    Orchestration Agent:
      1) ExtractionAgent → ParsedRequest structured
      2) CatalogMatcher deterministic → Normalized items
      3) Build Order with normalized items
      4) InventoryAgent
      5) QuotingAgent
      6) TransactionAgent
      7) Clarification if unknown or low-confidence items
    """

    def __init__(self, model_provider: OpenAIProvider, confidence_threshold: float = 0.55, use_llm_inventory: bool = False):
        self.agent_id = "orchestration_agent"
        self.agent_name = "Orchestration Agent"

        self.extraction_agent = ExtractionAgent(model_provider=model_provider)
        self.matcher = CatalogMatcher(confidence_threshold=confidence_threshold)
        self.inventory_agent = InventoryAgent(model_provider=model_provider)
        self.quoting_agent = QuotingAgent(model_provider=model_provider)
        self.transaction_agent = TransactionAgent(model_provider=model_provider)
        self.use_llm_inventory = use_llm_inventory

    def process_quote_request(self, request_text: str) -> str:
        # Step 1: Extract
        parsed = self.extraction_agent.parse_request(request_text)

        # Guardrails: request_date fallback
        try:
            req_date = parsed.request_date or datetime.now().strftime("%Y-%m-%d")
            if not parsed.request_date:
                if parsed.notes:
                    parsed.notes += " | request_date not provided; defaulted to today."
                else:
                    parsed.notes = "request_date not provided; defaulted to today."
        except Exception:
            req_date = datetime.now().strftime("%Y-%m-%d")

        # Step 2: Match and normalize
        with status("Catalog Matching - normalize_items"):
            match_out = self.matcher.normalize(parsed)

        # Clarification path if unknown or low-confidence items
        if match_out.unknown_items:
            suggestions = []
            for text_line in match_out.unknown_items:
                cands = get_catalog_matches(text_line, top_k=3)
                suggestions.append({
                    "requested": text_line,
                    "suggestions": [c["item_name"] for c in cands],
                })
            return json.dumps({
                "status": "clarification_required",
                "reason": "Some items could not be confidently matched to inventory.",
                "unknown_items": suggestions,
            })

        # Step 3: Build Order
        expected_date = parsed.expected_delivery_date
        if not expected_date:
            expected_date = (datetime.fromisoformat(req_date) + timedelta(days=14)).strftime("%Y-%m-%d")

        order = Order(
            id=uuid.uuid4().hex,
            request_date=req_date,
            expected_delivery_date=expected_date,
            items=[
                OrderItem(
                    item_name=n.item_name,
                    quantity=n.quantity_units,
                    unit_price=n.unit_price,
                )
                for n in match_out.items
            ],
        )

        # Step 4: Inventory
        with status("Inventory Agent - process_order"):
            if self.use_llm_inventory:
                inv_res = self.inventory_agent.process_order_llm(order=order)
            else:
                inv_res = self.inventory_agent.process_order_direct(order=order)

            if not inv_res or not inv_res.is_success:
                return f"Ordering Error: {inv_res.agent_error if inv_res else 'Unknown inventory error'}"

        stock_order = inv_res.stock_order

        # Step 5: Quote
        with status("Quoting Agent - generate_quote"):
            quote = self.quoting_agent.generate_quote(order=order, customer_request=request_text)
            if not quote:
                return "Ordering Error: We cannot provide a quote for your request at this point in time."

        # Step 6: Transactions
        with status("Transaction Agent - process_transactions"):
            trx_res = self.transaction_agent.process_transactions(order=order, quote=quote, stock_order=stock_order)
            if not trx_res:
                return "Ordering Error: An error occurred while processing transactions."

        response = QuoteResult(order=order, quote=quote)
        return str(response)

class ExtractionAgent:
    """
    Extracts a ParsedRequest from raw user text with strict structured output.
    Does NOT convert units; preserves raw items as mentioned by the customer.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "extraction_agent"
        self.agent_name = "Extraction Agent"
        self.agent = Agent(
            model=OpenAIModel(OPENAI_DEFAULT_MODEL, provider=model_provider),
            output_type=ParsedRequest,
            system_prompt=self._get_system_prompt(),
            tools=[],
        )

    def parse_request(self, request_text: str) -> ParsedRequest:
        with status("Extraction Agent - parse_request"):
            res = self.agent.run_sync(request_text)
            return res.output

    @classmethod
    def _get_system_prompt(cls) -> str:
        return """
        You are a structured information extractor.
        Goal: Parse the customer's text into a ParsedRequest JSON strictly matching the schema.

        Rules:
        - Do NOT convert units. Preserve the customer's unit choices (reams, packs, boxes, units, sheets, etc.).
        - Extract request_date YYYY-MM-DD if available; else omit and leave a note if needed.
        - Extract expected_delivery_date only if clearly specified; otherwise omit.
        - raw_items: Each item must include:
        - text: short normalized text e.g. A4 paper
        - quantity_value: numeric value as provided
        - quantity_unit: the unit as provided e.g. reams
        - item_hint: optional alias e.g. printer paper

        - notes: include any constraints like budgets or special instructions.

        Return only valid JSON conforming to ParsedRequest.
        """


class CatalogMatcher:
    def __init__(self, confidence_threshold: float = 0.55, fallback_min: float = 0.40, top_k: int = 5):
        self.confidence_threshold = confidence_threshold
        self.fallback_min = fallback_min
        self.top_k = top_k

    def normalize(self, parsed: ParsedRequest) -> MatchingOutput:
        norm_items, unknowns = [], []
        inv_df = fetch_inventory_df()

        for line in parsed.raw_items:
            query = (line.item_hint or line.text or "").strip()
            log(f"[MATCHER] query='{query}'", LogLevel.DEBUG)

            # A0) exact inventory match
            exact = find_exact_in_inventory(query, inv_df)
            if exact:
                log(f"[MATCHER] exact hit -> {exact['item_name']}", LogLevel.DEBUG)
                qty_units = convert_units(line.quantity_value, line.quantity_unit)
                norm_items.append(NormalizedRequestedItem(
                    item_name=exact["item_name"], quantity_units=qty_units,
                    unit_price=exact["unit_price"], confidence=0.99, source_line=line.text
                ))
                continue

            # A1) canonical targets
            canon = canonical_targets_for(query)
            log(f"[MATCHER] canonical targets -> {canon}", LogLevel.DEBUG)
            if canon:
                hit = pick_first_available(inv_df, canon)
                if hit:
                    log(f"[MATCHER] canonical hit -> {hit['item_name']}", LogLevel.DEBUG)
                    qty_units = convert_units(line.quantity_value, line.quantity_unit)
                    norm_items.append(NormalizedRequestedItem(
                        item_name=hit["item_name"], quantity_units=qty_units,
                        unit_price=hit["unit_price"], confidence=0.99, source_line=line.text
                    ))
                    continue

            # B) fuzzy
            candidates = get_catalog_matches(query, self.top_k)
            log(f"[MATCHER] candidates -> {[c['item_name'] for c in candidates]}", LogLevel.DEBUG)
            best = candidates[0] if candidates else None

            if best and best["score"] >= self.confidence_threshold:
                log(f"[MATCHER] accepted@threshold -> {best['item_name']} score={best['score']:.2f}", LogLevel.DEBUG)
                qty_units = convert_units(line.quantity_value, line.quantity_unit)
                norm_items.append(NormalizedRequestedItem(
                    item_name=best["item_name"], quantity_units=qty_units,
                    unit_price=best["unit_price"], confidence=float(best["score"]),
                    source_line=line.text
                ))
            elif best and best["score"] >= self.fallback_min:
                log(f"[MATCHER] accepted@fallback -> {best['item_name']} score={best['score']:.2f}", LogLevel.DEBUG)
                qty_units = convert_units(line.quantity_value, line.quantity_unit)
                norm_items.append(NormalizedRequestedItem(
                    item_name=best["item_name"], quantity_units=qty_units,
                    unit_price=best["unit_price"], confidence=float(best["score"]),
                    source_line=line.text
                ))
            else:
                log(f"[MATCHER] unknown -> '{line.text}'", LogLevel.DEBUG)
                unknowns.append(line.text)

        return MatchingOutput(items=norm_items, unknown_items=unknowns)

def suggest_substitutes(query: str, k: int = 3) -> List[str]:
    try:
        cands = get_catalog_matches(query, top_k=k)
        return [c["item_name"] for c in cands]
    except Exception:
        return []

def info_tool_call(func):
    """
    Decorator that prints the function name and returns the function with its doc-string.
    Intended to be used with tool decorators for debugging and documentation.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log(f"Tool: {func.__name__}", LogLevel.INFO)
        return func(*args, **kwargs)
    return wrapper


def tool(func):
    """
    Decorator to mark a function as a Pydantic-AI Tool to be used by agents.
    """
    return Tool(info_tool_call(func))


def get_all_inventory_items() -> List[InventoryItem]:
    """
    Retrieves all items from the inventory as of the current date.
    """
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    items = [
        InventoryItem(
            item_name=row["item_name"],
            category=row["category"],
            unit_price=row["unit_price"],
            min_stock_level=row["min_stock_level"],
        )
        for _, row in inventory_df.iterrows()
    ]
    return items


@tool
def get_all_inventory_items_tool() -> List[InventoryItem]:
    """
    Retrieves all items from the inventory as of the current date.
    """
    return get_all_inventory_items()


def get_inventory_item(item_name: str) -> Optional[InventoryItem]:
    """
    Retrieves a specific item from the inventory by its name.
    """
    inventory_df = pd.read_sql(
        text("SELECT * FROM inventory WHERE item_name = :item_name"),
        db_engine,
        params={"item_name": item_name},
    )
    if inventory_df.empty:
        return None
    row = inventory_df.iloc[0]
    return InventoryItem(
        item_name=row["item_name"],
        category=row["category"],
        unit_price=row["unit_price"],
        min_stock_level=row["min_stock_level"],
    )


@tool
def get_inventory_item_tool(item_name: str) -> Optional[InventoryItem]:
    """
    Retrieves a specific item from the inventory by its name.
    """
    return get_inventory_item(item_name=item_name)


@tool
def get_order_id_tool(request_text: str) -> str:
    """
    Extracts/generates an order ID.
    """
    return uuid.uuid4().hex


@tool
def search_quote_history_tool(order: Order, customer_request: str) -> List[Dict]:
    """
    Returns a list of similar quotes from the quote history based on the customer request.
    """
    log("Searching Quote History", LogLevel.DEBUG)
    log(f"Original Customer Request: {customer_request}", LogLevel.DEBUG)

    search_terms = [item.item_name.strip().lower() for item in order.items]
    search_results = search_quote_history(search_terms=search_terms, limit=5)
    if search_results and len(search_results) > 0:
        log(
            f"Found similar quotes: {len(search_results)} for products {', '.join(search_terms)}",
            LogLevel.INFO,
        )
    return search_results


@tool
def get_cash_balance_tool(as_of_date: str) -> float:
    """
    Retrieves the cash balance as of a specific date.
    """
    return get_cash_balance(as_of_date=as_of_date)


@tool
def get_supplier_delivery_date_tool(order_date: str, quantity: int) -> str:
    """
    Retrieves the expected delivery date from the supplier for a given order date and quantity.
    """
    return get_supplier_delivery_date(input_date_str=order_date, quantity=quantity)


@tool
def get_stock_level_tool(item_name: str, as_of_date: str) -> str:
    """
    Retrieves the stock level of a specific item as of a given date.

    Returns:
        str: JSON-serialized DataFrame with columns ['item_name', 'current_stock'].
    """
    return get_stock_level(item_name=item_name, as_of_date=as_of_date).to_json()


# Deterministic tools used by CatalogMatcher

UNIT_MAP = {
    "unit": 1,
    "units": 1,
    "sheet": 1,
    "sheets": 1,
    "ream": 500,
    "reams": 500,
    "pack": 100,
    "packs": 100,
    "box": 5000,
    "boxes": 5000,
}


def convert_units(quantity_value: float, quantity_unit: str) -> int:
    factor = UNIT_MAP.get(str(quantity_unit).strip().lower(), 1)
    return int(round(float(quantity_value) * factor))




# Lightweight synonyms for better similarity
SYNONYMS = {
    "printer paper": "standard copy paper",
    "copy paper": "standard copy paper",
    "copier paper": "standard copy paper",
    "multi purpose": "standard copy paper",
    "multipurpose": "standard copy paper",
    "multi-purpose": "standard copy paper",
    "bond paper": "uncoated paper",
    "posterboard": "poster paper",
    "poster board": "poster paper",
    "drawing paper": "poster paper",           # default
    "display board": "poster paper",
    "masking tape": "decorative adhesive tape (washi tape)",
    "packaging tape": "decorative adhesive tape (washi tape)",
    "craft paper": "crepe paper",
    "washi": "decorative adhesive tape (washi tape)",
    "washi tape": "decorative adhesive tape (washi tape)",
    "invite cards": "invitation cards",
    "invitation sets": "invitation cards",
}

# Canonical preference lists per phrase pattern. First available in inventory is chosen.
ALIAS_MAP: List[Tuple[str, List[str]]] = [
    # Core office sizes and types
    (r"\ba4\b.*\bprinter\b|\ba4\b.*\bprinting\b|\ba4\b.*\bcopy\b|\ba4\b\s*(paper|sheets)\b", ["A4 paper", "Standard copy paper", "Letter-sized paper"]),
    (r"\ba3\b.*\bprinter\b|\ba3\b.*\bprinting\b|\ba3\b.*\bposter\b|\ba3\b\s*(paper|sheets)\b", ["Poster paper", "220 gsm poster paper", "Large poster paper (24x36 inches)"]),
    (r"\ba5\b.*\bprinter\b|\ba5\b.*\bprinting\b|\ba5\b\s*(paper|sheets)\b", ["Standard copy paper", "A4 paper", "Letter-sized paper"]),
    (r"\b8\.5\s*[x×]\s*11\b|\bletter\s*size(d)?\b", ["Standard copy paper", "Letter-sized paper", "A4 paper"]),
    (r"\b11\s*[x×]\s*17\b|\btabloid\b|\bledger\b", ["Poster paper", "Large poster paper (24x36 inches)"]),
    (r"\b24\s*[x×]\s*36\b", ["Large poster paper (24x36 inches)", "Poster paper"]),

    # Rolls and large-format
    (r"\brolls?\b.*\bbanner\b|\bbanner\b.*\brolls?\b|\b36\s*inch\b.*\bbanner\b", ["Rolls of banner paper (36-inch width)"]),
    (r"\bbanner\s*paper\b", ["Banner paper", "Rolls of banner paper (36-inch width)"]),

    # Photo, glossy, matte, uncoated
    (r"\bglossy\s*photo\b", ["Photo paper"]),
    (r"\bphoto\s*paper\b", ["Photo paper"]),
    (r"\bglossy\b.*\bpaper\b", ["Glossy paper", "Photo paper"]),
    (r"\bmatte\b.*\bpaper\b", ["Matte paper"]),
    (r"\buncoated\b.*\bpaper\b|\bbond\b.*\bpaper\b", ["Uncoated paper", "Standard copy paper"]),

    # Eco and recycled
    (r"\beco[- ]?friendly\b.*\bpaper\b", ["Eco-friendly paper", "Recycled paper"]),
    (r"\brecycled\b.*\bpaper\b", ["Recycled paper", "Eco-friendly paper"]),
    (r"\beco[- ]?friendly\b.*\bcard\s*stock|eco[- ]?friendly\b.*\bcardstock\b", ["250 gsm cardstock", "Cardstock"]),

    # Cardstock and weights
    (r"\bcard\s*stock|cardstock\b", ["Cardstock", "250 gsm cardstock", "100 lb cover stock"]),
    (r"\bheavy\s*(card\s*stock|cardstock)\b|\bheavy[- ]?weight\b.*\bcard(stock)?\b", ["250 gsm cardstock", "Heavyweight paper", "Cardstock"]),
    (r"\b100\s*lb\b.*\bcover\b", ["100 lb cover stock"]),
    (r"\b80\s*lb\b.*\btext\b", ["80 lb text paper"]),
    (r"\b220\s*gsm\b.*\bposter\b", ["220 gsm poster paper"]),
    (r"\b220\s*gsm\b.*\bcard(stock)?\b|\b200\s*gsm\b.*\bcard(stock)?\b|\b250\s*gsm\b.*\bcard(stock)?\b", ["250 gsm cardstock", "Cardstock"]),

    # Colors and decorative
    (r"\bcolored\s*poster\s*paper\b|\bcolorful\s*poster\s*paper\b", ["Poster paper", "220 gsm poster paper"]),
    (r"\bcolored\s*paper\b|\bcolorful\s*paper\b|\bprinter\s*paper.*colorful\b", ["Colored paper", "Bright-colored paper", "Construction paper"]),
    (r"\bcolored\s*card(stock)?\b|\bcolorful\s*card(stock)?\b", ["Cardstock", "250 gsm cardstock", "100 lb cover stock"]),
    (r"\bconstruction\s*paper\b", ["Construction paper"]),
    (r"\bwrapping\s*paper\b", ["Wrapping paper"]),
    (r"\bdecorative\s*paper\b", ["Decorative paper"]),
    (r"\bglitter\s*paper\b", ["Glitter paper"]),
    (r"\bkraft\s*paper\b", ["Kraft paper"]),

    # Posters and boards
    (r"\bposter\s*paper\b", ["Poster paper", "220 gsm poster paper"]),
    (r"\bposter\s*board(s)?\b|\bdisplay\s*board(s)?\b", ["Poster paper", "Large poster paper (24x36 inches)"]),

    # Envelopes, letterhead, flyers, folders
    (r"\bletterhead\b", ["Letterhead paper"]),
    (r"\benvelope(s)?\b", ["Envelopes"]),
    (r"\bflyers?\b", ["Flyers"]),
    (r"\bfolder(s)?\b.*\bpresentation\b|\bpresentation\b.*\bfolder(s)?\b", ["Presentation folders"]),
    (r"\binvitation\b.*\bcard(s)?\b|\bdecorative\b.*\binvitation\b", ["Invitation cards"]),

    # Notes and notepads
    (r"\bsticky\s*note(s)?\b", ["Sticky notes"]),
    (r"\bnote\s*pad(s)?\b|\bnotepad(s)?\b", ["Notepads"]),

    # Party supplies
    (r"\bparty\s*streamers?\b|\sstreamers?\b", ["Party streamers"]),
    (r"\bpaper\s*plate(s)?\b|\bplate(s)?\b", ["Paper plates"]),
    (r"\bpaper\s*cups?\b|\bcups?\b|\bdisposable\s*cups?\b", ["Paper cups", "Disposable cups"]),
    (r"\bnapkin(s)?\b", ["Paper napkins"]),
    (r"\btable\s*cover(s)?\b", ["Table covers"]),
    (r"\bparty\s*bag(s)?\b|\bpaper\s*party\s*bag(s)?\b", ["Paper party bags"]),
    (r"\bname\s*tag(s)?\b", ["Name tags with lanyards"]),
    (r"\btape\b|\bmasking\s*tape\b|\bpackaging\s*tape\b", ["Decorative adhesive tape (washi tape)"]),
]

def normalize_text(s: str) -> str:
    return " ".join(str(s).lower().replace('"', "").replace("’", "'").split())

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def _synonymize(s: str) -> str:
    s_norm = normalize_text(s)
    for k, v in SYNONYMS.items():
        s_norm = s_norm.replace(k, v)
    # size hints -> canonical tokens
    s_norm = s_norm.replace("8.5 x 11", "8.5x11").replace("8.5x 11", "8.5x11")
    s_norm = s_norm.replace("11 x 17", "11x17").replace("24 x 36", "24x36")
    return s_norm

def token_score(query: str, candidate: str) -> float:
    q = _synonymize(query)
    c = _synonymize(candidate)
    ratio = SequenceMatcher(None, q, c).ratio()
    tq, tc = set(_tokenize(q)), set(_tokenize(c))
    jacc = (len(tq & tc) / len(tq | tc)) if (tq | tc) else 0.0
    substr = 1.0 if any(tok in c for tok in tq if len(tok) >= 3) else 0.0
    return 0.55 * ratio + 0.35 * jacc + 0.10 * substr

def fetch_inventory_df() -> pd.DataFrame:
    return pd.read_sql("SELECT item_name, category, unit_price FROM inventory", db_engine)

def find_exact_in_inventory(name: str, inv_df: pd.DataFrame) -> Optional[Dict]:
    row = inv_df.loc[inv_df["item_name"].str.lower() == str(name).lower()]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "item_name": str(r["item_name"]),
        "unit_price": float(r["unit_price"]),
        "category": str(r["category"]),
        "score": 1.0,
    }

def pick_first_available(inv_df: pd.DataFrame, names: List[str]) -> Optional[Dict]:
    for n in names:
        hit = find_exact_in_inventory(n, inv_df)
        if hit:
            return hit
    return None

def canonical_targets_for(query: str) -> Optional[List[str]]:
    q = _synonymize(query)
    for pattern, targets in ALIAS_MAP:
        if re.search(pattern, q):
            return targets
    # size-only heuristics when no phrase patterns matched
    if re.search(r"\b24x36\b", q):
        return ["Large poster paper (24x36 inches)", "Poster paper"]
    if re.search(r"\b11x17\b", q):
        return ["Poster paper", "Large poster paper (24x36 inches)"]
    if re.search(r"\b8\.5x11\b|\bletter\s*size(d)?\b", q):
        return ["Standard copy paper", "Letter-sized paper", "A4 paper"]
    return None

# Soft category preference by keyword
CATEGORY_KEYWORDS = {
    "poster": {"paper", "specialty", "large_format"},
    "cardstock": {"paper", "specialty"},
    "gsm": {"specialty"},
    "washi": {"product"},
    "tape": {"product"},
    "streamer": {"product"},
    "envelope": {"product"},
    "notepad": {"product"},
    "sticky": {"product"},
    "printer paper": {"paper"},
    "copy paper": {"paper"},
    "banner": {"large_format", "paper"},
}

def preferred_categories(query: str) -> Set[str]:
    q = _synonymize(query)
    prefs: Set[str] = set()
    for kw, cats in CATEGORY_KEYWORDS.items():
        if kw in q:
            prefs |= cats
    return prefs

def search_inventory(query: str, inv_df: pd.DataFrame) -> List[Dict]:
    prefs = preferred_categories(query)
    scored = []
    for _, r in inv_df.iterrows():
        name = str(r["item_name"])
        cat = str(r["category"])
        score = token_score(query, name)
        if prefs and cat in prefs:
            score += 0.08  # soft boost
        scored.append(
            {"item_name": name, "unit_price": float(r["unit_price"]), "category": cat, "score": float(score)}
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def get_catalog_matches(query_text: str, top_k: int = 5) -> List[Dict]:
    inv_df = fetch_inventory_df()

    # 0) exact inventory name match
    exact = find_exact_in_inventory(query_text, inv_df)
    if exact:
        return [exact]

    cands = canonical_targets_for(query_text)
    if cands:
        hit = pick_first_available(inv_df, cands)
        if hit:
            return [hit]
        pooled = []
        for canon in cands:
            pooled.extend(search_inventory(canon, inv_df))
        if pooled:
            pooled.sort(key=lambda x: x["score"], reverse=True)
            return pooled[:top_k]

    return search_inventory(query_text, inv_df)[:top_k]

@tool
def convert_units_tool(quantity_value: float, quantity_unit: str) -> int:
    """
    Convert a quantity to base units.
    Known units: unit(s), sheet(s), ream(s)=500, pack(s)=100, box(es)=5000.
    Returns:
        int: quantity in units.
    """
    return convert_units(quantity_value, quantity_unit)


@tool
def get_catalog_matches_tool(query_text: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k catalog candidates for the given free-text query.
    Returns list of dicts: {item_name, score, unit_price, category}
    """
    return get_catalog_matches(query_text, top_k)


def run_test_scenarios(
    sample_limit: int = None,
    use_llm_inventory: bool = False,
    confidence_threshold: float = 0.55,
):
    """
    Runs all test scenarios from quote_requests_sample.csv in chronological order.
    Prints responses and saves test_results.csv.
    """
    log("Initializing Database...", LogLevel.INFO)
    init_database(db_engine=db_engine)

    # Load sample requests
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        log(f"FATAL: Error loading test data: {e}", LogLevel.ERROR)
        return []

    # Provider setup
    dotenv.load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        log("FATAL: OPENAI_API_KEY not set in environment or .env file.", LogLevel.ERROR)
        return []

    openai_provider = OpenAIProvider(
        base_url="https://openai.vocareum.com/v1",
        api_key=OPENAI_API_KEY,
    )

    orchestrator = OrchestrationAgent(
        model_provider=openai_provider,
        confidence_threshold=confidence_threshold,
        use_llm_inventory=use_llm_inventory,
    )

    # Initial financial snapshot
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Determine number of samples to process
    total_rows = len(quote_requests_sample)
    if sample_limit is None or sample_limit <= 0 or sample_limit > total_rows:
        sample_limit = total_rows

    results = []
    for idx, row in quote_requests_sample.iterrows():
        if idx >= sample_limit:
            break

        request_date = row["request_date"].strftime("%Y-%m-%d")
        context_job = row.get("job", "")
        context_event = row.get("event", "")
        request_text = row.get("request", "")

        print(f"\n=== Request {idx+1} / {sample_limit} ===")
        print(f"Context: {context_job} organizing {context_event}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Include the request date in the prompt for consistency
        request_with_date = f"{request_text} (Date of request: {request_date})"

        # Process request
        response = orchestrator.process_quote_request(request_text=request_with_date)

        # Update financials as-of this request date
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    # Adjust logging
    # log_level = LogLevel.DEBUG
    log_level = LogLevel.INFO

    # Run all or a subset
    results = run_test_scenarios(
        sample_limit=None,          # or an integer like 5
        use_llm_inventory=None,  # True to use Inventory LLM path
        confidence_threshold=0.55 # tuned for fewer clarifications
    )
