Overview 

This project implements a multi-agent paper-supplies quoting system backed by an SQLite database. It reads natural-language customer requests, extracts orders, checks inventory and delivery timelines, generates a customer quote, and records financial transactions.

Agent workflow diagram explanation 

The workflow diagram shows a request flowing through four agents with a clarification branch and tool calls at each step.

Orchestration Agent
Role: Coordinates the pipeline, enforces step order, aggregates results, and handles failures or clarifications.
Decision points:
If Catalog Matcher reports unknown or low-confidence items, return a single clarification payload before any inventory or quoting calls.
Select inventory path: deterministic by default; optional LLM path exists for parity and experimentation.

Extraction Agent
Role: Convert raw request text into a structured ParsedRequest with dates and raw line items. This is the first LLM step.
Why LLM here: robust language understanding and schema filling; downstream components rely on typed data.

Catalog Matcher
Role: Deterministically normalize raw items to canonical SKUs and convert units to base quantities.
Tools and data:
convert_units_tool → convert_units helper for ream/pack/box/sheets mapping.
get_catalog_matches_tool → get_catalog_matches helper for alias-then-fuzzy catalog lookup.
Why deterministic: reduces hallucination risk, provides traceable decisions, and supports confidence thresholds that drive the clarification loop.

Inventory Agent
Role: Validate feasibility of the order: stock availability, minimum-stock replenishment, supplier lead time (+1 day shipping), and cash balance as-of the request date.
Tools and data:
get_stock_level_tool → get_stock_level helper for net stock as-of date
get_supplier_delivery_date_tool → lead-time heuristic
get_cash_balance_tool → cash as-of date
get_all_inventory_items_tool → static inventory reference
Design choice: deterministic path is default for reliability; an LLM path is available but off by default.

Quoting Agent
Role: Produce a Quote with per-item discounted prices, totals, and customer-facing text. This is the second LLM step.
Tools and data:
search_quote_history_tool → search_quote_history helper to guide policy and tone
Rationale: LLM excels at discount narrative and style; deterministic reconciliation can be added to enforce total consistency.

Transaction Agent
Role: Persist stock_orders and sales as transactions in a single source of truth.
Data helpers:
create_transaction for atomic inserts and returning IDs
Rationale: fully deterministic and idempotent-friendly, enabling accurate financial reporting.
Why this architecture

Separation of concerns: Language understanding (LLM) is isolated to Extraction and customer messaging; everything else is deterministic and testable.
Guardrails via confidence and clarifications: Matching returns confidence; low-confidence items trigger a single clarification loop before any costly operations.
Operational reliability: Inventory checks and transactions are deterministic to ensure predictable outcomes and straightforward debugging.
Extensibility: The matcher accepts new aliases and thresholds; the inventory step can switch between deterministic and LLM paths without changing orchestration.

What the system returns

Success: a formatted string "total, quote_text, order_json"
Need more info: a clarification payload such as {"status": "clarification_required", "reason": "...", "unknown_items": [{"requested": "x", "suggestions": ["A", "B"]}]}
Error: "Ordering Error: <agent_error>"
End-to-end flow

Request received
Extraction Agent runs (LLM) and returns a ParsedRequest
Catalog Matcher normalizes items to canonical SKUs and converts units
If any item is unknown or below confidence threshold, a clarification payload is returned
Build an Order from normalized items
Inventory Agent runs
Deterministic by default: checks stock, min levels, restock lead time (+1 day to ship), cash balance
If a restock arrives after expected delivery, throws RestockTimeoutError (handled gracefully)
Quoting Agent runs (LLM) to produce quote items, discounts, text
Transactions are recorded for stock_orders and sales
Orchestration Agent returns the final string
Where LLMs are used

Extraction Agent: always
Quoting Agent: always
Inventory Agent: only if use_llm_inventory=True (deterministic path is default)
Data model

SQLite database file: munder_difflin.db
Tables:
inventory: item_name, category, unit_price, current_stock, min_stock_level
transactions: id (auto), item_name, transaction_type ('stock_orders' or 'sales'), units, price, transaction_date
quote_requests: original requests used for history
quotes: past quotes joined with quote_requests, including metadata such as job_type, order_size, event_type
Initialization:
inventory is generated from paper_supplies with random stocks and min levels (reproducible via seed)
transactions seeded with starting cash and historical stock purchases
quotes and quote_requests loaded from CSV
Unit conventions and lead time rules

Units conversion (customer input → base units)
ream = 500
pack = 100
box = 5000
sheets/units = 1
Restock lead times:
quantity ≤ 10: same day
11–100: +1 day
101–1000: +4 days
1000: +7 days

Delivery feasibility requires restock_date + 1 day ≤ expected_delivery_date
Catalog matcher highlights

Curated alias patterns for common phrases: printer paper → Standard copy paper; A3 → Poster paper; washi → Decorative adhesive tape (washi tape), etc.
Size/weight heuristics: A3, A4, A5, 8.5x11, 11x17, 24x36; GSM and lb text weights mapped to closest SKUs.
Soft category preference: boosts rather than excludes.
Fallback acceptance: if best match is below the main threshold but above a lower bound, it is auto-accepted to reduce clarification loops.
If a canonical target is not in the current inventory due to coverage, it fuzzy-searches canonical names to pick the closest stocked alternative (e.g., Poster paper → 220 gsm poster paper).
Requirements

Python 3.10+
CSVs in the project root:
quote_requests.csv
quotes.csv
quote_requests_sample.csv

Python packages:
pydantic>=2.6
sqlalchemy>=2.0
pandas
numpy
python-dotenv
pydantic-ai

Install dependencies

python -m venv .venv
source .venv/bin/activate (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install pydantic "sqlalchemy>=2.0" pandas numpy python-dotenv pydantic-ai

Environment variables

Required:
OPENAI_API_KEY 
Optional:
OPENAI_BASE_URL (defaults to https://openai.vocareum.com/v1)
Set via .env:
OPENAI_API_KEY=your_api_key_here
Or export in shell:
macOS/Linux: export OPENAI_API_KEY=your_api_key_here
Windows PowerShell: setx OPENAI_API_KEY "your_api_key_here" (restart terminal)

How to run

Ensure CSVs and .env are present in the working directory.
Delete any stale DB if desired: rm munder_difflin.db
Run:
python project_starter.py
The script:
Initializes the database
Iterates through quote_requests_sample.csv in date order
For each request, runs the agentic pipeline
Prints response and financial updates
Writes test_results.csv

Evaluation results summary 

Based on the generated test_results.csv, the system shows several strengths:

Robust item normalization: Most standard office-supply requests mapped correctly without human clarification, especially size-aware requests such as A4, A3, 8.5x11, and large-format 24x36. The curated aliases (e.g., printer paper → Standard copy paper, A3 → Poster paper, tape → Decorative adhesive tape) noticeably reduced ambiguous matches.
Deterministic inventory checks: The direct inventory path consistently enforced stock sufficiency, minimum stock replenishment, supplier lead-time with +1 day shipping, and cash balance constraints, producing clear, actionable errors (e.g., Restock too late) instead of silent failures.
Controlled LLM usage: Language understanding (extraction) and customer-facing quote text leverage LLMs, while pricing math, unit conversions, and inventory logic are deterministic, improving reliability and traceability.
End-to-end accounting: Successful runs produce both stock_orders and sales transactions, enabling the financial report to reflect cash movements and inventory valuation after each request.

Future improvements

Quote reconciliation and guardrails: Add a deterministic “quote finalizer” that recalculates totals from per-item discounted prices and enforces exact equality with discounted_total_amount after rounding. Reject or auto-correct LLM outputs that violate the invariant.
Smarter matching and coverage: Augment the matcher with a lightweight embeddings index for recall on unusual phrasing, and automatically learn new aliases from clarified requests. Consider a “closest stocked alternative” policy that proposes substitutes when a canonical SKU isn’t in the current 80% inventory coverage.
Clarification UX and batching: When disambiguation is required, return a compact, multi-item clarification payload with top-3 choices per item, and accept a single user confirmation to proceed—reducing back-and-forth and improving throughput.
Evaluation harness: Add a small metrics report alongside test_results.csv that tracks match rate, clarification rate, order success rate, and average processing time per step. This makes improvements measurable over time.
