# CRE-Val: Commercial Real Estate Portfolio Valuation & Performance Underwriting Pipeline

> An end-to-end institutional-grade underwriting system that ingests raw rent rolls from any major property management platform, validates every record, runs a full 10-year levered DCF, monitors covenant compliance in real time, and auto-generates a typeset Investment Committee Memorandum — all from a single CLI command.

---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [Key Capabilities & Architecture](#key-capabilities--architecture)
3. [The Automated 10-Year Pro Forma](#the-automated-10-year-pro-forma)
4. [Investment Committee Memorandum (ICM) Output Spec](#investment-committee-memorandum-icm-output-spec)
5. [CLI Usage Guide](#cli-usage-guide)
6. [Project Structure](#project-structure)
7. [Technology Stack](#technology-stack)
8. [Database Schema](#database-schema)
9. [Supported Source Formats](#supported-source-formats)
10. [DCF Engine & Returns Methodology](#dcf-engine--returns-methodology)
11. [Alert & Covenant Monitoring System](#alert--covenant-monitoring-system)
12. [Exit Codes](#exit-codes)

---

## Executive Overview

CRE-Val is a production-ready underwriting pipeline built for commercial real estate investment management. It bridges the gap between raw, format-inconsistent data exports from property management systems and the rigorous financial modeling required for institutional investment decisions.

The system handles the full stack of a deal workflow:

- **Ingestion** — Reads CSV/Excel rent rolls from Yardi, RealPage, MRI, AppFolio, or Entrata and normalises every field against a declarative JSON mapping registry.
- **Validation** — Enforces strict Pydantic v2 schemas with cross-field business rules (e.g., lease end must follow lease start) and surfaces per-record errors without failing the batch.
- **Persistence** — Appends validated records to a local SQLite database with full referential integrity, WAL journaling, and SQLAlchemy ORM coverage.
- **Underwriting** — Projects a 10-year levered cash flow waterfall (PGI → EGI → NOI → LNCF) using asset-class-specific expense logic for both NNN/commercial and multifamily assets.
- **DCF Analysis** — Computes NPV, unlevered and levered IRR, equity multiple, and terminal value via Newton-Raphson + bisection dual-solver against a Year 11 cap rate exit.
- **Covenant Monitoring** — Continuously checks DSCR and delinquency thresholds and surfaces structured alerts at three severity levels: INFO, WARNING, CRITICAL.
- **Reporting** — Produces a 3-page, chart-embedded, typeset Investment Committee Memorandum PDF ready for distribution.

---

## Key Capabilities & Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CRE-VAL PIPELINE                                   │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐    │
│  │  Raw Source  │────▶│  ETL + Pydantic│───▶│  SQLite (portfolio.db)   │    │
│  │  Rent Rolls  │     │  Validation  │     │  properties/leases/       │    │
│  │  (CSV/XLSX)  │     │  (schemas.py)│     │  expenses tables          │    │
│  └──────────────┘     └──────────────┘     └────────────┬─────────────┘    │
│                                                          │                   │
│  ┌───────────────────────────────────────────────────────▼───────────────┐  │
│  │                    Underwriting Engine (underwriting.py)              │  │
│  │   CommercialAsset (NNN)  │  MultifamilyAsset  │  abstract Asset base  │  │
│  │   10-year pro forma loop │  DCF / IRR solver  │  debt service layer   │  │
│  └───────────────────────────┬───────────────────────────────────────────┘  │
│                              │                                               │
│  ┌───────────────────────────▼───────────────────────────────────────────┐  │
│  │              Asset Manager + Covenant Monitor (asset_manager.py)      │  │
│  │   Budget-vs-Actual  │  DSCR Alerts  │  Delinquency Alerts             │  │
│  └───────────────────────────┬───────────────────────────────────────────┘  │
│                              │                                               │
│  ┌───────────────────────────▼───────────────────────────────────────────┐  │
│  │               IC Memorandum Generator (custom_reports.py)             │  │
│  │   Page 1: Cover + KPIs   │  Page 2: Pro Forma  │  Page 3: DCF + Rec  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Capabilities

| Capability | Detail |
|---|---|
| Multi-platform ingestion | Declarative JSON mapping supports Yardi, RealPage, MRI, AppFolio, Entrata — add new sources in one JSON block |
| Schema validation | Pydantic v2 with `PositiveInt`/`PositiveFloat` type aliases, cross-field date ordering validator, full error traceability |
| ORM persistence | SQLAlchemy 2.0 `Mapped`/`mapped_column` with `CheckConstraint`, composite indexes, and cascade deletes |
| Asset-class branching | Abstract `_compute_operating_expenses()` hook — NNN tenants absorb pass-throughs; multifamily uses ratio-based OpEx |
| IRR solver | Newton-Raphson primary, bisection fallback over [-50%, 500%], tolerance 1e-9 |
| Covenant alerts | Raise-and-catch exception pattern; structured dicts with severity levels consumable by any UI layer |
| PDF generation | fpdf2 + matplotlib; charts rendered to BytesIO buffers and embedded inline; Latin-1 safe string sanitizer |

---

## The Automated 10-Year Pro Forma

The pro forma engine compounds a full income/expense waterfall for each of the 10 hold years:

```
Year N Waterfall
─────────────────────────────────────────────────────────────
  Potential Gross Income (PGI)
    = base_rent_psf × total_sqft × (1 + rent_escalation_rate)^(N-1)

  Effective Gross Income (EGI)
    = PGI × (1 − vacancy_rate) × (1 − credit_loss_rate)

  Operating Expenses (OpEx)
    NNN / Commercial:
      mgmt_fee_rate × EGI
      + capex_reserve_psf × total_sqft × (1 + expense_growth_rate)^(N-1)
    Multifamily:
      operating_expense_ratio × Year-1 PGI × (1 + expense_growth_rate)^(N-1)

  Net Operating Income (NOI)
    = EGI − OpEx

  Annual Debt Service (ADS)
    = PMT(debt_interest_rate/12, amortization_years×12, −loan_amount) × 12
    [Fixed payment; computed once at origination]

  Levered Net Cash Flow (LNCF)
    = NOI − ADS

  Debt Service Coverage Ratio (DSCR)
    = NOI / ADS
─────────────────────────────────────────────────────────────
```

| Column | Description |
|---|---|
| `year` | Hold year 1–10 |
| `pgi` | Potential Gross Income |
| `vacancy_loss` | PGI × vacancy_rate |
| `credit_loss` | Post-vacancy income × credit_loss_rate |
| `egi` | Effective Gross Income |
| `operating_expenses` | Asset-class specific OpEx |
| `noi` | Net Operating Income |
| `debt_service` | Annual mortgage payment (constant) |
| `lncf` | Levered Net Cash Flow |
| `dscr` | Debt coverage multiple |

---

## Investment Committee Memorandum (ICM) Output Spec

The generated PDF is a 3-page letter-format document:

```
┌──────────────────────────────────────────────────────────┐
│  PAGE 1 — COVER & EXECUTIVE SUMMARY                      │
│  ─────────────────────────────────────────────────────   │
│  · Property name, asset class, prepared-by header        │
│  · Key metrics table: Purchase Price, LTV, NOI Yr1,      │
│    DSCR Yr1, Levered IRR, Equity Multiple, Exit Cap      │
│  · Alert summary box (if any DSCR/delinquency flags)     │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│  PAGE 2 — 10-YEAR PRO FORMA DETAIL                       │
│  ─────────────────────────────────────────────────────   │
│  · Full 10-row income/expense waterfall table            │
│  · NOI vs. Debt Service bar chart (matplotlib embedded)  │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│  PAGE 3 — DCF ANALYSIS & IC RECOMMENDATION               │
│  ─────────────────────────────────────────────────────   │
│  · DCF metrics: NPV, Levered IRR, Equity Multiple,       │
│    Terminal Value, Loan Balance, Net Exit Proceeds        │
│  · Cumulative levered cash flow waterfall chart          │
│  · Recommendation banner: HOLD or SELL                   │
│    (colour-coded green/red with rationale text)          │
└──────────────────────────────────────────────────────────┘
```

### Recommendation Logic

| Condition | Decision | Colour |
|---|---|---|
| IRR ≥ 12% AND Equity Multiple ≥ 1.5× | **HOLD** | Green |
| IRR < 8% OR Equity Multiple < 1.2× | **SELL** | Red |
| Otherwise | **HOLD** (marginal) | Green |

---

## CLI Usage Guide

### Basic Usage

```bash
# Run the full pipeline on a property directory
python main.py --property-dir data/raw_source/harbor_tower --format yardi

# Specify asset class explicitly
python main.py --property-dir data/raw_source/metro_flats \
               --format realpage \
               --asset-class multifamily

# Override purchase price and exit cap rate
python main.py --property-dir data/raw_source/harbor_tower \
               --format yardi \
               --purchase-price 12500000 \
               --exit-cap-rate 0.055

# Custom output directory for the PDF
python main.py --property-dir data/raw_source/harbor_tower \
               --format yardi \
               --output-dir reports/q2_2026
```

### All Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--property-dir` | Yes | — | Path to folder containing rent roll CSV/XLSX files |
| `--format` | Yes | — | Source system format: `yardi`, `realpage`, `mri`, `appfolio`, `entrata` |
| `--asset-class` | No | `office` | Asset class: `office`, `retail`, `multifamily`, `industrial`, `mixed_use`, `hotel`, `land` |
| `--purchase-price` | No | Auto-estimated | Acquisition price in dollars. If omitted, estimated as `(weighted_avg_rent × sqft) / exit_cap_rate` |
| `--exit-cap-rate` | No | `0.055` | Terminal exit cap rate applied to Year 11 NOI for terminal value |
| `--output-dir` | No | `reports/` | Directory where the ICM PDF will be written |
| `--db-path` | No | `data/secure_vault/portfolio.db` | Override the SQLite database path |
| `--discount-rate` | No | `0.090` | Equity discount rate for NPV calculation |
| `--ltv` | No | `0.65` | Loan-to-value ratio |

### Output

A successful run produces:

```
reports/
└── {PROPERTY_ID}_ICM_{YYYYMMDD}.pdf
```

And prints a summary to stdout:

```
[CRE-Val] Validated 5/5 leases (100.0% success rate)
[CRE-Val] Loaded 5 lease records into portfolio.db
[CRE-Val] Pro forma complete — Year 1 NOI: $687,000 | DSCR: 1.43x
[CRE-Val] Levered IRR: 14.2% | Equity Multiple: 1.87x | NPV: $1,240,000
[CRE-Val] ICM written → reports/BLDG-001_ICM_20260524.pdf
```

---

## Project Structure

```
cre-val/
├── main.py                          # CLI entry point — orchestrates full pipeline
├── requirements.txt                 # Python dependencies
├── data/
│   ├── raw_source/                  # Input rent rolls (CSV / XLSX)
│   │   └── {property_dir}/
│   │       └── rent_roll.csv
│   └── secure_vault/
│       └── portfolio.db             # SQLite database (auto-created)
├── reports/                         # Generated ICM PDFs (auto-created)
├── src/
│   ├── config/
│   │   ├── mapping_config.json      # Declarative source-system field maps
│   │   ├── schemas.py               # Pydantic v2 LeaseRecord model
│   │   └── loader.py                # ETL: parse → validate → BatchResult
│   ├── database/
│   │   └── db_manager.py            # SQLAlchemy ORM + load_dataframe_to_db()
│   ├── core/
│   │   └── underwriting.py          # Asset / CommercialAsset / MultifamilyAsset + DCF
│   ├── analytics/
│   │   └── asset_manager.py         # AssetManager, DSCRBreach, DelinquencySpike alerts
│   └── visual/
│       └── custom_reports.py        # ICMemorandum PDF builder (fpdf2 + matplotlib)
└── README.md
```

---

## Technology Stack

| Layer | Library | Version | Purpose |
|---|---|---|---|
| Validation | `pydantic` | v2.x | LeaseRecord schema enforcement |
| ORM / DB | `sqlalchemy` | 2.0+ | Table definitions, session management, FK enforcement |
| Data processing | `pandas` | 2.x | DataFrame transforms, batch loading |
| PDF generation | `fpdf2` | 2.x | Typeset multi-page PDF output |
| Charting | `matplotlib` | 3.x | NOI/DS bar chart and cumulative CF waterfall |
| Numeric | `numpy` | — | NPV/IRR polynomial evaluation |
| CLI | `argparse` | stdlib | Argument parsing |
| Database | `sqlite3` | stdlib | Embedded file-based persistence with WAL mode |

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pydantic sqlalchemy pandas fpdf2 matplotlib numpy openpyxl
```

---

## Database Schema

### `properties` table

| Column | Type | Constraints | Description |
|---|---|---|---|
| `property_id` | TEXT | PRIMARY KEY | Unique asset identifier |
| `property_name` | TEXT | NOT NULL | Human-readable property name |
| `address` | TEXT | — | Street address |
| `city` | TEXT | — | City |
| `state` | TEXT(2) | — | Two-letter state code |
| `asset_class` | TEXT | NOT NULL, CHECK | One of: office, retail, multifamily, industrial, mixed_use, hotel, land |
| `total_sqft` | INTEGER | — | Rentable square footage |
| `year_built` | INTEGER | — | Year of construction |
| `acquisition_date` | DATE | — | Purchase closing date |
| `acquisition_price` | REAL | — | Purchase price |
| `target_exit_cap_rate` | REAL | NOT NULL, > 0 | Target terminal cap rate |
| `created_at` / `updated_at` | DATETIME | — | Audit timestamps |

### `leases` table

| Column | Type | Constraints | Description |
|---|---|---|---|
| `lease_id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Surrogate key |
| `property_id` | TEXT | FK → properties, NOT NULL | Parent property |
| `tenant_name` | TEXT | NOT NULL | Tenant legal name |
| `square_footage` | INTEGER | NOT NULL, > 0 | Leased area |
| `base_rent_psf` | REAL | NOT NULL, > 0 | Annual base rent per square foot |
| `lease_start` | DATE | NOT NULL | Commencement date |
| `lease_end` | DATE | NOT NULL, > lease_start | Expiration date |
| `is_delinquent` | INTEGER (0/1) | NOT NULL | Current delinquency flag |

Indexes: `property_id`, `lease_start`, `is_delinquent`

### `expenses` table

| Column | Type | Constraints | Description |
|---|---|---|---|
| `expense_id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Surrogate key |
| `property_id` | TEXT | FK → properties, NOT NULL | Parent property |
| `expense_year` | INTEGER | NOT NULL | Fiscal year |
| `expense_month` | INTEGER | 1–12 or NULL | Month (NULL = annual aggregate) |
| `category` | TEXT | NOT NULL, CHECK | taxes, insurance, repairs_maintenance, management_fees, utilities, capital_expenditures, other |
| `amount` | REAL | NOT NULL, ≥ 0 | Dollar amount |
| `description` | TEXT | — | Free-text note |

Indexes: `property_id`, `(expense_year, expense_month)`, `category`

---

## Supported Source Formats

The field mapping registry at `src/config/mapping_config.json` currently supports five platforms. Adding a sixth requires only a new JSON block — no Python changes.

| Platform | `--format` key | Date Format | Notes |
|---|---|---|---|
| Yardi Voyager | `yardi` | `%m/%d/%Y` | `delinquent` field: `"Y"/"N"` |
| RealPage | `realpage` | `%Y-%m-%d` | `delinquency_status` field: `"DELINQUENT"/"CURRENT"` |
| MRI Software | `mri` | `%d-%b-%Y` | `payment_default` field: `1/0` integer |
| AppFolio | `appfolio` | `%m/%d/%Y` | `past_due` field: `"true"/"false"` string |
| Entrata | `entrata` | `%Y/%m/%d` | `status` field: `"Late"/"Current"` |

---

## DCF Engine & Returns Methodology

### Debt Service

Debt service is computed as a constant-payment mortgage (level-pay):

```
ADS = PMT(r/12, n×12, −LoanAmount) × 12
  where r = debt_interest_rate, n = amortization_years
```

### Terminal Value & Exit Proceeds

```
Terminal Value   = Year 11 NOI / exit_cap_rate
Loan Balance     = remaining principal after 10 years of payments
Net Exit Proceeds = Terminal Value − Loan Balance
```

### DCF Equity Cash Flow Stream

```
t=0:      −(Purchase Price − Loan Amount)      [equity invested]
t=1..9:   LNCF_t
t=10:     LNCF_10 + Net Exit Proceeds
```

### IRR Solver

The solver applies Newton-Raphson with analytical gradient:

```python
npv(r)  = sum(cf[t] / (1+r)^t for t in range(len(cf)))
npv'(r) = sum(-t * cf[t] / (1+r)^(t+1) for t in range(len(cf)))
r_new   = r - npv(r) / npv'(r)
```

If Newton-Raphson diverges, a bisection fallback scans `[-50%, +500%]` with tolerance `1e-9`.

### Returns Summary

| Metric | Definition |
|---|---|
| Levered IRR | IRR of equity cash flow stream above |
| NPV | `npv(discount_rate)` of equity cash flow stream |
| Equity Multiple | `(sum(LNCFs) + Net Exit Proceeds) / Equity Invested` |
| Terminal Value | Year 11 NOI / exit_cap_rate |

---

## Alert & Covenant Monitoring System

The `AssetManager` continuously monitors two covenant triggers:

### DSCR Alert Ladder

| DSCR Range | Severity | Alert Type |
|---|---|---|
| DSCR ≥ 1.40x | — | No alert |
| 1.25x ≤ DSCR < 1.40x | WARNING | Approaching covenant threshold |
| DSCR < 1.25x | CRITICAL | `DSCRBreach` exception raised and captured |

### Delinquency Alert Ladder

| Delinquency Rate | Severity | Alert Type |
|---|---|---|
| < 3.0% | — | No alert |
| 3.0% – 5.0% | INFO | Elevated delinquency trend |
| > 5.0% | CRITICAL | `DelinquencySpike` exception raised and captured |

### Alert Object Structure

Each alert in the `alerts` list of a property report has the following shape:

```json
{
  "type": "DSCR_BREACH",
  "severity": "CRITICAL",
  "message": "DSCR 1.18x is below covenant minimum 1.25x",
  "metric_value": 1.18,
  "threshold": 1.25
}
```

The `overall_status` field on the property report is the maximum severity across all alerts: `"OK"` → `"INFO"` → `"WARNING"` → `"CRITICAL"`.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Full success — all records validated, ICM generated |
| `1` | Partial success — some validation errors, pipeline continued with valid subset |
| `2` | Fatal failure — no valid records, or unrecoverable error |
