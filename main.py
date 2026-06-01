#!/usr/bin/env python3
"""
CRE-Val  —  Command-line underwriting and reporting pipeline.

Usage
-----
    python main.py \\
        --property-dir  data/raw_source/harbor_tower \\
        --format        yardi \\
        --asset-class   office \\
        --purchase-price 42000000 \\
        --exit-cap-rate  0.055 \\
        --ltv            0.65 \\
        [--output-dir    reports] \\
        [--db            data/secure_vault/portfolio.db] \\
        [--verbose]

The pipeline:
  1. Discover all CSV / Excel files in --property-dir
  2. Load and validate each row through the mapping + Pydantic layer
  3. Upsert properties and load validated leases into the SQLite database
  4. Build Asset objects (CommercialAsset or MultifamilyAsset) from DB metadata
     and CLI assumptions
  5. Run the 10-year pro forma and DCF for each property
  6. Produce an AssetManager performance report (budget vs. actual, alerts)
  7. Render a 3-page Investment Committee Memorandum PDF per property
  8. Print a results summary table to stdout
  9. Exit 0 if all properties are OK/INFO/WARNING; exit 1 if any are CRITICAL
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import copy

import pandas as pd

from src.analytics.asset_manager import AssetManager
from src.config.loader import BatchResult, load_mapping_config, validate_batch
from src.config.schemas import LeaseRecord
from src.core.underwriting import CommercialAsset, MultifamilyAsset, Asset, LeaseInfo
from src.database.db_manager import (
    Lease,
    Property,
    _DB_PATH,
    get_session_factory,
    init_db,
    load_dataframe_to_db,
)
from src.visual.custom_reports import ICMemorandum


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            CRE-Val  —  end-to-end commercial real estate underwriting pipeline.
            Extracts, validates, underwrites, and reports on rent roll data.
        """),
    )

    # Required
    p.add_argument(
        "--property-dir", required=True, metavar="PATH",
        help="Directory containing source CSV/Excel rent roll files.",
    )
    p.add_argument(
        "--format", required=True,
        choices=["yardi", "realpage", "mri", "appfolio", "entrata"],
        help="Source system format to apply during field mapping.",
    )

    # Underwriting assumptions (applied to all properties in the run unless
    # a property already exists in the database with its own metadata)
    p.add_argument("--asset-class",    default="office",
                   choices=["office", "retail", "industrial", "multifamily",
                             "mixed_use", "hotel"],
                   help="Asset class (default: office).")
    p.add_argument("--purchase-price", type=float, default=None,
                   metavar="DOLLARS",
                   help="Acquisition price in dollars (required for new properties).")
    p.add_argument("--exit-cap-rate",  type=float, default=0.055,
                   help="Terminal exit cap rate assumption (default: 0.055).")
    p.add_argument("--ltv",            type=float, default=0.65,
                   help="Loan-to-value ratio (default: 0.65).")
    p.add_argument("--interest-rate",  type=float, default=0.065,
                   help="Mortgage interest rate (default: 0.065).")
    p.add_argument("--discount-rate",  type=float, default=0.09,
                   help="Equity discount rate for NPV (default: 0.09).")
    p.add_argument("--vacancy-rate",   type=float, default=0.05,
                   help="Market vacancy assumption (default: 0.05).")
    p.add_argument("--rent-growth",    type=float, default=0.03,
                   help="Annual rent escalation rate (default: 0.03).")

    # I/O
    p.add_argument("--output-dir",  default="reports",  metavar="PATH",
                   help="Directory for PDF output (default: reports/).")
    p.add_argument("--db",          default=str(_DB_PATH), metavar="PATH",
                   help="SQLite database path.")
    p.add_argument("--verbose",     action="store_true",
                   help="Print per-row validation detail.")

    return p


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_files(directory: str | Path) -> list[Path]:
    """Return all CSV and Excel files under *directory* (non-recursive)."""
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"--property-dir not found: {base}")
    exts  = {".csv", ".xlsx", ".xls"}
    files = [f for f in sorted(base.iterdir()) if f.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(
            f"No CSV or Excel files found in {base}. "
            f"Supported extensions: {sorted(exts)}"
        )
    return files


# ---------------------------------------------------------------------------
# Source file reader
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> list[dict[str, Any]]:
    """Load a CSV or Excel file into a list of raw row dicts."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Property upsert helpers
# ---------------------------------------------------------------------------


def _weighted_avg_rent(records: list[LeaseRecord]) -> float:
    total_sqft = sum(r.square_footage for r in records)
    if total_sqft == 0:
        return 0.0
    return sum(r.base_rent_psf * r.square_footage for r in records) / total_sqft


def _upsert_property(
    property_id: str,
    records:     list[LeaseRecord],
    session_factory,
    args:        argparse.Namespace,
) -> None:
    """
    Insert a property row if it doesn't exist.  Skips silently if the property
    is already in the database so re-runs are idempotent.
    """
    with session_factory() as s:
        existing = s.get(Property, property_id)
        if existing:
            return

        purchase_price = (
            args.purchase_price
            if args.purchase_price
            else _weighted_avg_rent(records) * sum(r.square_footage for r in records) / args.exit_cap_rate
        )

        s.add(Property(
            property_id          = property_id,
            property_name        = property_id,   # caller can update manually
            asset_class          = args.asset_class,
            total_sqft           = sum(r.square_footage for r in records),
            acquisition_price    = round(purchase_price, 2),
            target_exit_cap_rate = args.exit_cap_rate,
        ))
        s.commit()


def _load_leases_to_db(
    property_id: str,
    records:     list[LeaseRecord],
    db_path:     Path,
) -> int:
    """Convert validated LeaseRecord objects to a DataFrame and append to DB."""
    rows = [
        {
            "property_id":    r.property_id,
            "tenant_name":    r.tenant_name,
            "square_footage": r.square_footage,
            "base_rent_psf":  r.base_rent_psf,
            "lease_start":    str(r.lease_start),
            "lease_end":      str(r.lease_end),
            "is_delinquent":  int(r.is_delinquent),
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    return load_dataframe_to_db(df, "leases", db_path)


# ---------------------------------------------------------------------------
# Asset builder
# ---------------------------------------------------------------------------


def _build_asset(
    property_id:     str,
    records:         list[LeaseRecord],
    session_factory,
    args:            argparse.Namespace,
) -> Asset:
    """
    Construct a CommercialAsset or MultifamilyAsset from:
      • DB metadata  (purchase_price, total_sqft, exit_cap_rate, asset_class)
      • Rent roll    (weighted-average base_rent_psf derived from validated leases)
      • CLI flags    (ltv, interest_rate, discount_rate, vacancy_rate, rent_growth)
    """
    with session_factory() as s:
        prop = s.get(Property, property_id)

    if prop is None:
        raise RuntimeError(f"Property '{property_id}' not found in database — upsert failed.")

    purchase_price = prop.acquisition_price or (
        args.purchase_price
        or _weighted_avg_rent(records)
        * (prop.total_sqft or sum(r.square_footage for r in records))
        / args.exit_cap_rate
    )

    total_sqft    = prop.total_sqft or sum(r.square_footage for r in records)
    base_rent_psf = _weighted_avg_rent(records)
    asset_class   = prop.asset_class or args.asset_class
    exit_cap      = prop.target_exit_cap_rate or args.exit_cap_rate

    shared = dict(
        property_id          = property_id,
        purchase_price       = purchase_price,
        total_sqft           = total_sqft,
        base_rent_psf        = base_rent_psf,
        rent_escalation_rate = args.rent_growth,
        general_inflation    = args.rent_growth,
        vacancy_rate         = args.vacancy_rate,
        credit_loss_rate     = 0.010,
        exit_cap_rate        = exit_cap,
        loan_to_value        = args.ltv,
        debt_interest_rate   = args.interest_rate,
        amortization_years   = 30,
        discount_rate        = args.discount_rate,
        lease_schedule       = [
            LeaseInfo(
                tenant_name     = r.tenant_name,
                square_footage  = r.square_footage,
                base_rent_psf   = r.base_rent_psf,
                escalation_type = r.escalation_type,
                recovery_type   = r.recovery_type,
            )
            for r in records
        ],
    )

    if asset_class == "multifamily":
        return MultifamilyAsset(**shared, operating_expense_ratio=0.42)
    return CommercialAsset(**shared)


# ---------------------------------------------------------------------------
# Printer helpers
# ---------------------------------------------------------------------------

_SEV_ICON = {
    "OK":       "✓",
    "INFO":     "ℹ",
    "WARNING":  "▲",
    "CRITICAL": "✖",
    "ERROR":    "?",
}

_SEV_LABEL = {
    "OK":       "\033[32mOK\033[0m",
    "INFO":     "\033[36mINFO\033[0m",
    "WARNING":  "\033[33mWARNING\033[0m",
    "CRITICAL": "\033[31mCRITICAL\033[0m",
    "ERROR":    "\033[31mERROR\033[0m",
}


def _print_header() -> None:
    print()
    print("  CRE-Val  —  Investment Underwriting Pipeline")
    print("  " + "─" * 60)


def _print_summary_table(results: list[dict]) -> None:
    print()
    w_id, w_name, w_irr, w_dscr, w_st = 14, 24, 10, 8, 10
    header = (
        f"  {'Property ID':<{w_id}}  {'Name':<{w_name}}"
        f"  {'IRR':>{w_irr}}  {'DSCR Y1':>{w_dscr}}  {'Status':<{w_st}}"
    )
    print(header)
    print("  " + "─" * (w_id + w_name + w_irr + w_dscr + w_st + 12))
    for r in results:
        sev = r["status"]
        print(
            f"  {r['property_id']:<{w_id}}"
            f"  {(r['name'] or '')[:w_name]:<{w_name}}"
            f"  {r['irr']:>{w_irr}}"
            f"  {r['dscr']:>{w_dscr}}"
            f"  {_SEV_LABEL.get(sev, sev):<{w_st}}"
        )
    print()


# ---------------------------------------------------------------------------
# Per-property config resolver
# ---------------------------------------------------------------------------

_DEFAULT_ASSUMPTIONS: dict[str, float] = {
    "rent_growth":   0.025,
    "vacancy_rate":  0.050,
    "exit_cap_rate": 0.0625,
    "discount_rate": 0.090,
    "ltv":           0.65,
    "interest_rate": 0.065,
}


def _resolve_prop_cfg(
    prop_id:     str,
    mapping_cfg: dict,
    args:        argparse.Namespace,
) -> tuple[str, dict]:
    """
    Return (source_format, assumptions) for *prop_id*.

    Checks mapping_cfg for a property-level entry keyed by prop_id.
    Falls back to the --format CLI flag and sensible market defaults when
    no property-specific block exists so the pipeline never KeyErrors on
    an unknown directory name.
    """
    prop_cfg = mapping_cfg.get(prop_id, {})
    fmt = prop_cfg.get("source_format", args.format)

    assumptions = {
        k: prop_cfg.get(k, getattr(args, k, _DEFAULT_ASSUMPTIONS[k]))
        for k in _DEFAULT_ASSUMPTIONS
    }
    return fmt, assumptions


def _apply_assumptions(
    args:      argparse.Namespace,
    overrides: dict,
) -> argparse.Namespace:
    """Return a shallow copy of *args* with *overrides* applied."""
    patched = copy.copy(args)
    for k, v in overrides.items():
        setattr(patched, k, v)
    return patched


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> int:
    """
    Execute the full extraction → validation → underwriting → reporting loop.

    Returns
    -------
    0   all properties OK / INFO / WARNING
    1   at least one CRITICAL alert (lender covenant likely breached)
    2   pipeline error (bad arguments, missing files, DB failure)
    """
    _print_header()

    # ── 1. Resolve paths ──────────────────────────────────────────────
    db_path    = Path(args.db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_files = _discover_files(args.property_dir)
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        return 2

    print(f"  Source format   : {args.format.upper()}")
    print(f"  Files found     : {len(source_files)}")
    for f in source_files:
        print(f"    • {f.name}")
    print()

    # ── 2. Load mapping config + init database ─────────────────────────
    mapping_cfg = load_mapping_config()
    engine      = init_db(db_path)
    session_fac = get_session_factory(engine)
    print(f"  Database        : {db_path}")
    print()

    # ── 3. Ingest + validate all source files ──────────────────────────
    all_rows: list[dict] = []
    for f in source_files:
        try:
            rows = _read_file(f)
            all_rows.extend(rows)
            if args.verbose:
                print(f"  Loaded {len(rows):>4} rows from {f.name}")
        except Exception as exc:
            print(f"  WARNING: Could not read {f.name} — {exc}")

    if not all_rows:
        print("  ERROR: No rows loaded from any source file.")
        return 2

    batch: BatchResult = validate_batch(args.format, all_rows, mapping_cfg)
    print(f"  Rows ingested   : {len(all_rows)}")
    print(f"  Rows valid      : {len(batch.valid)}")
    print(f"  Rows rejected   : {batch.error_count}  "
          f"(success rate {batch.success_rate:.0%})")

    if args.verbose and batch.errors:
        print()
        print("  Validation errors:")
        for e in batch.errors[:10]:
            print(f"    Row {e['row_index']}: {e['error'][:100]}")
        if len(batch.errors) > 10:
            print(f"    … and {len(batch.errors) - 10} more")

    if not batch.valid:
        print("  ERROR: All rows failed validation — nothing to underwrite.")
        return 2

    # ── 4. Group validated records by property_id ──────────────────────
    from collections import defaultdict
    grouped: dict[str, list[LeaseRecord]] = defaultdict(list)
    for rec in batch.valid:
        grouped[rec.property_id].append(rec)

    print()
    print(f"  Properties found: {len(grouped)}")
    for pid, recs in grouped.items():
        print(f"    • {pid}  ({len(recs)} leases)")

    # Resolve per-property format + underwriting assumptions once.
    # Falls back to CLI flags and _DEFAULT_ASSUMPTIONS for any property
    # not explicitly listed in mapping_config.json.
    prop_configs: dict[str, tuple[str, dict]] = {
        pid: _resolve_prop_cfg(pid, mapping_cfg, args)
        for pid in grouped
    }

    # ── 5. Upsert properties + load leases ────────────────────────────
    print()
    print("  Loading data to database …")
    for pid, records in grouped.items():
        try:
            _, assumptions = prop_configs[pid]
            prop_args = _apply_assumptions(args, assumptions)
            _upsert_property(pid, records, session_fac, prop_args)
            n = _load_leases_to_db(pid, records, db_path)
            if args.verbose:
                print(f"    {pid}: {n} lease rows written")
        except Exception as exc:
            print(f"  WARNING: DB load failed for {pid} — {exc}")

    # ── 6. Build Asset objects ────────────────────────────────────────
    print("  Building underwriting models …")
    assets: dict[str, Asset] = {}
    for pid, records in grouped.items():
        try:
            _, assumptions = prop_configs[pid]
            prop_args = _apply_assumptions(args, assumptions)
            assets[pid] = _build_asset(pid, records, session_fac, prop_args)
        except Exception as exc:
            print(f"  WARNING: Could not build Asset for {pid} — {exc}")

    if not assets:
        print("  ERROR: No assets could be constructed.")
        return 2

    # ── 7. Underwrite + report + PDF ──────────────────────────────────
    manager = AssetManager(assets, db_path=db_path)
    results: list[dict] = []
    exit_code = 0

    print()
    print("  Generating reports …")
    print()

    for pid, asset in assets.items():
        try:
            pf     = asset.generate_10_year_pro_forma()
            dcf    = asset.calculate_dcf(pf)
            report = manager.generate_property_report(pid, as_of=date.today())

            pdf_path = output_dir / f"{pid}_ICM_{date.today().strftime('%Y%m%d')}.pdf"
            memo = ICMemorandum(
                asset         = asset,
                pro_forma     = pf,
                dcf           = dcf,
                report        = report,
                prepared_by   = "CRE-Val Platform",
            )
            memo.build(pdf_path)

            status = report.get("overall_status", "OK")
            if status == "CRITICAL":
                exit_code = 1

            results.append({
                "property_id": pid,
                "name":        report.get("property_name"),
                "irr":         f"{dcf.irr:.2%}",
                "dscr":        f"{dcf.dscr_year_1:.2f}x",
                "status":      status,
                "pdf":         str(pdf_path),
            })

            alerts = report.get("alerts", [])
            icon   = _SEV_ICON.get(status, "?")
            print(f"  {icon} {pid}")
            print(f"      IRR {dcf.irr:.2%}  ·  DSCR {dcf.dscr_year_1:.2f}x  ·  "
                  f"NPV ${dcf.npv:,.0f}  ·  EM {dcf.equity_multiple:.2f}x")
            if alerts:
                for a in alerts:
                    sev = a["severity"]
                    print(f"      [{sev}] {a['message'][:80]}")
            print(f"      PDF → {pdf_path}")
            print()

        except Exception as exc:
            print(f"  ERROR processing {pid}: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results.append({
                "property_id": pid, "name": None,
                "irr": "N/A", "dscr": "N/A",
                "status": "ERROR", "pdf": "",
            })

    # ── 8. Summary table ──────────────────────────────────────────────
    _print_summary_table(results)

    crit = sum(1 for r in results if r["status"] == "CRITICAL")
    ok   = sum(1 for r in results if r["status"] in ("OK", "INFO", "WARNING"))
    print(f"  {ok} propert{'y' if ok == 1 else 'ies'} within thresholds  ·  "
          f"{crit} CRITICAL alert{'s' if crit != 1 else ''}")
    print(f"  PDFs saved to: {output_dir.resolve()}")
    print()

    return exit_code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
