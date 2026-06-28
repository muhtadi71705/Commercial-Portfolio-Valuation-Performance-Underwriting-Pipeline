"""
CRE-Val  —  Interactive Investment Underwriting Dashboard

Run with:
    streamlit run src/ui/app.py

The dashboard accepts a rent roll CSV upload, routes it through the ETL
validation layer, runs the full 10-year pro forma → DCF → equity waterfall →
sensitivity matrix pipeline using sidebar assumption overrides, and displays
key metrics, a color-coded sensitivity table, and a PDF export button.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CRE-Val  |  Investment Underwriting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------

from src.analytics.simulator import run_macro_sensitivity_matrix, SensitivityMatrix
from src.config.loader import validate_batch, load_mapping_config
from src.config.schemas import LeaseRecord
from src.core.underwriting import (
    CommercialAsset,
    MultifamilyAsset,
    Asset,
    DCFResult,
    LeaseInfo,
    ProFormaYear,
    pro_forma_to_dataframe,
)
from src.core.waterfall import run_waterfall, WaterfallResult
from src.visual.custom_reports import ICMemorandum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORMATS     = ["yardi", "realpage", "mri", "appfolio", "entrata"]
_ASSET_CLASS = ["office", "retail", "industrial", "multifamily", "mixed_use", "hotel"]

_GREEN_HEX = "#15803D"
_AMBER_HEX = "#B45309"
_RED_HEX   = "#B91C1C"
_NAVY_HEX  = "#1B3A6B"
_LIGHT_HEX = "#F0F3F8"


# ---------------------------------------------------------------------------
# Helpers — pipeline
# ---------------------------------------------------------------------------

def _weighted_avg_rent(records: list[LeaseRecord]) -> float:
    total_sqft = sum(r.square_footage for r in records)
    if total_sqft == 0:
        return 0.0
    return sum(r.base_rent_psf * r.square_footage for r in records) / total_sqft


def _build_asset(
    records:           list[LeaseRecord],
    property_id:       str,
    purchase_price:    float,
    asset_class:       str,
    exit_cap_rate:     float,
    vacancy_rate:      float,
    rent_growth:       float,
    general_inflation: float,
    ltv:               float,
    interest_rate:     float,
    discount_rate:     float,
) -> Asset:
    total_sqft    = sum(r.square_footage for r in records)
    base_rent_psf = _weighted_avg_rent(records)

    shared = dict(
        property_id          = property_id,
        purchase_price       = purchase_price,
        total_sqft           = total_sqft,
        base_rent_psf        = base_rent_psf,
        rent_escalation_rate = rent_growth,
        general_inflation    = general_inflation,
        vacancy_rate         = vacancy_rate,
        credit_loss_rate     = 0.010,
        exit_cap_rate        = exit_cap_rate,
        loan_to_value        = ltv,
        debt_interest_rate   = interest_rate,
        amortization_years   = 30,
        discount_rate        = discount_rate,
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


def _build_report(
    records:  list[LeaseRecord],
    asset:    Asset,
    dcf:      DCFResult,
    asset_class: str,
) -> dict[str, Any]:
    """Build a minimal report dict compatible with ICMemorandum without a DB."""
    total_rent  = sum(r.base_rent_psf * r.square_footage for r in records)
    delinq_rent = sum(
        r.base_rent_psf * r.square_footage for r in records if r.is_delinquent
    )
    delinq_rate = delinq_rent / total_rent if total_rent > 0 else 0.0

    alerts: list[dict] = []
    if dcf.dscr_year_1 < 1.25:
        alerts.append({
            "severity": "CRITICAL",
            "message":  f"DSCR {dcf.dscr_year_1:.2f}x is below minimum 1.25x "
                        "threshold — lender covenant risk.",
        })
    elif dcf.dscr_year_1 < 1.40:
        alerts.append({
            "severity": "WARNING",
            "message":  f"DSCR {dcf.dscr_year_1:.2f}x is in warning band (1.25x–1.40x) — "
                        "monitor closely.",
        })

    if delinq_rate > 0.05:
        alerts.append({
            "severity": "WARNING",
            "message":  f"Tenant delinquency {delinq_rate:.1%} exceeds the 5% threshold — "
                        "contracted rent at risk.",
        })
    elif delinq_rate > 0.03:
        alerts.append({
            "severity": "INFO",
            "message":  f"Delinquency rate {delinq_rate:.1%} approaching warning threshold.",
        })

    if dcf.irr < 0.08:
        alerts.append({
            "severity": "CRITICAL",
            "message":  f"Levered IRR {dcf.irr:.2%} is below the 8% minimum hurdle rate.",
        })

    sev_rank = {"CRITICAL": 3, "WARNING": 2, "INFO": 1, "OK": 0}
    if not alerts:
        status = "OK"
    else:
        status = max(alerts, key=lambda a: sev_rank.get(a["severity"], 0))["severity"]

    return {
        "property_name":  asset.property_id,
        "asset_class":    asset_class,
        "overall_status": status,
        "alerts":         alerts,
        "lease_roll": [
            {
                "tenant_name":    r.tenant_name,
                "square_footage": r.square_footage,
                "base_rent_psf":  r.base_rent_psf,
                "annual_rent":    r.base_rent_psf * r.square_footage,
                "lease_end":      str(r.lease_end),
                "is_delinquent":  r.is_delinquent,
            }
            for r in sorted(records, key=lambda x: x.square_footage, reverse=True)
        ],
    }


def _generate_pdf_bytes(
    asset:       Asset,
    pf:          list[ProFormaYear],
    dcf:         DCFResult,
    report:      dict[str, Any],
    waterfall:   WaterfallResult,
    sensitivity: SensitivityMatrix,
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        ICMemorandum(
            asset=asset, pro_forma=pf, dcf=dcf,
            report=report, waterfall=waterfall,
            sensitivity=sensitivity,
        ).build(tmp_path)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers — display
# ---------------------------------------------------------------------------

def _irr_color(irr: float) -> str:
    if irr >= 0.12:
        return _GREEN_HEX
    if irr >= 0.08:
        return _AMBER_HEX
    return _RED_HEX


def _build_sensitivity_df(sm: SensitivityMatrix) -> pd.DataFrame:
    """Return a (vacancies × cap_rates) DataFrame of formatted combined IRR strings."""
    rows = {}
    for v in sm.vacancy_range:
        row = {}
        for c in sm.cap_rate_range:
            cell = sm.get(v, c)
            row[f"{c:.2%}"] = f"{cell.combined_irr:.2%}" if cell else "N/A"
        rows[f"{v:.1%}"] = row
    df = pd.DataFrame(rows).T
    df.index.name   = "Vacancy ↓  /  Exit Cap →"
    return df


def _build_lp_gp_df(sm: SensitivityMatrix) -> pd.DataFrame:
    """Return a multi-index DataFrame with LP IRR and GP IRR per cell."""
    records = []
    for v in sm.vacancy_range:
        for c in sm.cap_rate_range:
            cell = sm.get(v, c)
            if cell:
                records.append({
                    "Vacancy":  f"{v:.1%}",
                    "Exit Cap": f"{c:.2%}",
                    "Combined IRR": f"{cell.combined_irr:.2%}",
                    "LP IRR":   f"{cell.lp_irr:.2%}",
                    "GP IRR":   f"{cell.gp_irr:.2%}",
                })
    return pd.DataFrame(records).set_index(["Vacancy", "Exit Cap"])


def _color_irr_cell(val: str) -> str:
    """pandas Styler apply function — color IRR cells by hurdle zone."""
    try:
        irr = float(val.strip("%")) / 100
    except (ValueError, AttributeError):
        return ""
    if irr >= 0.12:
        return f"background-color: #D1FAE5; color: {_GREEN_HEX}; font-weight: bold"
    if irr >= 0.08:
        return f"background-color: #FEF3C7; color: {_AMBER_HEX}; font-weight: bold"
    return f"background-color: #FEE2E2; color: {_RED_HEX}; font-weight: bold"


def _style_sensitivity(df: pd.DataFrame):
    return (
        df.style
          .applymap(_color_irr_cell)
          .set_properties(**{
              "text-align": "center",
              "font-size":  "13px",
              "padding":    "6px 12px",
          })
          .set_table_styles([
              {"selector": "th", "props": [
                  ("background-color", _NAVY_HEX),
                  ("color", "white"),
                  ("font-weight", "bold"),
                  ("text-align", "center"),
                  ("padding", "6px 12px"),
              ]},
              {"selector": "th.index_name", "props": [
                  ("font-style", "italic"),
              ]},
          ])
    )


def _fmt_dollar(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.2f}M"
    return f"${v:,.0f}"


def _status_badge(status: str) -> str:
    colors = {
        "OK":       ("#D1FAE5", _GREEN_HEX),
        "INFO":     ("#DBEAFE", "#1D4ED8"),
        "WARNING":  ("#FEF3C7", _AMBER_HEX),
        "CRITICAL": ("#FEE2E2", _RED_HEX),
        "ERROR":    ("#FEE2E2", _RED_HEX),
    }
    bg, fg = colors.get(status, ("#F3F4F6", "#374151"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:12px;font-weight:bold;font-size:13px">{status}</span>'
    )


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Top banner strip */
        .cre-banner {
            background: #1B3A6B;
            color: #B89344;
            padding: 6px 18px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            border-radius: 4px;
            margin-bottom: 18px;
        }
        /* Section dividers */
        .section-title {
            font-size: 13px;
            font-weight: 700;
            color: #1B3A6B;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border-bottom: 2px solid #1B3A6B;
            padding-bottom: 4px;
            margin: 18px 0 10px;
        }
        /* Metric card override — bigger value */
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
        }
        /* Sidebar header */
        [data-testid="stSidebar"] .sidebar-section {
            font-size: 11px;
            font-weight: 700;
            color: #1B3A6B;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 14px 0 6px;
        }
        /* Alert pills */
        .alert-critical { color: #B91C1C; font-weight: 700; }
        .alert-warning  { color: #B45309; font-weight: 700; }
        .alert-info     { color: #1D4ED8; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict[str, Any]:
    """Render all sidebar controls and return the current assumption dict."""
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/CRE--Val-Underwriting%20Platform-1B3A6B"
            "?style=for-the-badge",
            use_container_width=True,
        )
        st.markdown("---")

        st.markdown('<div class="sidebar-section">Source Configuration</div>',
                    unsafe_allow_html=True)
        source_format = st.selectbox(
            "Rent Roll Format",
            _FORMATS,
            index=0,
            help="Select the property management system the CSV was exported from.",
        )
        asset_class = st.selectbox(
            "Asset Class",
            _ASSET_CLASS,
            index=0,
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-section">Acquisition</div>',
                    unsafe_allow_html=True)
        purchase_price_input = st.number_input(
            "Purchase Price ($)",
            min_value=500_000,
            max_value=5_000_000_000,
            value=45_000_000,
            step=500_000,
            format="%d",
            help="Leave as-is to use the auto-calculated price from the rent roll.",
        )
        ltv = st.slider(
            "Loan-to-Value (LTV)",
            min_value=0.40, max_value=0.85,
            value=0.65, step=0.01,
            format="%.0f%%",
            help="Percentage of purchase price financed with debt.",
        )
        interest_rate = st.slider(
            "Mortgage Interest Rate",
            min_value=0.03, max_value=0.12,
            value=0.065, step=0.0025,
            format="%.2f%%",
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-section">Market Assumptions</div>',
                    unsafe_allow_html=True)
        exit_cap_rate = st.slider(
            "Exit Cap Rate",
            min_value=0.03, max_value=0.12,
            value=0.055, step=0.0025,
            format="%.2f%%",
            help="Terminal capitalisation rate used to compute the reversion value.",
        )
        vacancy_rate = st.slider(
            "Market Vacancy Rate",
            min_value=0.00, max_value=0.35,
            value=0.05, step=0.005,
            format="%.1f%%",
        )
        rent_growth = st.slider(
            "Rent Growth Rate",
            min_value=0.00, max_value=0.08,
            value=0.03, step=0.0025,
            format="%.2f%%",
            help="Annual rent escalation applied to blended PSF and as fallback for "
                 "leases without an explicit escalation clause.",
        )
        general_inflation = st.slider(
            "General Inflation (CPI)",
            min_value=0.00, max_value=0.08,
            value=0.025, step=0.0025,
            format="%.2f%%",
            help="Drives CPI-Linked lease escalations and operating expense growth.",
        )
        discount_rate = st.slider(
            "Equity Discount Rate",
            min_value=0.05, max_value=0.20,
            value=0.09, step=0.005,
            format="%.1f%%",
            help="Levered equity hurdle rate used for NPV calculation.",
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#6B7280'>CRE-Val Platform  ·  Institutional "
            "Grade Underwriting</small>",
            unsafe_allow_html=True,
        )

    return dict(
        source_format      = source_format,
        asset_class        = asset_class,
        purchase_price     = float(purchase_price_input),
        ltv                = ltv,
        interest_rate      = interest_rate,
        exit_cap_rate      = exit_cap_rate,
        vacancy_rate       = vacancy_rate,
        rent_growth        = rent_growth,
        general_inflation  = general_inflation,
        discount_rate      = discount_rate,
    )


# ---------------------------------------------------------------------------
# Validation step (cached by file content)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _parse_and_validate(
    file_bytes: bytes,
    source_format: str,
    mapping_cfg_json: str,
) -> tuple[list[dict], list[dict]]:
    """
    Parse CSV bytes and validate through the ETL layer.

    Returns (valid_rows_as_dicts, error_dicts).
    Cached by (file_bytes, source_format) so re-running with different
    assumption sliders doesn't re-parse.
    """
    import json, io
    mapping_cfg = json.loads(mapping_cfg_json)
    df    = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
    rows  = df.to_dict(orient="records")
    batch = validate_batch(source_format, rows, mapping_cfg)

    valid_dicts = [
        {
            "property_id":     r.property_id,
            "tenant_name":     r.tenant_name,
            "square_footage":  r.square_footage,
            "base_rent_psf":   r.base_rent_psf,
            "lease_start":     str(r.lease_start),
            "lease_end":       str(r.lease_end),
            "is_delinquent":   r.is_delinquent,
            "escalation_type": r.escalation_type,
            "recovery_type":   r.recovery_type,
        }
        for r in batch.valid
    ]
    return valid_dicts, batch.errors


def _dicts_to_records(dicts: list[dict]) -> list[LeaseRecord]:
    from datetime import datetime
    records = []
    for d in dicts:
        records.append(LeaseRecord(
            property_id     = d["property_id"],
            tenant_name     = d["tenant_name"],
            square_footage  = d["square_footage"],
            base_rent_psf   = d["base_rent_psf"],
            lease_start     = datetime.fromisoformat(d["lease_start"]).date(),
            lease_end       = datetime.fromisoformat(d["lease_end"]).date(),
            is_delinquent   = d["is_delinquent"],
            escalation_type = d["escalation_type"],
            recovery_type   = d["recovery_type"],
        ))
    return records


# ---------------------------------------------------------------------------
# Metrics section
# ---------------------------------------------------------------------------

def _render_metrics(
    dcf:       DCFResult,
    waterfall: WaterfallResult,
    asset:     Asset,
) -> None:
    st.markdown('<div class="section-title">Key Performance Metrics</div>',
                unsafe_allow_html=True)

    # Row 1 — IRR + DSCR + EM
    c1, c2, c3, c4, c5 = st.columns(5)
    hurdle = asset.discount_rate

    irr_delta = dcf.irr - hurdle
    c1.metric(
        "Combined IRR",
        f"{dcf.irr:.2%}",
        delta=f"{irr_delta:+.2%} vs hurdle",
        delta_color="normal",
    )
    c2.metric("LP IRR  (90%)", f"{waterfall.lp_irr:.2%}")
    c3.metric("GP IRR  (10%)", f"{waterfall.gp_irr:.2%}")
    c4.metric(
        "DSCR  (Year 1)",
        f"{dcf.dscr_year_1:.2f}x",
        delta="above 1.25x" if dcf.dscr_year_1 >= 1.25 else "below 1.25x",
        delta_color="normal" if dcf.dscr_year_1 >= 1.25 else "inverse",
    )
    c5.metric(
        "Equity Multiple",
        f"{dcf.equity_multiple:.2f}x",
        delta="above 1.0x" if dcf.equity_multiple >= 1.0 else "below 1.0x",
        delta_color="normal" if dcf.equity_multiple >= 1.0 else "inverse",
    )

    # Row 2 — capital structure + NPV
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Purchase Price",    _fmt_dollar(asset.purchase_price))
    c7.metric("Equity Invested",   _fmt_dollar(dcf.equity_invested))
    c8.metric("NPV",               _fmt_dollar(dcf.npv),
              delta_color="normal" if dcf.npv >= 0 else "inverse")
    c9.metric("Going-In Cap Rate", f"{dcf.year_1_cap_rate:.2%}")
    c10.metric("Exit Value",       _fmt_dollar(dcf.exit_value))


# ---------------------------------------------------------------------------
# Sensitivity section
# ---------------------------------------------------------------------------

def _render_sensitivity(sm: SensitivityMatrix, asset: Asset) -> None:
    st.markdown('<div class="section-title">Macro Sensitivity Analysis</div>',
                unsafe_allow_html=True)

    st.caption(
        f"Combined IRR across {len(sm.vacancy_range)} vacancy scenarios × "
        f"{len(sm.cap_rate_range)} exit cap rate scenarios.  "
        f"Color thresholds: **green** ≥ 12%  ·  **amber** 8–12%  ·  **red** < 8%"
    )

    tab1, tab2 = st.tabs(["Combined IRR Heatmap", "LP / GP Split Detail"])

    with tab1:
        irr_df = _build_sensitivity_df(sm)
        st.dataframe(
            _style_sensitivity(irr_df),
            use_container_width=True,
            height=int(35 * (len(sm.vacancy_range) + 1) + 40),
        )

        # Base-case callout
        base_cell = sm.get(asset.vacancy_rate, asset.exit_cap_rate)
        if base_cell:
            bc_irr = base_cell.combined_irr
            col = _irr_color(bc_irr)
            st.markdown(
                f'<div style="background:{_LIGHT_HEX};border-left:4px solid {col};'
                f'padding:8px 14px;border-radius:4px;font-size:13px;">'
                f'<b>Base Case</b> — Vacancy {asset.vacancy_rate:.1%}  ·  '
                f'Exit Cap {asset.exit_cap_rate:.2%}  →  '
                f'<span style="color:{col};font-weight:bold">'
                f'Combined IRR {bc_irr:.2%}</span>  ·  '
                f'LP IRR {base_cell.lp_irr:.2%}  ·  '
                f'GP IRR {base_cell.gp_irr:.2%}</div>',
                unsafe_allow_html=True,
            )

    with tab2:
        lp_gp_df = _build_lp_gp_df(sm)
        st.dataframe(
            lp_gp_df.style
                .applymap(_color_irr_cell, subset=["Combined IRR"])
                .set_properties(**{"text-align": "center", "font-size": "13px"})
                .set_table_styles([{
                    "selector": "th",
                    "props": [
                        ("background-color", _NAVY_HEX),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                    ],
                }]),
            use_container_width=True,
            height=int(35 * len(lp_gp_df) + 65),
        )


# ---------------------------------------------------------------------------
# Pro forma section
# ---------------------------------------------------------------------------

def _render_pro_forma(pf: list[ProFormaYear]) -> None:
    st.markdown('<div class="section-title">10-Year Pro Forma</div>',
                unsafe_allow_html=True)

    df = pro_forma_to_dataframe(pf)

    money_cols = [c for c in df.columns if c != "Rent PSF ($)"]
    styled = (
        df.style
          .format({c: "${:,.0f}" for c in money_cols})
          .format({"Rent PSF ($)": "${:.2f}"})
          .applymap(
              lambda v: (
                  f"color:{_GREEN_HEX};font-weight:bold" if v >= 0
                  else f"color:{_RED_HEX};font-weight:bold"
              ),
              subset=["Levered Net Cash Flow ($)"],
          )
          .set_table_styles([
              {"selector": "th", "props": [
                  ("background-color", _NAVY_HEX),
                  ("color", "white"),
                  ("font-weight", "bold"),
                  ("text-align", "right"),
              ]},
              {"selector": "th.row_heading", "props": [
                  ("background-color", _NAVY_HEX),
                  ("color", "white"),
              ]},
          ])
          .set_properties(**{"text-align": "right", "font-size": "12px"})
    )
    st.dataframe(styled, use_container_width=True, height=425)


# ---------------------------------------------------------------------------
# Lease roll section
# ---------------------------------------------------------------------------

def _render_lease_roll(records: list[LeaseRecord]) -> None:
    st.markdown('<div class="section-title">Active Lease Roll</div>',
                unsafe_allow_html=True)

    rows = [
        {
            "Tenant":        r.tenant_name,
            "SF":            r.square_footage,
            "Rent PSF":      r.base_rent_psf,
            "Annual Rent":   r.base_rent_psf * r.square_footage,
            "Lease End":     str(r.lease_end),
            "Escalation":    r.escalation_type or "—",
            "Recovery":      r.recovery_type or "Gross",
            "Delinquent":    "YES" if r.is_delinquent else "—",
        }
        for r in sorted(records, key=lambda x: x.square_footage, reverse=True)
    ]
    df = pd.DataFrame(rows)

    def _flag_delinquent(row):
        if row["Delinquent"] == "YES":
            return [f"color:{_RED_HEX};font-weight:bold"] * len(row)
        return [""] * len(row)

    styled = (
        df.style
          .apply(_flag_delinquent, axis=1)
          .format({"SF": "{:,}", "Rent PSF": "${:.2f}", "Annual Rent": "${:,.0f}"})
          .set_table_styles([{
              "selector": "th",
              "props": [
                  ("background-color", _NAVY_HEX),
                  ("color", "white"),
                  ("font-weight", "bold"),
              ],
          }])
          .set_properties(**{"font-size": "12px"})
          .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True, height=int(35 * len(rows) + 65))


# ---------------------------------------------------------------------------
# Alerts section
# ---------------------------------------------------------------------------

def _render_alerts(report: dict[str, Any]) -> None:
    status  = report.get("overall_status", "OK")
    alerts  = report.get("alerts", [])

    st.markdown('<div class="section-title">Risk & Alert Status</div>',
                unsafe_allow_html=True)

    cols = st.columns([1, 4])
    cols[0].markdown(f"**Status**  {_status_badge(status)}",
                     unsafe_allow_html=True)

    if not alerts:
        cols[1].success("No active alerts — all metrics within underwriting thresholds.")
        return

    for a in alerts:
        sev = a["severity"]
        msg = a["message"]
        if sev == "CRITICAL":
            cols[1].error(f"**CRITICAL** — {msg}")
        elif sev == "WARNING":
            cols[1].warning(f"**WARNING** — {msg}")
        else:
            cols[1].info(f"**INFO** — {msg}")


# ---------------------------------------------------------------------------
# Export section
# ---------------------------------------------------------------------------

def _render_export(
    asset:       Asset,
    pf:          list[ProFormaYear],
    dcf:         DCFResult,
    report:      dict[str, Any],
    waterfall:   WaterfallResult,
    sensitivity: SensitivityMatrix,
) -> None:
    st.markdown('<div class="section-title">Export</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 5])
    with col1:
        with st.spinner("Generating PDF …"):
            pdf_bytes = _generate_pdf_bytes(
                asset, pf, dcf, report, waterfall, sensitivity
            )
        filename = f"{asset.property_id}_ICM_{date.today().strftime('%Y%m%d')}.pdf"
        st.download_button(
            label="⬇  Download Investment Committee Memo (PDF)",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )
    with col2:
        st.caption(
            f"The PDF includes: Cover & Executive Summary · 10-Year Pro Forma · "
            f"DCF Analysis + Equity Waterfall + Lease Roll · "
            f"Macro Sensitivity Matrix  ({len(pdf_bytes) / 1024:.0f} KB)"
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()

    # ── Page header ───────────────────────────────────────────────────
    st.markdown(
        '<div class="cre-banner">CRE-VAL  ·  INVESTMENT UNDERWRITING PLATFORM  ·  '
        'STRICTLY CONFIDENTIAL  -  FOR INTERNAL USE ONLY</div>',
        unsafe_allow_html=True,
    )

    st.title("Commercial Real Estate Underwriting Dashboard")
    st.caption(
        "Upload a validated rent roll CSV to run the full pro forma → DCF → "
        "waterfall → sensitivity pipeline with your sidebar assumptions."
    )

    # ── Sidebar ───────────────────────────────────────────────────────
    cfg = _render_sidebar()

    # ── File uploader ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Rent Roll Upload</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a rent roll CSV file",
        type=["csv"],
        help=(
            "Supported formats: Yardi, RealPage, MRI, AppFolio, Entrata. "
            "Select the matching format in the sidebar before uploading."
        ),
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info(
            "Upload a rent roll CSV using the widget above, then set your "
            "underwriting assumptions in the sidebar to run the analysis.",
            icon="📂",
        )
        _render_sample_schema()
        return

    # ── ETL parsing + validation (cached by file content) ────────────
    import json
    mapping_cfg = load_mapping_config()
    mapping_cfg_json = json.dumps(mapping_cfg)

    with st.spinner("Validating rent roll …"):
        valid_dicts, errors = _parse_and_validate(
            uploaded.read(), cfg["source_format"], mapping_cfg_json
        )

    if errors:
        with st.expander(f"⚠  {len(errors)} row(s) rejected during validation", expanded=False):
            for e in errors[:20]:
                st.text(f"Row {e['row_index']}: {e['error'][:120]}")
            if len(errors) > 20:
                st.text(f"… and {len(errors) - 20} more")

    if not valid_dicts:
        st.error(
            "All rows failed validation — check that you selected the correct "
            "source format in the sidebar.",
            icon="🚫",
        )
        return

    records = _dicts_to_records(valid_dicts)

    # ETL status bar
    total = len(valid_dicts) + len(errors)
    success_pct = len(valid_dicts) / total if total else 0.0
    ecol1, ecol2, ecol3 = st.columns(3)
    ecol1.metric("Rows Ingested",  total)
    ecol2.metric("Rows Valid",     len(valid_dicts))
    ecol3.metric("Validation Rate", f"{success_pct:.0%}",
                 delta_color="off")

    # Group by property_id — use the first if multiple present
    from collections import defaultdict
    grouped: dict[str, list[LeaseRecord]] = defaultdict(list)
    for r in records:
        grouped[r.property_id].append(r)

    if len(grouped) > 1:
        pid = st.selectbox(
            "Multiple properties detected — select one to underwrite:",
            sorted(grouped.keys()),
        )
    else:
        pid = next(iter(grouped))

    prop_records = grouped[pid]

    st.markdown(
        f"**{pid}** — {len(prop_records)} lease(s)  ·  "
        f"{sum(r.square_footage for r in prop_records):,} SF  ·  "
        f"Weighted avg. rent ${_weighted_avg_rent(prop_records):.2f} PSF",
    )
    st.divider()

    # ── Pipeline ─────────────────────────────────────────────────────
    with st.spinner("Running underwriting pipeline …"):
        try:
            asset = _build_asset(
                records           = prop_records,
                property_id       = pid,
                purchase_price    = cfg["purchase_price"],
                asset_class       = cfg["asset_class"],
                exit_cap_rate     = cfg["exit_cap_rate"],
                vacancy_rate      = cfg["vacancy_rate"],
                rent_growth       = cfg["rent_growth"],
                general_inflation = cfg["general_inflation"],
                ltv               = cfg["ltv"],
                interest_rate     = cfg["interest_rate"],
                discount_rate     = cfg["discount_rate"],
            )
            pf          = asset.generate_10_year_pro_forma()
            dcf         = asset.calculate_dcf(pf)
            waterfall   = run_waterfall(list(dcf.equity_cash_flows))
            sensitivity = run_macro_sensitivity_matrix(asset)
            report      = _build_report(prop_records, asset, dcf, cfg["asset_class"])

        except Exception as exc:
            st.error(f"Pipeline error: {exc}", icon="🚫")
            st.exception(exc)
            return

    # ── Render results ────────────────────────────────────────────────
    _render_alerts(report)
    st.divider()
    _render_metrics(dcf, waterfall, asset)
    st.divider()
    _render_sensitivity(sensitivity, asset)
    st.divider()
    _render_pro_forma(pf)
    st.divider()
    _render_lease_roll(prop_records)
    st.divider()
    _render_export(asset, pf, dcf, report, waterfall, sensitivity)


# ---------------------------------------------------------------------------
# Sample schema helper (shown before upload)
# ---------------------------------------------------------------------------

def _render_sample_schema() -> None:
    st.markdown('<div class="section-title">Expected CSV Schema</div>',
                unsafe_allow_html=True)
    st.caption(
        "Column names vary by source format. Below is the Yardi column mapping."
    )
    sample = pd.DataFrame([
        {
            "PropCode":       "BLDG-001",
            "TenantName":     "TENANT-A",
            "UnitSqFt":       "12000",
            "BaseRentPerSF":  "45.00",
            "LeaseFromDate":  "01/01/2024",
            "LeaseToDate":    "12/31/2029",
            "DelinquentFlag": "N",
            "Escalation_Type": "Fixed-3%",
            "Recovery_Type":  "NNN",
        }
    ])
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.caption(
        "For other formats (RealPage, MRI, AppFolio, Entrata) the column names "
        "differ — select the matching format in the sidebar."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
