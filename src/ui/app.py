"""
CRE-Val  —  Enterprise Investment Underwriting Dashboard

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config  (must be the first Streamlit call in the module)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CRE-Val  |  Investment Underwriting",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**CRE-Val** — Institutional-grade commercial real estate underwriting "
            "platform. Proprietary and confidential."
        ),
        "Get Help": None,
        "Report a Bug": None,
    },
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
# Brand palette
# ---------------------------------------------------------------------------

_NAVY       = "#1B3A6B"
_NAVY_LIGHT = "#2D5A9E"
_GOLD       = "#B89344"
_GREEN      = "#15803D"
_GREEN_BG   = "#D1FAE5"
_AMBER      = "#B45309"
_AMBER_BG   = "#FEF3C7"
_RED        = "#B91C1C"
_RED_BG     = "#FEE2E2"
_SLATE      = "#64748B"
_LIGHT      = "#F0F3F8"
_BORDER     = "#E2E8F0"
_WHITE      = "#FFFFFF"

_FORMATS     = ["yardi", "realpage", "mri", "appfolio", "entrata"]
_ASSET_CLASS = ["office", "retail", "industrial", "multifamily", "mixed_use", "hotel"]


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ── Global typography ───────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* ── Confidential banner ─────────────────────────────────────── */
        .conf-banner {{
            background: {_NAVY};
            color: {_GOLD};
            padding: 7px 20px;
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            border-radius: 6px;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* ── Page title area ─────────────────────────────────────────── */
        .page-eyebrow {{
            font-size: 11px;
            font-weight: 700;
            color: {_SLATE};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 2px;
        }}
        .page-title {{
            font-size: 28px;
            font-weight: 800;
            color: {_NAVY};
            line-height: 1.2;
            margin-bottom: 4px;
        }}
        .page-subtitle {{
            font-size: 13px;
            color: {_SLATE};
            margin-bottom: 20px;
        }}

        /* ── Section headers ─────────────────────────────────────────── */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 24px 0 12px;
        }}
        .section-header-line {{
            flex: 1;
            height: 1px;
            background: {_BORDER};
        }}
        .section-header-label {{
            font-size: 11px;
            font-weight: 700;
            color: {_SLATE};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            white-space: nowrap;
        }}

        /* ── KPI cards ───────────────────────────────────────────────── */
        .kpi-card {{
            background: {_WHITE};
            border: 1px solid {_BORDER};
            border-radius: 10px;
            padding: 16px 18px 14px;
            height: 100%;
            transition: box-shadow 0.15s ease;
        }}
        .kpi-card:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }}
        .kpi-eyebrow {{
            font-size: 10.5px;
            font-weight: 700;
            color: {_SLATE};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 800;
            color: #111827;
            line-height: 1.15;
            margin-bottom: 4px;
        }}
        .kpi-delta {{
            font-size: 11.5px;
            font-weight: 600;
            margin-top: 2px;
        }}
        .kpi-sub {{
            font-size: 11px;
            color: {_SLATE};
            margin-top: 4px;
        }}

        /* ── Risk banner (delinquency alert) ─────────────────────────── */
        .risk-banner {{
            border-radius: 8px;
            padding: 14px 18px;
            margin: 12px 0;
            display: flex;
            gap: 14px;
            align-items: flex-start;
        }}
        .risk-banner-icon {{ font-size: 20px; flex-shrink: 0; margin-top: 1px; }}
        .risk-banner-title {{
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .risk-banner-body {{
            font-size: 12.5px;
            line-height: 1.55;
        }}
        .risk-banner--critical {{
            background: {_RED_BG};
            border: 1px solid #FECACA;
            border-left: 4px solid {_RED};
        }}
        .risk-banner--critical .risk-banner-title {{ color: {_RED}; }}
        .risk-banner--critical .risk-banner-body  {{ color: #7F1D1D; }}
        .risk-banner--warning {{
            background: {_AMBER_BG};
            border: 1px solid #FDE68A;
            border-left: 4px solid {_AMBER};
        }}
        .risk-banner--warning .risk-banner-title {{ color: {_AMBER}; }}
        .risk-banner--warning .risk-banner-body  {{ color: #78350F; }}
        .risk-banner--delinquency {{
            background: #FFF7ED;
            border: 1px solid #FED7AA;
            border-left: 4px solid #EA580C;
        }}
        .risk-banner--delinquency .risk-banner-title {{ color: #C2410C; }}
        .risk-banner--delinquency .risk-banner-body  {{ color: #7C2D12; }}
        .risk-banner--ok {{
            background: {_GREEN_BG};
            border: 1px solid #A7F3D0;
            border-left: 4px solid {_GREEN};
        }}
        .risk-banner--ok .risk-banner-title {{ color: {_GREEN}; }}
        .risk-banner--ok .risk-banner-body  {{ color: #14532D; }}

        /* ── Landing page hero ───────────────────────────────────────── */
        .landing-hero {{
            background: linear-gradient(135deg, {_NAVY} 0%, {_NAVY_LIGHT} 100%);
            border-radius: 12px;
            padding: 36px 40px;
            color: white;
            margin: 8px 0 24px;
        }}
        .landing-hero h2 {{
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 8px;
            color: white;
        }}
        .landing-hero p {{
            font-size: 13.5px;
            opacity: 0.85;
            line-height: 1.6;
            max-width: 520px;
        }}
        .feature-pill {{
            display: inline-block;
            background: rgba(255,255,255,0.12);
            color: {_GOLD};
            border: 1px solid rgba(184,147,68,0.4);
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.06em;
            margin: 3px 2px;
        }}

        /* ── ETL status strip ────────────────────────────────────────── */
        .etl-strip {{
            background: {_LIGHT};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            padding: 10px 16px;
            display: flex;
            gap: 24px;
            align-items: center;
            margin: 12px 0;
            font-size: 13px;
        }}
        .etl-stat {{ display: flex; gap: 6px; align-items: center; }}
        .etl-stat-label {{ color: {_SLATE}; font-weight: 500; }}
        .etl-stat-value {{ font-weight: 700; color: {_NAVY}; }}

        /* ── Property identity bar ───────────────────────────────────── */
        .prop-bar {{
            background: {_NAVY};
            color: white;
            border-radius: 8px;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 14px 0;
        }}
        .prop-bar-name {{
            font-size: 16px;
            font-weight: 700;
        }}
        .prop-bar-meta {{
            font-size: 12px;
            opacity: 0.75;
            margin-top: 2px;
        }}
        .prop-bar-right {{
            text-align: right;
            font-size: 12.5px;
            opacity: 0.85;
        }}

        /* ── Tab styling ─────────────────────────────────────────────── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 2px solid {_BORDER};
        }}
        [data-testid="stTabs"] [data-baseweb="tab"] {{
            font-size: 13px;
            font-weight: 600;
            padding: 8px 18px;
            border-radius: 6px 6px 0 0;
        }}

        /* ── Sidebar styling ─────────────────────────────────────────── */
        [data-testid="stSidebar"] {{
            border-right: 1px solid {_BORDER};
        }}
        .sb-section {{
            font-size: 10.5px;
            font-weight: 700;
            color: {_SLATE};
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 14px 0 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .sb-section::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: {_BORDER};
        }}

        /* ── Dataframe header override ───────────────────────────────── */
        [data-testid="stDataFrame"] thead th {{
            background-color: {_NAVY} !important;
            color: white !important;
            font-weight: 700 !important;
        }}

        /* ── Download button ─────────────────────────────────────────── */
        [data-testid="stDownloadButton"] > button {{
            background: {_NAVY};
            color: white;
            border: none;
            font-weight: 600;
            letter-spacing: 0.03em;
        }}
        [data-testid="stDownloadButton"] > button:hover {{
            background: {_NAVY_LIGHT};
            color: white;
        }}

        /* ── Delinquent row badge ─────────────────────────────────────── */
        .delinq-badge {{
            background: {_RED_BG};
            color: {_RED};
            font-weight: 700;
            font-size: 10px;
            padding: 1px 7px;
            border-radius: 10px;
            letter-spacing: 0.05em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sb_section(label: str) -> None:
    st.sidebar.markdown(
        f'<div class="sb-section">{label}</div>', unsafe_allow_html=True
    )


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        # Logo / wordmark
        st.markdown(
            f"""
            <div style="padding:12px 0 8px;">
                <div style="font-size:20px;font-weight:800;color:{_NAVY};
                            letter-spacing:-0.02em;">CRE-Val</div>
                <div style="font-size:11px;color:{_SLATE};font-weight:500;
                            letter-spacing:0.04em;">Investment Underwriting Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Data Source ────────────────────────────────────────────────
        _sb_section("Data Source")
        source_format = st.selectbox(
            "Rent Roll Format",
            _FORMATS,
            format_func=str.title,
            help=(
                "Select the property management system your CSV was exported from. "
                "This controls the field-name mapping applied during ETL validation. "
                "Supported: Yardi, RealPage, MRI, AppFolio, Entrata."
            ),
        )
        asset_class = st.selectbox(
            "Asset Class",
            _ASSET_CLASS,
            format_func=lambda x: x.replace("_", " ").title(),
            help=(
                "Determines the operating expense model. Commercial classes "
                "(Office, Retail, Industrial) use the NNN/Base-Year-Stop expense "
                "recovery engine. Multifamily uses an expense-ratio model."
            ),
        )

        # ── Deal Terms ─────────────────────────────────────────────────
        _sb_section("Deal Terms")
        purchase_price_input = st.number_input(
            "Purchase Price ($)",
            min_value=500_000,
            max_value=5_000_000_000,
            value=45_000_000,
            step=500_000,
            format="%d",
            help=(
                "Total acquisition cost in dollars. Used as the basis for LTV "
                "calculation, going-in cap rate, and debt yield. If left at the "
                "default, this value is used as-is — it does not auto-calculate "
                "from the rent roll in the dashboard."
            ),
        )
        ltv = st.slider(
            "Loan-to-Value (LTV)",
            min_value=0.40, max_value=0.85,
            value=0.65, step=0.01,
            format="%.0f%%",
            help=(
                "Percentage of the acquisition price financed by senior debt. "
                "Higher LTV amplifies equity returns in outperformance scenarios "
                "but compresses DSCR coverage and increases lender covenant risk "
                "in downside scenarios. Assumes a 30-year fully-amortizing loan."
            ),
        )
        interest_rate = st.slider(
            "Mortgage Interest Rate",
            min_value=0.03, max_value=0.12,
            value=0.065, step=0.0025,
            format="%.2f%%",
            help=(
                "Annual all-in interest rate on the acquisition loan. Directly "
                "drives the annual Debt Service figure and DSCR. A 25bps move in "
                "rate typically shifts DSCR by ~3–5 basis points on a 65% LTV deal."
            ),
        )
        discount_rate = st.slider(
            "Equity Discount Rate",
            min_value=0.05, max_value=0.20,
            value=0.09, step=0.005,
            format="%.1f%%",
            help=(
                "The minimum required return on equity capital used to compute NPV. "
                "A positive NPV means the IRR exceeds this hurdle. Typically set at "
                "the fund's weighted cost of equity capital or LP distribution hurdle."
            ),
        )

        # ── Market Macro Assumptions ───────────────────────────────────
        _sb_section("Market Macro Assumptions")
        exit_cap_rate = st.slider(
            "Exit Cap Rate",
            min_value=0.03, max_value=0.12,
            value=0.055, step=0.0025,
            format="%.2f%%",
            help=(
                "Capitalisation rate at which the asset is assumed to trade at the "
                "end of the 10-year hold: Terminal Value = Year 11 NOI ÷ Exit Cap. "
                "A 50bps expansion in exit cap on a $50M asset typically reduces "
                "gross exit value by $4–6M depending on NOI growth."
            ),
        )
        vacancy_rate = st.slider(
            "Market Vacancy Rate",
            min_value=0.00, max_value=0.35,
            value=0.05, step=0.005,
            format="%.1f%%",
            help=(
                "Stabilised physical vacancy assumption applied to Potential Gross "
                "Income (PGI) to derive Effective Gross Income (EGI). Captures both "
                "structural and cyclical vacancy. Does not replace per-tenant "
                "delinquency risk — see the Auditing tab for live delinquency rates."
            ),
        )
        rent_growth = st.slider(
            "Rent Growth Rate",
            min_value=0.00, max_value=0.08,
            value=0.03, step=0.0025,
            format="%.2f%%",
            help=(
                "Annual rent escalation applied to the blended weighted-average PSF "
                "and used as the fallback for leases with no explicit clause. Leases "
                "with Fixed-Rate, CPI-Linked, or Stepped-Jump clauses override this "
                "assumption on a per-lease basis."
            ),
        )
        general_inflation = st.slider(
            "General Inflation (CPI)",
            min_value=0.00, max_value=0.08,
            value=0.025, step=0.0025,
            format="%.2f%%",
            help=(
                "Macroeconomic inflation rate. Drives rent escalation on CPI-Linked "
                "leases and the annual growth rate of operating expenses throughout "
                "the hold period. Decoupled from rent growth to model scenarios "
                "where expense inflation outpaces rent."
            ),
        )

        st.divider()
        st.markdown(
            f"<div style='font-size:11px;color:{_SLATE};text-align:center;'>"
            f"CRE-Val Platform  ·  Institutional Grade</div>",
            unsafe_allow_html=True,
        )

    return dict(
        source_format     = source_format,
        asset_class       = asset_class,
        purchase_price    = float(purchase_price_input),
        ltv               = ltv,
        interest_rate     = interest_rate,
        discount_rate     = discount_rate,
        exit_cap_rate     = exit_cap_rate,
        vacancy_rate      = vacancy_rate,
        rent_growth       = rent_growth,
        general_inflation = general_inflation,
    )


# ---------------------------------------------------------------------------
# Pipeline helpers (unchanged from original)
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
    shared = dict(
        property_id          = property_id,
        purchase_price       = purchase_price,
        total_sqft           = sum(r.square_footage for r in records),
        base_rent_psf        = _weighted_avg_rent(records),
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
    records:     list[LeaseRecord],
    asset:       Asset,
    dcf:         DCFResult,
    asset_class: str,
) -> dict[str, Any]:
    total_rent  = sum(r.base_rent_psf * r.square_footage for r in records)
    delinq_rent = sum(
        r.base_rent_psf * r.square_footage for r in records if r.is_delinquent
    )
    delinq_rate   = delinq_rent / total_rent if total_rent > 0 else 0.0
    delinq_count  = sum(1 for r in records if r.is_delinquent)

    alerts: list[dict] = []
    if dcf.dscr_year_1 < 1.25:
        alerts.append({
            "severity": "CRITICAL",
            "message":  f"DSCR {dcf.dscr_year_1:.2f}x is below the 1.25x minimum — "
                        "lender covenant breach risk.",
        })
    elif dcf.dscr_year_1 < 1.40:
        alerts.append({
            "severity": "WARNING",
            "message":  f"DSCR {dcf.dscr_year_1:.2f}x is in warning band (1.25x–1.40x). "
                        "Monitor debt coverage closely.",
        })

    if delinq_rate > 0.05:
        alerts.append({
            "severity": "WARNING",
            "message":  f"Tenant delinquency {delinq_rate:.1%} ({delinq_count} tenant"
                        f"{'s' if delinq_count != 1 else ''}) exceeds the 5% threshold "
                        "— contracted rent collections at risk.",
        })
    elif delinq_rate > 0.03:
        alerts.append({
            "severity": "INFO",
            "message":  f"Delinquency rate {delinq_rate:.1%} is approaching the 5% "
                        "warning threshold.",
        })

    if dcf.irr < 0.08:
        alerts.append({
            "severity": "CRITICAL",
            "message":  f"Levered IRR {dcf.irr:.2%} is below the 8% minimum hurdle rate.",
        })

    sev_rank = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
    status = (
        "OK" if not alerts
        else max(alerts, key=lambda a: sev_rank.get(a["severity"], 0))["severity"]
    )

    return {
        "property_name":   asset.property_id,
        "asset_class":     asset_class,
        "overall_status":  status,
        "alerts":          alerts,
        "delinquency_rate": delinq_rate,
        "delinquency_rent": delinq_rent,
        "delinquency_count": delinq_count,
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
# ETL caching layer
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _parse_and_validate(
    file_bytes: bytes,
    source_format: str,
    mapping_cfg_json: str,
) -> tuple[list[dict], list[dict]]:
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
    return [
        LeaseRecord(
            property_id     = d["property_id"],
            tenant_name     = d["tenant_name"],
            square_footage  = d["square_footage"],
            base_rent_psf   = d["base_rent_psf"],
            lease_start     = datetime.fromisoformat(d["lease_start"]).date(),
            lease_end       = datetime.fromisoformat(d["lease_end"]).date(),
            is_delinquent   = d["is_delinquent"],
            escalation_type = d["escalation_type"],
            recovery_type   = d["recovery_type"],
        )
        for d in dicts
    ]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_dollar(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.2f}M"
    return f"${v:,.0f}"


def _irr_border(irr: float) -> str:
    if irr >= 0.12:
        return _GREEN
    if irr >= 0.08:
        return _AMBER
    return _RED


def _dscr_border(dscr: float) -> str:
    if dscr >= 1.40:
        return _GREEN
    if dscr >= 1.25:
        return _AMBER
    return _RED


def _section_header(label: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-header-label">{label}</span>'
        f'<div class="section-header-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _kpi_card(
    label:        str,
    value:        str,
    delta:        str | None = None,
    delta_up:     bool       = True,
    border_color: str        = _GREEN,
    sub:          str | None = None,
) -> str:
    arrow = "▲" if delta_up else "▼"
    d_color = _GREEN if delta_up else _RED
    delta_html = (
        f'<div class="kpi-delta" style="color:{d_color};">{arrow} {delta}</div>'
        if delta else ""
    )
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card" style="border-top:3px solid {border_color};">'
        f'<div class="kpi-eyebrow">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}{sub_html}'
        f'</div>'
    )


def _color_irr_cell(val: str) -> str:
    try:
        irr = float(val.strip("%")) / 100
    except (ValueError, AttributeError):
        return ""
    if irr >= 0.12:
        return f"background-color:{_GREEN_BG};color:{_GREEN};font-weight:700"
    if irr >= 0.08:
        return f"background-color:{_AMBER_BG};color:{_AMBER};font-weight:700"
    return f"background-color:{_RED_BG};color:{_RED};font-weight:700"


def _build_sensitivity_df(sm: SensitivityMatrix) -> pd.DataFrame:
    rows = {}
    for v in sm.vacancy_range:
        row = {}
        for c in sm.cap_rate_range:
            cell = sm.get(v, c)
            row[f"{c:.2%}"] = f"{cell.combined_irr:.2%}" if cell else "N/A"
        rows[f"{v:.1%}"] = row
    df = pd.DataFrame(rows).T
    df.index.name = "Vacancy  ↓  /  Exit Cap  →"
    return df


def _build_lp_gp_df(sm: SensitivityMatrix) -> pd.DataFrame:
    records = []
    for v in sm.vacancy_range:
        for c in sm.cap_rate_range:
            cell = sm.get(v, c)
            if cell:
                records.append({
                    "Vacancy":       f"{v:.1%}",
                    "Exit Cap":      f"{c:.2%}",
                    "Combined IRR":  f"{cell.combined_irr:.2%}",
                    "LP IRR (90%)":  f"{cell.lp_irr:.2%}",
                    "GP IRR (10%)":  f"{cell.gp_irr:.2%}",
                })
    return pd.DataFrame(records).set_index(["Vacancy", "Exit Cap"])


def _style_sensitivity_df(df: pd.DataFrame):
    return (
        df.style
          .applymap(_color_irr_cell)
          .set_properties(**{
              "text-align": "center",
              "font-size":  "13px",
              "padding":    "8px 14px",
          })
          .set_table_styles([
              {"selector": "th", "props": [
                  ("background-color", _NAVY),
                  ("color", "white"),
                  ("font-weight", "700"),
                  ("text-align", "center"),
                  ("padding", "8px 14px"),
              ]},
              {"selector": "th.index_name", "props": [("font-style", "italic")]},
          ])
    )


# ---------------------------------------------------------------------------
# Risk banners
# ---------------------------------------------------------------------------

def _render_risk_banners(
    report:   dict[str, Any],
    asset:    Asset,
    dcf:      DCFResult,
) -> None:
    status      = report["overall_status"]
    alerts      = report["alerts"]
    delinq_rate = report.get("delinquency_rate", 0.0)
    delinq_rent = report.get("delinquency_rent", 0.0)
    delinq_count = report.get("delinquency_count", 0)

    def _banner(cls: str, icon: str, title: str, body: str) -> None:
        st.markdown(
            f'<div class="risk-banner risk-banner--{cls}">'
            f'<div class="risk-banner-icon">{icon}</div>'
            f'<div><div class="risk-banner-title">{title}</div>'
            f'<div class="risk-banner-body">{body}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Delinquency ≥ 30% — specific dynamic cash flow adjustment banner
    if delinq_rate >= 0.30:
        model_vacancy   = asset.vacancy_rate
        suggest_vacancy = max(model_vacancy, delinq_rate * 0.8)
        _banner(
            "delinquency", "⚠️",
            f"Elevated Delinquency Alert  —  {delinq_rate:.1%} of Contracted Rent at Risk",
            f"{delinq_count} tenant{'s' if delinq_count != 1 else ''} are flagged delinquent, "
            f"representing <strong>{_fmt_dollar(delinq_rent)}/yr</strong> of contracted revenue. "
            f"The pro forma applies a structural 1.0% credit loss haircut but does not fully "
            f"absorb this operational shortfall. "
            f"<strong>Model adjustment recommended:</strong> increase the Market Vacancy Rate "
            f"from {model_vacancy:.1%} to at least <strong>{suggest_vacancy:.1%}</strong> in "
            f"the sidebar to reflect realistic EGI collection in your underwriting projections.",
        )

    # CRITICAL alerts
    crit = [a for a in alerts if a["severity"] == "CRITICAL"]
    for a in crit:
        _banner("critical", "🚨", "Critical Alert", a["message"])

    # WARNING alerts (skip delinquency if already shown in the 30% banner)
    warnings = [
        a for a in alerts
        if a["severity"] == "WARNING"
        and (delinq_rate < 0.30 or "delinquency" not in a["message"].lower())
    ]
    for a in warnings:
        _banner("warning", "⚠️", "Warning", a["message"])

    # Clean state
    if status == "OK":
        _banner(
            "ok", "✅",
            "All Metrics Within Thresholds",
            "DSCR, IRR, and delinquency are all within acceptable underwriting ranges. "
            "No active risk alerts for this asset.",
        )


# ---------------------------------------------------------------------------
# Tab 1 — Investment Summary
# ---------------------------------------------------------------------------

def _render_tab_summary(
    asset:       Asset,
    pf:          list[ProFormaYear],
    dcf:         DCFResult,
    waterfall:   WaterfallResult,
    report:      dict[str, Any],
    sensitivity: SensitivityMatrix,
    cfg:         dict[str, Any],
) -> None:
    # ── Primary KPIs ──────────────────────────────────────────────────
    _section_header("Primary Returns")

    irr_delta = dcf.irr - asset.discount_rate
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(
        _kpi_card(
            "Combined IRR",
            f"{dcf.irr:.2%}",
            delta=f"{irr_delta:+.2%} vs hurdle",
            delta_up=(irr_delta >= 0),
            border_color=_irr_border(dcf.irr),
            sub=f"Hurdle: {asset.discount_rate:.1%}",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _kpi_card(
            "LP IRR  (90%)",
            f"{waterfall.lp_irr:.2%}",
            border_color=_irr_border(waterfall.lp_irr),
            sub=f"EM {waterfall.lp_equity_multiple:.2f}x",
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _kpi_card(
            "GP IRR  (10%)",
            f"{waterfall.gp_irr:.2%}",
            border_color=_irr_border(waterfall.gp_irr),
            sub=f"EM {waterfall.gp_equity_multiple:.2f}x",
        ),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _kpi_card(
            "DSCR  Year 1",
            f"{dcf.dscr_year_1:.2f}x",
            delta="Covenant safe" if dcf.dscr_year_1 >= 1.25 else "Covenant breach",
            delta_up=(dcf.dscr_year_1 >= 1.25),
            border_color=_dscr_border(dcf.dscr_year_1),
            sub="Min threshold: 1.25x",
        ),
        unsafe_allow_html=True,
    )
    c5.markdown(
        _kpi_card(
            "Equity Multiple",
            f"{dcf.equity_multiple:.2f}x",
            delta="Positive return" if dcf.equity_multiple >= 1.0 else "Capital loss",
            delta_up=(dcf.equity_multiple >= 1.0),
            border_color=(_GREEN if dcf.equity_multiple >= 1.5 else
                          _AMBER if dcf.equity_multiple >= 1.0 else _RED),
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Capital Structure ──────────────────────────────────────────────
    _section_header("Capital Structure & Valuation")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.markdown(
        _kpi_card("Purchase Price",    _fmt_dollar(asset.purchase_price),
                  border_color=_NAVY),
        unsafe_allow_html=True,
    )
    c7.markdown(
        _kpi_card("Equity Invested",   _fmt_dollar(dcf.equity_invested),
                  sub=f"LTV {asset.loan_to_value:.0%}",
                  border_color=_NAVY),
        unsafe_allow_html=True,
    )
    c8.markdown(
        _kpi_card(
            "Net Present Value",
            _fmt_dollar(dcf.npv),
            delta="Positive NPV" if dcf.npv >= 0 else "Negative NPV",
            delta_up=(dcf.npv >= 0),
            border_color=(_GREEN if dcf.npv >= 0 else _RED),
        ),
        unsafe_allow_html=True,
    )
    c9.markdown(
        _kpi_card("Going-In Cap Rate", f"{dcf.year_1_cap_rate:.2%}",
                  sub=f"Debt yield {dcf.debt_yield:.2%}",
                  border_color=_NAVY),
        unsafe_allow_html=True,
    )
    c10.markdown(
        _kpi_card("Gross Exit Value",  _fmt_dollar(dcf.exit_value),
                  sub=f"Net: {_fmt_dollar(dcf.net_exit_proceeds)}",
                  border_color=_NAVY),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity Waterfall Summary ───────────────────────────────────────
    _section_header("Equity Waterfall  —  90 / 10 LP / GP Split  (Hurdles: 8% · 12%)")

    wf = waterfall
    wf_data = {
        "":                   ["LP  (90%)",                 "GP  (10%)"],
        "Equity Committed":   [_fmt_dollar(wf.lp_equity),  _fmt_dollar(wf.gp_equity)],
        "Total Distributions":[_fmt_dollar(wf.lp_distributions), _fmt_dollar(wf.gp_distributions)],
        "IRR":                [f"{wf.lp_irr:.2%}",         f"{wf.gp_irr:.2%}"],
        "Equity Multiple":    [f"{wf.lp_equity_multiple:.2f}x", f"{wf.gp_equity_multiple:.2f}x"],
        "Tier 1  (0–8%)":     [_fmt_dollar(wf.tier1_distributed * 0.9),
                               _fmt_dollar(wf.tier1_distributed * 0.1)],
        "Tier 2  (8–12%)":    [_fmt_dollar(wf.tier2_distributed * 0.9),
                               _fmt_dollar(wf.tier2_distributed * 0.1)],
        "Tier 3  (12%+)":     [_fmt_dollar(wf.tier3_distributed * 0.9),
                               _fmt_dollar(wf.tier3_distributed * 0.1)],
    }
    wf_df = pd.DataFrame(wf_data).set_index("")
    st.dataframe(
        wf_df.style.set_table_styles([
            {"selector": "th", "props": [
                ("background-color", _NAVY), ("color", "white"), ("font-weight", "700"),
            ]},
        ]).set_properties(**{"text-align": "right", "font-size": "13px"}),
        use_container_width=True,
        height=310,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 10-Year Pro Forma ──────────────────────────────────────────────
    _section_header("10-Year Pro Forma Projection")

    df = pro_forma_to_dataframe(pf)
    money_cols = [c for c in df.columns if c != "Rent PSF ($)"]
    styled_pf = (
        df.style
          .format({c: "${:,.0f}" for c in money_cols})
          .format({"Rent PSF ($)": "${:.2f}"})
          .applymap(
              lambda v: (
                  f"color:{_GREEN};font-weight:700" if v >= 0
                  else f"color:{_RED};font-weight:700"
              ),
              subset=["Levered Net Cash Flow ($)"],
          )
          .set_table_styles([
              {"selector": "th", "props": [
                  ("background-color", _NAVY), ("color", "white"),
                  ("font-weight", "700"), ("text-align", "right"),
              ]},
              {"selector": "th.row_heading", "props": [
                  ("background-color", _NAVY), ("color", "white"),
              ]},
          ])
          .set_properties(**{"text-align": "right", "font-size": "12px"})
    )
    st.dataframe(styled_pf, use_container_width=True, height=425)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PDF Export ────────────────────────────────────────────────────
    _section_header("Export Investment Committee Memorandum")

    dl_col, info_col = st.columns([2, 5])
    with dl_col:
        with st.spinner("Preparing PDF …"):
            pdf_bytes = _generate_pdf_bytes(
                asset, pf, dcf, report, wf, sensitivity
            )
        filename = f"{asset.property_id}_ICM_{date.today().strftime('%Y%m%d')}.pdf"
        st.download_button(
            label="⬇  Download ICM PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )
    with info_col:
        st.markdown(
            f"<div style='font-size:12.5px;color:{_SLATE};padding-top:6px;'>"
            f"Generates a <strong>4-page Investment Committee Memorandum</strong> including "
            f"Cover & Executive Summary · 10-Year Pro Forma · DCF + Equity Waterfall + "
            f"Lease Roll · Macro Sensitivity Matrix.  "
            f"<strong>{len(pdf_bytes) / 1024:.0f} KB</strong></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 — Macro Sensitivity Matrix
# ---------------------------------------------------------------------------

def _render_tab_sensitivity(sm: SensitivityMatrix, asset: Asset) -> None:
    st.markdown(
        f"<div style='font-size:13px;color:{_SLATE};margin-bottom:12px;'>"
        f"Parametric stress test: <strong>{len(sm.vacancy_range)} vacancy levels</strong> × "
        f"<strong>{len(sm.cap_rate_range)} exit cap rate scenarios</strong>.  "
        f"Each cell shows the Combined Levered IRR. "
        f"Color thresholds: "
        f"<span style='color:{_GREEN};font-weight:700'>green ≥ 12%</span>  ·  "
        f"<span style='color:{_AMBER};font-weight:700'>amber 8–12%</span>  ·  "
        f"<span style='color:{_RED};font-weight:700'>red &lt; 8%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    heat_tab, split_tab = st.tabs(["Combined IRR Heatmap", "LP / GP Split Detail"])

    with heat_tab:
        irr_df = _build_sensitivity_df(sm)
        st.dataframe(
            _style_sensitivity_df(irr_df),
            use_container_width=True,
            height=int(40 * (len(sm.vacancy_range) + 1) + 50),
        )

        # Base-case callout
        base_cell = sm.get(asset.vacancy_rate, asset.exit_cap_rate)
        if base_cell:
            bc_irr   = base_cell.combined_irr
            bc_color = _irr_border(bc_irr)
            st.markdown(
                f'<div style="background:{_LIGHT};border-left:4px solid {bc_color};'
                f'padding:10px 16px;border-radius:0 6px 6px 0;font-size:13px;margin-top:10px;">'
                f'<span style="font-weight:700;color:{_NAVY}">Base Case</span>  '
                f'Vacancy <strong>{asset.vacancy_rate:.1%}</strong>  ·  '
                f'Exit Cap <strong>{asset.exit_cap_rate:.2%}</strong>  →  '
                f'<span style="color:{bc_color};font-weight:700;font-size:15px">'
                f'{bc_irr:.2%}</span> Combined IRR  ·  '
                f'LP {base_cell.lp_irr:.2%}  ·  GP {base_cell.gp_irr:.2%}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                f"Base case (vacancy {asset.vacancy_rate:.1%}, "
                f"exit cap {asset.exit_cap_rate:.2%}) is outside the stress grid.",
                icon="ℹ️",
            )

    with split_tab:
        lp_gp_df = _build_lp_gp_df(sm)
        st.dataframe(
            lp_gp_df.style
                .applymap(_color_irr_cell, subset=["Combined IRR"])
                .set_properties(**{"text-align": "center", "font-size": "13px"})
                .set_table_styles([{
                    "selector": "th",
                    "props": [
                        ("background-color", _NAVY), ("color", "white"),
                        ("font-weight", "700"), ("text-align", "center"),
                    ],
                }]),
            use_container_width=True,
            height=int(38 * len(lp_gp_df) + 70),
        )
        st.caption(
            "LP receives 90% of distributions up to each hurdle tier.  "
            "GP carries the 10% co-invest and earns the promote above the 12% hurdle.  "
            "Hurdle tiers: Tier 1 (0–8%), Tier 2 (8–12%), Tier 3 (12%+)."
        )


# ---------------------------------------------------------------------------
# Tab 3 — Ingested Data Auditing
# ---------------------------------------------------------------------------

def _render_tab_auditing(
    records:    list[LeaseRecord],
    valid_dicts: list[dict],
    errors:     list[dict],
    report:     dict[str, Any],
    cfg:        dict[str, Any],
) -> None:
    total       = len(valid_dicts) + len(errors)
    success_pct = len(valid_dicts) / total if total else 0.0
    delinq_rate = report.get("delinquency_rate", 0.0)
    delinq_count = report.get("delinquency_count", 0)

    # ── ETL Validation Status ─────────────────────────────────────────
    _section_header("ETL Validation Pipeline")

    if success_pct == 1.0:
        st.success(
            f"**{len(valid_dicts)} of {total} rows passed validation** — "
            f"100% ingestion success rate. No schema or type errors detected.",
            icon="✅",
        )
    elif success_pct >= 0.8:
        st.warning(
            f"**{len(valid_dicts)} of {total} rows passed validation** "
            f"({success_pct:.0%} success rate). "
            f"{len(errors)} row(s) were rejected — review errors below.",
            icon="⚠️",
        )
    else:
        st.error(
            f"**{len(valid_dicts)} of {total} rows passed validation** "
            f"({success_pct:.0%} success rate). "
            f"High rejection rate — verify you selected the correct source format "
            f"(**{cfg['source_format'].title()}**) in the sidebar.",
            icon="🚫",
        )

    # ETL stats row
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Rows Ingested",  total)
    v2.metric("Rows Valid",     len(valid_dicts))
    v3.metric("Rows Rejected",  len(errors),
              delta=f"{len(errors)} errors" if errors else None,
              delta_color="inverse" if errors else "off")
    v4.metric("Validation Rate", f"{success_pct:.0%}",
              delta_color="off")

    if errors:
        with st.expander(f"View {len(errors)} validation error(s)", expanded=False):
            for e in errors[:30]:
                st.markdown(
                    f'<div style="font-family:monospace;font-size:12px;'
                    f'color:{_RED};padding:2px 0;">'
                    f'Row {e["row_index"]}: {e["error"][:160]}</div>',
                    unsafe_allow_html=True,
                )
            if len(errors) > 30:
                st.caption(f"… and {len(errors) - 30} more errors not shown.")

    # ── Delinquency Health Check ───────────────────────────────────────
    _section_header("Portfolio Delinquency Health")

    if delinq_rate == 0.0:
        st.success(
            "**No delinquent tenants detected.** All contracted rent is current.",
            icon="✅",
        )
    elif delinq_rate < 0.03:
        st.info(
            f"**{delinq_rate:.1%} delinquency rate** ({delinq_count} tenant"
            f"{'s' if delinq_count != 1 else ''}). "
            "Within acceptable range — no action required.",
            icon="ℹ️",
        )
    elif delinq_rate < 0.30:
        st.warning(
            f"**{delinq_rate:.1%} delinquency rate** ({delinq_count} tenant"
            f"{'s' if delinq_count != 1 else ''}). "
            "Exceeds the 5% threshold. Consider increasing the Market Vacancy Rate "
            "assumption in the sidebar to reflect collection risk.",
            icon="⚠️",
        )
    else:
        delinq_rent = report.get("delinquency_rent", 0.0)
        st.error(
            f"**ELEVATED DELINQUENCY: {delinq_rate:.1%}** — "
            f"{delinq_count} tenant{'s' if delinq_count != 1 else ''}, "
            f"{_fmt_dollar(delinq_rent)}/yr of contracted rent at risk.  "
            f"The pro forma model's 1.0% credit loss haircut substantially understates "
            f"this operational risk. **Adjust the Market Vacancy Rate to "
            f"{max(0.05, delinq_rate * 0.8):.1%}+ before presenting to the Investment Committee.**",
            icon="🚨",
        )

    # ── Active Lease Roll ──────────────────────────────────────────────
    _section_header(f"Active Lease Roll  —  {len(records)} Tenant(s)")

    rows = [
        {
            "Tenant":          r.tenant_name,
            "SF":              r.square_footage,
            "Rent PSF":        r.base_rent_psf,
            "Annual Rent ($)": r.base_rent_psf * r.square_footage,
            "Lease End":       str(r.lease_end),
            "Escalation":      r.escalation_type or "—",
            "Recovery":        r.recovery_type or "Gross",
            "Status":          "⚠ Delinquent" if r.is_delinquent else "Current",
        }
        for r in sorted(records, key=lambda x: x.square_footage, reverse=True)
    ]
    df = pd.DataFrame(rows)

    def _row_style(row):
        if row["Status"] != "Current":
            return [
                f"background-color:{_RED_BG};color:{_RED};font-weight:700"
            ] * len(row)
        return [""] * len(row)

    styled_lr = (
        df.style
          .apply(_row_style, axis=1)
          .format({"SF": "{:,}", "Rent PSF": "${:.2f}", "Annual Rent ($)": "${:,.0f}"})
          .set_table_styles([{
              "selector": "th",
              "props": [
                  ("background-color", _NAVY), ("color", "white"), ("font-weight", "700"),
              ],
          }])
          .set_properties(**{"font-size": "12.5px"})
          .hide(axis="index")
    )
    st.dataframe(styled_lr, use_container_width=True,
                 height=int(38 * len(rows) + 65))

    # Lease summary stats
    total_sf   = sum(r.square_footage for r in records)
    total_rent = sum(r.base_rent_psf * r.square_footage for r in records)
    delinq_sf  = sum(r.square_footage for r in records if r.is_delinquent)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Leased SF",        f"{total_sf:,}")
    s2.metric("Gross Contracted Rent",  _fmt_dollar(total_rent))
    s3.metric("Delinquent SF",          f"{delinq_sf:,}",
              delta=f"{delinq_sf/total_sf:.1%} of portfolio" if total_sf else None,
              delta_color="inverse" if delinq_sf > 0 else "off")
    s4.metric("Wtd. Avg. Rent PSF",     f"${_weighted_avg_rent(records):.2f}")

    # ── Source Configuration Summary ──────────────────────────────────
    _section_header("Pipeline Configuration")

    cfg_rows = {
        "Source Format":    cfg["source_format"].title(),
        "Asset Class":      cfg["asset_class"].replace("_", " ").title(),
        "Purchase Price":   _fmt_dollar(cfg["purchase_price"]),
        "Exit Cap Rate":    f"{cfg['exit_cap_rate']:.2%}",
        "Market Vacancy":   f"{cfg['vacancy_rate']:.1%}",
        "Rent Growth":      f"{cfg['rent_growth']:.2%}",
        "CPI Inflation":    f"{cfg['general_inflation']:.2%}",
        "LTV":              f"{cfg['ltv']:.0%}",
        "Interest Rate":    f"{cfg['interest_rate']:.2%}",
        "Discount Rate":    f"{cfg['discount_rate']:.1%}",
        "Run Date":         date.today().strftime("%B %d, %Y"),
    }
    cfg_df = pd.DataFrame(
        [{"Parameter": k, "Value": v} for k, v in cfg_rows.items()]
    )
    st.dataframe(
        cfg_df.style
            .set_table_styles([{
                "selector": "th",
                "props": [("background-color", _NAVY), ("color", "white"),
                          ("font-weight", "700")],
            }])
            .set_properties(**{"font-size": "12.5px"})
            .hide(axis="index"),
        use_container_width=True,
        height=440,
    )


# ---------------------------------------------------------------------------
# Landing page (pre-upload)
# ---------------------------------------------------------------------------

def _render_landing() -> None:
    st.markdown(
        f"""
        <div class="landing-hero">
            <h2>Upload a Rent Roll to Begin</h2>
            <p>
                Drop a CSV exported from any supported property management system.
                The pipeline validates every row through the ETL schema layer,
                then runs a full 10-year pro forma, discounted cash flow analysis,
                three-tier equity waterfall, and macro sensitivity matrix — all
                driven by the assumptions in the sidebar.
            </p>
            <div style="margin-top:16px;">
                <span class="feature-pill">📋  ETL Validation</span>
                <span class="feature-pill">📊  10-Year Pro Forma</span>
                <span class="feature-pill">💰  DCF + IRR</span>
                <span class="feature-pill">🏦  Equity Waterfall</span>
                <span class="feature-pill">📈  Sensitivity Matrix</span>
                <span class="feature-pill">📄  PDF Export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Expected CSV schema (Yardi example)", expanded=False):
        st.caption(
            "Column names are remapped per format. Yardi columns shown below."
        )
        st.dataframe(
            pd.DataFrame([{
                "PropCode":       "BLDG-001",
                "TenantName":     "TENANT-A",
                "UnitSqFt":       "12000",
                "BaseRentPerSF":  "45.00",
                "LeaseFromDate":  "01/01/2024",
                "LeaseToDate":    "12/31/2029",
                "DelinquentFlag": "N",
                "Escalation_Type": "Fixed-3%",
                "Recovery_Type":  "NNN",
            }]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "**Supported formats:** Yardi · RealPage · MRI · AppFolio · Entrata  "
            "— select the matching format in the sidebar before uploading."
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()

    # ── Confidential banner ───────────────────────────────────────────
    st.markdown(
        '<div class="conf-banner">'
        '<span>🏢</span>'
        '<span>CRE-VAL  ·  INVESTMENT UNDERWRITING PLATFORM  ·  '
        'STRICTLY CONFIDENTIAL  —  FOR INTERNAL USE ONLY</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Page title ────────────────────────────────────────────────────
    st.markdown(
        f'<div class="page-eyebrow">Commercial Real Estate</div>'
        f'<div class="page-title">Investment Underwriting Dashboard</div>'
        f'<div class="page-subtitle">'
        f'Institutional-grade pro forma · DCF · waterfall · sensitivity analysis'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────
    cfg = _render_sidebar()

    # ── File uploader ─────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Rent Roll CSV",
        type=["csv"],
        help=(
            "Upload a rent roll CSV exported from your property management system. "
            "Select the matching source format in the sidebar first."
        ),
        label_visibility="collapsed",
    )

    if uploaded is None:
        _render_landing()
        return

    # ── ETL parse + validate ──────────────────────────────────────────
    import json
    mapping_cfg      = load_mapping_config()
    mapping_cfg_json = json.dumps(mapping_cfg)

    with st.spinner("Validating rent roll through ETL pipeline …"):
        valid_dicts, errors = _parse_and_validate(
            uploaded.read(), cfg["source_format"], mapping_cfg_json
        )

    if not valid_dicts:
        st.error(
            "**All rows failed validation.** "
            "Verify you selected the correct source format "
            f"(**{cfg['source_format'].title()}**) in the sidebar. "
            "Check the column names against the expected schema.",
            icon="🚫",
        )
        _render_landing()
        return

    records = _dicts_to_records(valid_dicts)

    # ── Property selection ────────────────────────────────────────────
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
    total_sf     = sum(r.square_footage for r in prop_records)
    avg_rent     = _weighted_avg_rent(prop_records)
    total_etl    = len(valid_dicts) + len(errors)

    # Property identity bar
    st.markdown(
        f'<div class="prop-bar">'
        f'<div><div class="prop-bar-name">{pid}</div>'
        f'<div class="prop-bar-meta">'
        f'{cfg["asset_class"].replace("_", " ").title()}  ·  '
        f'{len(prop_records)} lease(s)  ·  {total_sf:,} SF  ·  '
        f'Wtd. avg. rent ${avg_rent:.2f} PSF</div></div>'
        f'<div class="prop-bar-right">'
        f'{len(valid_dicts)}/{total_etl} rows validated  ·  '
        f'{date.today().strftime("%b %d, %Y")}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Pipeline ──────────────────────────────────────────────────────
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
            st.error(f"**Pipeline error:** {exc}", icon="🚫")
            st.exception(exc)
            return

    # ── Risk banners — shown above all tabs ───────────────────────────
    _render_risk_banners(report, asset, dcf)

    # ── Three-tab analytics view ──────────────────────────────────────
    tab_summary, tab_sensitivity, tab_auditing = st.tabs([
        "📊  Investment Summary",
        "📈  Macro Sensitivity Matrix",
        "📃  Ingested Data Auditing",
    ])

    with tab_summary:
        _render_tab_summary(
            asset, pf, dcf, waterfall, report, sensitivity, cfg
        )

    with tab_sensitivity:
        _render_tab_sensitivity(sensitivity, asset)

    with tab_auditing:
        _render_tab_auditing(
            prop_records, valid_dicts, errors, report, cfg
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
