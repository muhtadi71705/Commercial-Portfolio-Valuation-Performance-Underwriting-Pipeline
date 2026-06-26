"""
Investment Committee Memorandum (ICM) — PDF report generator.

Produces a clean 3-page typeset PDF using fpdf2 + matplotlib:
  Page 1  Cover & Executive Summary (key metrics, alert banner)
  Page 2  10-Year Pro Forma table + NOI/DS bar chart + cumulative-CF line chart
  Page 3  DCF Summary, Lease Roll table, and HOLD / MONITOR / SELL recommendation

Usage
-----
    from src.visual.custom_reports import ICMemorandum
    memo = ICMemorandum(asset, pro_forma, dcf_result, am_report)
    memo.build("reports/BLDG-001_ICM.pdf")
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fpdf import FPDF

from src.analytics.simulator import SensitivityMatrix
from src.core.underwriting import Asset, DCFResult, ProFormaYear
from src.core.waterfall import WaterfallResult

matplotlib.use("Agg")   # headless; no display needed


# ---------------------------------------------------------------------------
# Brand palette  (R, G, B)
# ---------------------------------------------------------------------------

NAVY  = (27,  58,  107)
GOLD  = (184, 147,  68)
LIGHT = (240, 243, 248)
WHITE = (255, 255, 255)
BLACK = (25,  25,  25)
MUTED = (110, 120, 135)
GREEN = (21,  128,  61)
AMBER = (180,  83,   9)
RED   = (185,  28,  28)

PAGE_W  = 215.9   # Letter width  mm
PAGE_H  = 279.4   # Letter height mm
MARGIN  = 14.0
BODY_W  = PAGE_W - 2 * MARGIN   # 187.9 mm usable width


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _m(v: float, decimals: int = 2) -> str:
    """Format a dollar value in millions: $3.40M, or $0.27M for small values."""
    if v == 0:
        return "$0"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v) / 1_000_000:.{decimals}f}M"


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _x(v: float) -> str:
    return f"{v:.2f}x"


def _dollar(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


# Helvetica is Latin-1 only; sanitise any string before handing it to fpdf2.
_UNICODE_MAP = str.maketrans({
    "–": "-",   # en dash
    "—": "-",   # em dash
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "•": "*",   # bullet
    "…": "...", # ellipsis
    "·": ".",   # middle dot
    "–": "-",
})


def _s(text: str) -> str:
    """Return *text* safe for Helvetica (Latin-1) rendering."""
    return (
        text.translate(_UNICODE_MAP)
            .encode("latin-1", errors="replace")
            .decode("latin-1")
    )


# ---------------------------------------------------------------------------
# Chart builders  (return BytesIO PNG buffers)
# ---------------------------------------------------------------------------


def _chart_noi_vs_ds(pro_forma: list[ProFormaYear], w_px: int, h_px: int) -> io.BytesIO:
    """Grouped bar chart: NOI vs Debt Service across 10 years."""
    years = [y.year for y in pro_forma]
    noi   = [y.net_operating_income / 1e6 for y in pro_forma]
    ds    = [y.debt_service          / 1e6 for y in pro_forma]

    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("#F0F3F8")
    ax.set_facecolor("#F0F3F8")

    x     = range(len(years))
    width = 0.38
    bars_n = ax.bar([i - width / 2 for i in x], noi, width, label="NOI",
                    color="#1B3A6B", zorder=3)
    bars_d = ax.bar([i + width / 2 for i in x], ds, width,  label="Debt Service",
                    color="#B89344", zorder=3, alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Y{y}" for y in years], fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, framealpha=0)
    ax.set_title("NOI vs. Debt Service", fontsize=9, fontweight="bold",
                 color="#1B3A6B", pad=6)
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_cumulative_cf(pro_forma: list[ProFormaYear], equity: float,
                          w_px: int, h_px: int) -> io.BytesIO:
    """Cumulative levered cash flow (equity-in as Year 0 outflow)."""
    cfs = [-equity / 1e6] + [y.levered_net_cash_flow / 1e6 for y in pro_forma]
    cum = []
    running = 0.0
    for c in cfs:
        running += c
        cum.append(running)
    xs = list(range(len(cum)))

    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor("#F0F3F8")
    ax.set_facecolor("#F0F3F8")

    pos = [max(c, 0) for c in cum]
    neg = [min(c, 0) for c in cum]
    ax.fill_between(xs, 0, pos, color="#15803D", alpha=0.25, zorder=2)
    ax.fill_between(xs, neg, 0, color="#B91C1C", alpha=0.25, zorder=2)
    ax.plot(xs, cum, color="#1B3A6B", linewidth=1.8, zorder=3, marker="o",
            markersize=3, markerfacecolor="#B89344", markeredgewidth=0)
    ax.axhline(0, color="#1B3A6B", linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Y{i}" for i in xs], fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.tick_params(labelsize=7)
    ax.set_title("Cumulative Levered Cash Flow", fontsize=9, fontweight="bold",
                 color="#1B3A6B", pad=6)
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Main report class
# ---------------------------------------------------------------------------


class ICMemorandum:
    """
    Three-page Investment Committee Memorandum PDF.

    Parameters
    ----------
    asset         : underwriting model (CommercialAsset or MultifamilyAsset)
    pro_forma     : output of asset.generate_10_year_pro_forma()
    dcf           : output of asset.calculate_dcf(pro_forma)
    report        : output of AssetManager.generate_property_report()
    property_name : display name (falls back to report['property_name'])
    prepared_by   : byline text
    """

    def __init__(
        self,
        asset:         Asset,
        pro_forma:     list[ProFormaYear],
        dcf:           DCFResult,
        report:        dict[str, Any],
        waterfall:     WaterfallResult | None = None,
        sensitivity:   SensitivityMatrix | None = None,
        property_name: str | None = None,
        prepared_by:   str = "CRE-Val Platform",
    ) -> None:
        self._asset        = asset
        self._pf           = pro_forma
        self._dcf          = dcf
        self._report       = report
        self._wf           = waterfall
        self._sm           = sensitivity
        self._total_pages  = 4 if sensitivity is not None else 3
        self._name         = property_name or report.get("property_name") or asset.property_id
        self._by           = prepared_by
        self._ac           = (report.get("asset_class") or "commercial").title()

        pdf = FPDF(orientation="P", unit="mm", format="Letter")
        pdf.set_margins(MARGIN, MARGIN, MARGIN)
        pdf.set_auto_page_break(auto=True, margin=14)
        self._pdf = pdf

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, output_path: str | Path) -> Path:
        """Render all pages and write the PDF to *output_path*."""
        self._page_cover()
        self._page_pro_forma()
        self._page_dcf_recommendation()
        if self._sm is not None:
            self._page_sensitivity_matrix()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._pdf.output(str(out))
        return out

    # ------------------------------------------------------------------
    # Page 1 — Cover & Executive Summary
    # ------------------------------------------------------------------

    def _page_cover(self) -> None:
        pdf = self._pdf
        pdf.add_page()

        # ── navy masthead ──────────────────────────────────────────────
        pdf.set_fill_color(*NAVY)
        pdf.rect(0, 0, PAGE_W, 38, style="F")

        pdf.set_xy(MARGIN, 9)
        pdf.set_font("Helvetica", "B", 17)
        pdf.set_text_color(*WHITE)
        pdf.cell(BODY_W, 9, "INVESTMENT COMMITTEE MEMORANDUM", align="L")

        pdf.set_xy(MARGIN, 20)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*GOLD)
        pdf.cell(BODY_W, 6, "STRICTLY CONFIDENTIAL  -  FOR INTERNAL USE ONLY", align="L")

        pdf.set_xy(PAGE_W - MARGIN - 35, 21)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*WHITE)
        pdf.cell(35, 5, date.today().strftime("%B %d, %Y"), align="R")

        # ── property identity block ────────────────────────────────────
        pdf.set_xy(MARGIN, 44)
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*NAVY)
        pdf.cell(BODY_W, 10, _s(self._name), align="L")

        pdf.set_xy(MARGIN, 55)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*MUTED)
        pdf.cell(BODY_W * 0.6, 6,
                 f"Asset Class: {self._ac}  ·  Property ID: {self._asset.property_id}  ·  "
                 f"Prepared by: {self._by}")

        self._hline(63)

        # ── two-column key metrics ─────────────────────────────────────
        col_w = BODY_W / 2 - 3
        left  = [
            ("Purchase Price",      _dollar(self._asset.purchase_price)),
            ("Loan Amount",         _dollar(self._asset._loan_amount)),
            ("Equity Invested",     _dollar(self._dcf.equity_invested)),
            ("Loan-to-Value",       _pct(self._asset.loan_to_value)),
            ("Interest Rate",       _pct(self._asset.debt_interest_rate)),
            ("Total Rentable SF",   f"{self._asset.total_sqft:,}"),
        ]
        right = [
            ("Going-In Cap Rate",   _pct(self._dcf.year_1_cap_rate)),
            ("Year 1 NOI",          _dollar(self._pf[0].net_operating_income)),
            ("DSCR (Year 1)",       _x(self._dcf.dscr_year_1)),
            ("Debt Yield",          _pct(self._dcf.debt_yield)),
            ("Exit Cap Rate",       _pct(self._asset.exit_cap_rate)),
            ("Base Rent PSF",       f"${self._asset.base_rent_psf:,.2f}"),
        ]

        pdf.set_xy(MARGIN, 68)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(BODY_W, 6, "INVESTMENT HIGHLIGHTS", align="L")

        y_start = 75
        row_h   = 7.2
        for i, ((lk, lv), (rk, rv)) in enumerate(zip(left, right)):
            y = y_start + i * row_h
            fill = (i % 2 == 0)
            if fill:
                pdf.set_fill_color(*LIGHT)
                pdf.rect(MARGIN, y, BODY_W, row_h, style="F")
            self._kv_row(MARGIN,          y, col_w, row_h, lk, lv)
            self._kv_row(MARGIN + col_w + 6, y, col_w, row_h, rk, rv)

        # ── DCF performance strip ──────────────────────────────────────
        y_dcf = y_start + len(left) * row_h + 6
        self._hline(y_dcf - 2)
        pdf.set_xy(MARGIN, y_dcf)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(BODY_W, 6, "DCF PERFORMANCE", align="L")

        dcf_items = [
            ("Levered IRR",         _pct(self._dcf.irr)),
            ("NPV @ Discount Rate", _dollar(self._dcf.npv)),
            ("Equity Multiple",     _x(self._dcf.equity_multiple)),
            ("Terminal NOI",        _dollar(self._dcf.terminal_noi)),
            ("Exit Value",          _dollar(self._dcf.exit_value)),
            ("Net Exit Proceeds",   _dollar(self._dcf.net_exit_proceeds)),
        ]

        y_d   = y_dcf + 7
        col3  = BODY_W / 3
        for j, (k, v) in enumerate(dcf_items):
            col = j % 3
            row = j // 3
            x   = MARGIN + col * col3
            y   = y_d + row * row_h
            if row % 2 == 0 and col == 0:
                pdf.set_fill_color(*LIGHT)
                pdf.rect(MARGIN, y, BODY_W, row_h, style="F")
            self._kv_row(x, y, col3 - 2, row_h, k, v)

        # ── alert status banner ────────────────────────────────────────
        status = self._report.get("overall_status", "OK")
        color, label = {
            "OK":       (GREEN, "OK  -  NO ACTIVE ALERTS  -  ALL METRICS WITHIN TARGETS"),
            "INFO":     (AMBER, "INFO  -  ELEVATED  -  MONITORING RECOMMENDED"),
            "WARNING":  (AMBER, "WARN  -  ONE OR MORE METRICS REQUIRE ATTENTION"),
            "CRITICAL": (RED,   "CRITICAL ALERT  -  IMMEDIATE ACTION REQUIRED"),
            "ERROR":    (RED,   "ERROR  -  CONFIGURATION ISSUE DETECTED"),
        }.get(status, (MUTED, f"STATUS: {status}"))

        banner_y = y_d + 2 * row_h + 8
        self._banner(banner_y, label, color, height=10)

        # ── alerts list (if any) ───────────────────────────────────────
        alerts = self._report.get("alerts", [])
        if alerts:
            ay = banner_y + 14
            pdf.set_xy(MARGIN, ay)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*NAVY)
            pdf.cell(BODY_W, 5, "ACTIVE ALERTS", align="L")
            ay += 6
            for a in alerts:
                sev_col = RED if a["severity"] == "CRITICAL" else AMBER if a["severity"] == "WARNING" else MUTED
                pdf.set_xy(MARGIN, ay)
                pdf.set_font("Helvetica", "B", 7.5)
                pdf.set_text_color(*sev_col)
                pdf.cell(28, 5, f"[{a['severity']}]")
                pdf.set_font("Helvetica", "", 7.5)
                pdf.set_text_color(*BLACK)
                pdf.multi_cell(BODY_W - 28, 5, _s(a["message"]))
                ay = pdf.get_y() + 1

        self._footer(1)

    # ------------------------------------------------------------------
    # Page 2 — 10-Year Pro Forma + Charts
    # ------------------------------------------------------------------

    def _page_pro_forma(self) -> None:
        pdf = self._pdf
        pdf.add_page()

        self._section_header("10-YEAR PRO FORMA PROJECTION")

        # ── column spec ───────────────────────────────────────────────
        cols = [
            ("Year",  10, "C"),
            ("Rent PSF", 18, "R"),
            ("PGI",   27, "R"),
            ("EGI",   26, "R"),
            ("OpEx",  23, "R"),
            ("NOI",   26, "R"),
            ("DS",    24, "R"),
            ("LNCF",  26, "R"),  # sums to 180 ≤ BODY_W
        ]
        row_h = 5.8

        # header row
        y = pdf.get_y() + 2
        pdf.set_xy(MARGIN, y)
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 7.5)
        for label, w, align in cols:
            pdf.cell(w, row_h + 0.5, label, border=0, align=align, fill=True)
        pdf.ln()

        # data rows
        for i, yr in enumerate(self._pf):
            y = pdf.get_y()
            fill = (i % 2 == 0)
            if fill:
                pdf.set_fill_color(*LIGHT)
            else:
                pdf.set_fill_color(*WHITE)
            pdf.set_text_color(*BLACK)
            pdf.set_font("Helvetica", "", 7)

            lncf_col = GREEN if yr.levered_net_cash_flow >= 0 else RED
            row_vals = [
                (str(yr.year),                        "C", BLACK),
                (f"${yr.rent_psf:.2f}",               "R", BLACK),
                (_m(yr.potential_gross_income),        "R", BLACK),
                (_m(yr.effective_gross_income),        "R", BLACK),
                (_m(yr.operating_expenses),            "R", BLACK),
                (_m(yr.net_operating_income),          "R", BLACK),
                (_m(yr.debt_service),                  "R", BLACK),
                (_m(yr.levered_net_cash_flow),         "R", lncf_col),
            ]
            for (val, align, color), (_, w, _align) in zip(row_vals, cols):
                pdf.set_text_color(*color)
                pdf.cell(w, row_h, val, border=0, align=align, fill=fill)
            pdf.ln()

        # ── charts ────────────────────────────────────────────────────
        chart_y = pdf.get_y() + 5
        self._hline(chart_y - 2)
        pdf.set_xy(MARGIN, chart_y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(BODY_W, 5, "CASH FLOW ANALYSIS CHARTS", align="L")

        chart_w    = (BODY_W - 6) / 2
        chart_h    = 56
        chart_y   += 7

        buf1 = _chart_noi_vs_ds(self._pf,
                                 w_px=int(chart_w * 4.5),
                                 h_px=int(chart_h * 4.5))
        buf2 = _chart_cumulative_cf(self._pf,
                                     equity=self._dcf.equity_invested,
                                     w_px=int(chart_w * 4.5),
                                     h_px=int(chart_h * 4.5))

        pdf.image(buf1, x=MARGIN,               y=chart_y, w=chart_w, h=chart_h)
        pdf.image(buf2, x=MARGIN + chart_w + 6, y=chart_y, w=chart_w, h=chart_h)

        self._footer(2)

    # ------------------------------------------------------------------
    # Page 3 — DCF Summary, Lease Roll, Recommendation
    # ------------------------------------------------------------------

    def _page_dcf_recommendation(self) -> None:
        pdf = self._pdf
        pdf.add_page()

        self._section_header("DISCOUNTED CASH FLOW ANALYSIS & RECOMMENDATION")

        # ── DCF summary table ──────────────────────────────────────────
        dcf_rows = [
            ("Equity Invested",      _dollar(self._dcf.equity_invested)),
            ("Year 1 NOI",           _dollar(self._pf[0].net_operating_income)),
            ("Year 10 NOI",          _dollar(self._pf[9].net_operating_income)),
            ("Terminal NOI (Y11)",   _dollar(self._dcf.terminal_noi)),
            ("Exit Cap Rate",        _pct(self._asset.exit_cap_rate)),
            ("Gross Exit Value",     _dollar(self._dcf.exit_value)),
            ("Loan Balance at Exit", _dollar(self._dcf.loan_balance_at_exit)),
            ("Net Exit Proceeds",    _dollar(self._dcf.net_exit_proceeds)),
            ("NPV",                  _dollar(self._dcf.npv)),
            ("Levered IRR",          _pct(self._dcf.irr)),
            ("Equity Multiple",      _x(self._dcf.equity_multiple)),
            ("Discount Rate",        _pct(self._asset.discount_rate)),
        ]

        col1 = BODY_W * 0.55
        col2 = BODY_W - col1
        row_h = 6.4

        pdf.set_xy(MARGIN, pdf.get_y() + 2)
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col1, row_h, "Metric", fill=True, border=0)
        pdf.cell(col2, row_h, "Value",  fill=True, border=0, align="R")
        pdf.ln()

        for i, (k, v) in enumerate(dcf_rows):
            fill = (i % 2 == 0)
            pdf.set_fill_color(*(LIGHT if fill else WHITE))
            pdf.set_text_color(*BLACK)
            pdf.set_font("Helvetica", "B" if i in (8, 9, 10) else "", 8)
            pdf.cell(col1, row_h, k, fill=fill, border=0)
            # color-code IRR and NPV based on sign
            if k in ("Levered IRR", "NPV", "Equity Multiple"):
                raw = self._dcf.irr if "IRR" in k else self._dcf.npv if "NPV" in k else self._dcf.equity_multiple
                pdf.set_text_color(*(GREEN if raw >= 0 else RED))
            pdf.cell(col2, row_h, v, fill=fill, border=0, align="R")
            pdf.set_text_color(*BLACK)
            pdf.ln()

        # ── equity waterfall ───────────────────────────────────────────
        if self._wf:
            wf = self._wf
            pdf.ln(4)
            self._hline(pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*NAVY)
            pdf.cell(BODY_W, 5, "EQUITY WATERFALL  (90% LP / 10% GP  |  Hurdles: 8% / 12%)", align="L")
            pdf.ln(7)

            # two-column side-by-side: LP left, GP right
            half  = BODY_W / 2 - 3
            wrow  = 6.0
            lx    = MARGIN
            rx    = MARGIN + half + 6

            def _wf_header(x: float, label: str) -> None:
                pdf.set_xy(x, pdf.get_y())
                pdf.set_fill_color(*NAVY)
                pdf.set_text_color(*WHITE)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(half, wrow, label, fill=True, border=0, align="C")

            row_y = pdf.get_y()
            _wf_header(lx, "INVESTOR  (LP  90%)")
            pdf.set_xy(rx, row_y)
            _wf_header(rx, "SPONSOR   (GP  10%)")
            pdf.ln()

            lp_rows = [
                ("LP Equity",    _dollar(wf.lp_equity)),
                ("LP IRR",       _pct(wf.lp_irr)),
                ("LP Eq. Multiple", _x(wf.lp_equity_multiple)),
                ("LP Distributions", _dollar(wf.lp_distributions)),
            ]
            gp_rows = [
                ("GP Equity",    _dollar(wf.gp_equity)),
                ("GP IRR",       _pct(wf.gp_irr)),
                ("GP Eq. Multiple", _x(wf.gp_equity_multiple)),
                ("GP Distributions", _dollar(wf.gp_distributions)),
            ]

            for i, ((lk, lv), (rk, rv)) in enumerate(zip(lp_rows, gp_rows)):
                row_y = pdf.get_y()
                fill  = (i % 2 == 0)
                if fill:
                    pdf.set_fill_color(*LIGHT)
                    pdf.rect(lx, row_y, half, wrow, style="F")
                    pdf.rect(rx, row_y, half, wrow, style="F")
                irr_row = (lk == "LP IRR" or rk == "GP IRR")
                lv_col  = GREEN if (irr_row and wf.lp_irr  >= 0) else BLACK
                rv_col  = GREEN if (irr_row and wf.gp_irr  >= 0) else BLACK
                self._kv_row(lx, row_y, half, wrow, lk, lv)
                pdf.set_text_color(*lv_col)
                self._kv_row(rx, row_y, half, wrow, rk, rv)
                pdf.set_text_color(*rv_col)
                pdf.set_xy(lx, row_y + wrow)

            # tier distribution summary
            pdf.ln(2)
            tier_y = pdf.get_y()
            pdf.set_fill_color(*LIGHT)
            pdf.rect(lx, tier_y, BODY_W, wrow, style="F")
            t_col = BODY_W / 3
            tiers = [
                (f"Tier 1 (0-8%)   {_dollar(wf.tier1_distributed)}",  0),
                (f"Tier 2 (8-12%)  {_dollar(wf.tier2_distributed)}",  1),
                (f"Tier 3 (12%+)   {_dollar(wf.tier3_distributed)}",  2),
            ]
            for label, col in tiers:
                pdf.set_xy(lx + col * t_col, tier_y)
                pdf.set_font("Helvetica", "", 7.5)
                pdf.set_text_color(*MUTED)
                pdf.cell(t_col, wrow, label, align="C")
            pdf.ln(wrow + 3)

        # ── lease roll ─────────────────────────────────────────────────
        lease_roll = self._report.get("lease_roll", [])
        if lease_roll:
            pdf.ln(4)
            self._hline(pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*NAVY)
            pdf.cell(BODY_W, 5, "ACTIVE LEASE ROLL", align="L")
            pdf.ln(6)

            lr_cols = [
                ("Tenant",        60, "L"),
                ("SF",            22, "R"),
                ("Rent PSF",      22, "R"),
                ("Annual Rent",   32, "R"),
                ("Lease End",     28, "C"),
                ("Delinquent",    22, "C"),
            ]
            lrow_h = 5.8

            pdf.set_fill_color(*NAVY)
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 7.5)
            for lbl, w, align in lr_cols:
                pdf.cell(w, lrow_h, lbl, border=0, align=align, fill=True)
            pdf.ln()

            for i, t in enumerate(lease_roll):
                fill    = (i % 2 == 0)
                delinq  = bool(t.get("is_delinquent"))
                pdf.set_fill_color(*(LIGHT if fill else WHITE))
                pdf.set_font("Helvetica", "", 7)
                row_data = [
                    (_s(t["tenant_name"][:28]),            "L", RED if delinq else BLACK),
                    (f"{t['square_footage']:,}",           "R", BLACK),
                    (f"${t['base_rent_psf']:.2f}",         "R", BLACK),
                    (_dollar(t["annual_rent"]),            "R", BLACK),
                    (str(t["lease_end"]),                  "C", BLACK),
                    ("YES" if delinq else "-",              "C", RED if delinq else MUTED),
                ]
                for (val, align, color), (_, w, _align) in zip(row_data, lr_cols):
                    pdf.set_text_color(*color)
                    pdf.cell(w, lrow_h, val, border=0, align=align, fill=fill)
                pdf.ln()

        # ── recommendation banner ──────────────────────────────────────
        rec, rationale, color = self._recommendation
        pdf.ln(6)
        banner_y = pdf.get_y()

        # outer frame
        pdf.set_fill_color(*color)
        pdf.rect(MARGIN, banner_y, BODY_W, 26, style="F")

        # label
        pdf.set_xy(MARGIN, banner_y + 4)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*WHITE)
        pdf.cell(BODY_W, 5, "INVESTMENT COMMITTEE RECOMMENDATION", align="C")

        # verdict
        pdf.set_xy(MARGIN, banner_y + 10)
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(BODY_W, 10, _s(rec), align="C")

        # rationale
        pdf.set_xy(MARGIN, banner_y + 20)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.cell(BODY_W, 5, _s(rationale), align="C")

        self._footer(3)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _section_header(self, title: str) -> None:
        pdf = self._pdf
        pdf.set_fill_color(*NAVY)
        pdf.rect(0, 0, PAGE_W, 18, style="F")
        pdf.set_xy(MARGIN, 5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*WHITE)
        pdf.cell(BODY_W * 0.7, 8, _s(title), align="L")
        pdf.set_xy(MARGIN, 5)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*GOLD)
        pdf.cell(BODY_W, 8, _s(self._name), align="R")
        pdf.set_xy(MARGIN, 22)

    def _hline(self, y: float) -> None:
        pdf = self._pdf
        pdf.set_draw_color(*NAVY)
        pdf.line(MARGIN, y, MARGIN + BODY_W, y)

    def _kv_row(self, x: float, y: float, w: float, h: float,
                key: str, val: str) -> None:
        pdf = self._pdf
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(w * 0.58, h, key)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*NAVY)
        pdf.cell(w * 0.42, h, val, align="R")

    def _banner(self, y: float, text: str, color: tuple,
                height: float = 10) -> None:
        pdf = self._pdf
        pdf.set_fill_color(*color)
        pdf.rect(MARGIN, y, BODY_W, height, style="F")
        pdf.set_xy(MARGIN, y + (height - 5) / 2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*WHITE)
        pdf.cell(BODY_W, 5, text, align="C")

    def _footer(self, page_num: int) -> None:
        pdf = self._pdf
        # Disable auto_page_break so the footer always lands on the current page,
        # not on a new one triggered by set_y being past the break threshold.
        pdf.set_auto_page_break(auto=False)
        pdf.set_y(PAGE_H - 10)
        pdf.set_font("Helvetica", "I", 6.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(BODY_W * 0.6, 5,
                 "CONFIDENTIAL  -  This document contains proprietary financial projections.",
                 align="L")
        pdf.cell(BODY_W * 0.4, 5, f"Page {page_num} of {self._total_pages}", align="R")
        pdf.set_auto_page_break(auto=True, margin=14)

    # ------------------------------------------------------------------
    # Page 4 — Macro Sensitivity Analysis
    # ------------------------------------------------------------------

    def _page_sensitivity_matrix(self) -> None:
        pdf = self._pdf
        pdf.add_page()
        self._section_header("MACRO SENSITIVITY ANALYSIS")

        sm        = self._sm
        cap_rates = sm.cap_rate_range
        vacancies = sm.vacancy_range
        n_cols    = len(cap_rates)

        # Subtitle
        pdf.set_xy(MARGIN, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        subtitle = (
            f"Parametric stress test: {len(vacancies)} vacancy levels x "
            f"{n_cols} exit cap rate scenarios  -  Combined IRR with LP / GP split"
        )
        pdf.cell(BODY_W, 5, _s(subtitle), align="L")
        pdf.ln(9)

        # Table layout
        label_w = 32.0
        col_w   = (BODY_W - label_w) / n_cols
        hdr_h   = 7.5
        row_h   = 15.0

        # Header row
        y = pdf.get_y()
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_xy(MARGIN, y)
        pdf.cell(label_w, hdr_h, "VACANCY  \\  EXIT CAP", fill=True, border=0, align="C")
        for c in cap_rates:
            pdf.cell(col_w, hdr_h, f"{c:.2%}  Exit Cap", fill=True, border=0, align="C")
        pdf.ln()

        # Data rows
        for i, v in enumerate(vacancies):
            y    = pdf.get_y()
            fill = (i % 2 == 0)
            pdf.set_fill_color(*(LIGHT if fill else WHITE))
            pdf.rect(MARGIN, y, BODY_W, row_h, style="F")

            # Vacancy label — centred vertically in the row
            pdf.set_xy(MARGIN, y + (row_h - 5) / 2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*NAVY)
            pdf.cell(label_w, 5, f"{v:.1%}", align="C")

            # Data cells: Combined IRR (large, colored) then LP/GP split below
            for j, c in enumerate(cap_rates):
                cell = sm.get(v, c)
                cx   = MARGIN + label_w + j * col_w

                if cell is None:
                    pdf.set_xy(cx, y + (row_h - 5) / 2)
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(*MUTED)
                    pdf.cell(col_w, 5, "N/A", align="C")
                    continue

                irr       = cell.combined_irr
                irr_color = GREEN if irr >= 0.12 else (AMBER if irr >= 0.08 else RED)

                # Line 1: Combined IRR
                pdf.set_xy(cx, y + 2.5)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(*irr_color)
                pdf.cell(col_w, 5, f"{irr:.2%}", align="C")

                # Line 2: LP / GP split
                pdf.set_xy(cx, y + 8.5)
                pdf.set_font("Helvetica", "", 7)
                pdf.set_text_color(*MUTED)
                pdf.cell(col_w, 4,
                         f"LP {cell.lp_irr:.1%}  /  GP {cell.gp_irr:.1%}",
                         align="C")

            pdf.set_xy(MARGIN, y + row_h)

        # Divider
        div_y = pdf.get_y() + 3
        self._hline(div_y)

        # Legend strip
        leg_y   = div_y + 4
        leg_col = BODY_W / 3
        pdf.set_fill_color(*LIGHT)
        pdf.rect(MARGIN, leg_y, BODY_W, 8, style="F")

        legend_items = [
            ("Combined IRR >= 12%  (Outperforms Hurdle)", GREEN),
            ("Combined IRR  8-12%  (Threshold Zone)",     AMBER),
            ("Combined IRR  < 8%   (Below Hurdle)",       RED),
        ]
        for k, (label, color) in enumerate(legend_items):
            bx = MARGIN + k * leg_col
            pdf.set_fill_color(*color)
            pdf.rect(bx + 3, leg_y + 2.5, 4, 3.5, style="F")
            pdf.set_xy(bx + 9, leg_y + 1.5)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*MUTED)
            pdf.cell(leg_col - 9, 5, _s(label))

        # Base-case callout (rendered only if base case falls within the grid)
        base_v    = self._asset.vacancy_rate
        base_c    = self._asset.exit_cap_rate
        base_cell = sm.get(base_v, base_c)

        if base_cell:
            call_y = leg_y + 12
            pdf.set_fill_color(*LIGHT)
            pdf.rect(MARGIN, call_y, BODY_W, 9, style="F")
            pdf.set_xy(MARGIN + 4, call_y + 2)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*NAVY)
            pdf.cell(22, 5, "Base Case:")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*BLACK)
            pdf.cell(
                BODY_W - 26, 5,
                _s(
                    f"Vacancy {base_v:.1%}  |  Exit Cap {base_c:.2%}  ->  "
                    f"Combined IRR {base_cell.combined_irr:.2%}  "
                    f"|  LP IRR {base_cell.lp_irr:.2%}  "
                    f"|  GP IRR {base_cell.gp_irr:.2%}"
                ),
            )

        self._footer(4)

    # ------------------------------------------------------------------
    # Recommendation logic
    # ------------------------------------------------------------------

    @property
    def _recommendation(self) -> tuple[str, str, tuple]:
        irr     = self._dcf.irr
        dscr    = self._dcf.dscr_year_1
        em      = self._dcf.equity_multiple
        status  = self._report.get("overall_status", "OK")
        severity_rank = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3, "ERROR": 3}
        sev = severity_rank.get(status, 0)

        if sev >= 3 or irr < 0.08 or dscr < 1.10 or em < 1.0:
            return (
                "SELL",
                "Critical financial or operational thresholds breached  -  divestiture warranted.",
                RED,
            )
        if irr >= 0.14 and dscr >= 1.40 and em >= 1.8 and sev == 0:
            return (
                "HOLD",
                f"Asset outperforming underwriting  |  IRR {_pct(irr)}  |  DSCR {_x(dscr)}  |  EM {_x(em)}",
                GREEN,
            )
        return (
            "MONITOR  -  HOLD",
            f"Meets minimum thresholds with emerging risks  |  IRR {_pct(irr)}  |  DSCR {_x(dscr)}",
            AMBER,
        )
