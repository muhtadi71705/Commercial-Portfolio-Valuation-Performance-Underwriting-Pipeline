"""
Investment Committee Memorandum (ICM) — PDF report generator.

Produces an institutional-grade PDF using ReportLab Platypus:
  Page 1  Cover & Executive Summary (key metrics, alert banner)
  Page 2  10-Year Pro Forma table + NOI/DS bar chart + cumulative-CF line chart
  Page 3  DCF Summary, Equity Waterfall, Lease Roll, HOLD / MONITOR / SELL recommendation
  Page 4  Macro Sensitivity Analysis (optional)

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
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.analytics.simulator import SensitivityMatrix
from src.core.underwriting import Asset, DCFResult, ProFormaYear
from src.core.waterfall import WaterfallResult

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------

NAVY  = colors.Color(27 / 255,  58 / 255,  107 / 255)
GOLD  = colors.Color(184 / 255, 147 / 255,  68 / 255)
LIGHT = colors.Color(240 / 255, 243 / 255, 248 / 255)
WHITE = colors.white
BLACK = colors.Color(25 / 255,  25 / 255,   25 / 255)
MUTED = colors.Color(110 / 255, 120 / 255, 135 / 255)
SLATE = colors.Color(100 / 255, 116 / 255, 139 / 255)
GREEN = colors.Color(21 / 255,  128 / 255,  61 / 255)
AMBER = colors.Color(180 / 255,  83 / 255,   9 / 255)
RED   = colors.Color(185 / 255,  28 / 255,  28 / 255)

# Raw RGB tuples used only in matplotlib
_NAVY_RGB  = (27 / 255,  58 / 255,  107 / 255)
_GOLD_RGB  = (184 / 255, 147 / 255,  68 / 255)
_GREEN_RGB = (21 / 255,  128 / 255,  61 / 255)
_RED_RGB   = (185 / 255,  28 / 255,  28 / 255)
_LIGHT_HEX = "#F0F3F8"


# ---------------------------------------------------------------------------
# Page geometry (Letter: 612 × 792 pt)
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = LETTER        # 612, 792
MARGIN         = 14 * mm       # 39.69 pt
BODY_W         = PAGE_W - 2 * MARGIN

COVER_HDR_H   = 38 * mm        # navy masthead height on cover
SECTION_HDR_H = 18 * mm        # section bar height on body pages
FOOTER_ZONE   = 14 * mm        # bottom reserved for footer


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------

def _ps(name: str, **kw) -> ParagraphStyle:
    base = dict(
        fontName="Helvetica", fontSize=9, textColor=BLACK,
        leading=13, leftIndent=0, rightIndent=0,
        spaceAfter=0, spaceBefore=0,
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


P_PROP_NAME  = _ps("prop_name",  fontName="Helvetica-Bold", fontSize=20,
                    textColor=NAVY, leading=26, spaceAfter=2 * mm)
P_SUBTITLE   = _ps("subtitle",   fontSize=9, textColor=MUTED, leading=12,
                    spaceAfter=2 * mm)
P_SECTION    = _ps("section",    fontName="Helvetica-Bold", fontSize=9,
                    textColor=NAVY, leading=12,
                    spaceBefore=4 * mm, spaceAfter=2 * mm)
P_BODY       = _ps("body",       fontSize=8, leading=11)
P_SMALL      = _ps("small",      fontSize=7.5, textColor=MUTED, leading=10)
P_ALERT_HDR  = _ps("alert_hdr",  fontName="Helvetica-Bold", fontSize=8,
                    textColor=NAVY, leading=12,
                    spaceBefore=3 * mm, spaceAfter=1 * mm)
P_BANNER_TXT = _ps("banner_txt", fontName="Helvetica-Bold", fontSize=9,
                    textColor=WHITE, leading=12, alignment=TA_CENTER)
P_REC_LABEL  = _ps("rec_label",  fontSize=8, textColor=WHITE,
                    leading=11, alignment=TA_CENTER)
P_REC_MAIN   = _ps("rec_main",   fontName="Helvetica-Bold", fontSize=20,
                    textColor=WHITE, leading=26, alignment=TA_CENTER)
P_REC_NOTE   = _ps("rec_note",   fontName="Helvetica-Oblique", fontSize=7.5,
                    textColor=WHITE, leading=10, alignment=TA_CENTER)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _m(v: float, decimals: int = 2) -> str:
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


_UNICODE_MAP = str.maketrans({
    "–": "-", "—": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "•": "*", "…": "...", "·": ".",
})


def _s(text: str) -> str:
    """Sanitise to Latin-1 for Helvetica (use in Table cell strings)."""
    return (
        text.translate(_UNICODE_MAP)
            .encode("latin-1", errors="replace")
            .decode("latin-1")
    )


def _sp(text: str) -> str:
    """Sanitise for Platypus Paragraph XML (escapes &, <, >)."""
    t = _s(text)
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Chart builders (unchanged — return BytesIO PNG buffers)
# ---------------------------------------------------------------------------

def _chart_noi_vs_ds(pro_forma: list[ProFormaYear],
                     w_px: int, h_px: int) -> io.BytesIO:
    years = [y.year for y in pro_forma]
    noi   = [y.net_operating_income / 1e6 for y in pro_forma]
    ds    = [y.debt_service          / 1e6 for y in pro_forma]

    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor(_LIGHT_HEX)
    ax.set_facecolor(_LIGHT_HEX)

    x, width = range(len(years)), 0.38
    ax.bar([i - width / 2 for i in x], noi, width, label="NOI",
           color=_NAVY_RGB, zorder=3)
    ax.bar([i + width / 2 for i in x], ds, width, label="Debt Service",
           color=_GOLD_RGB, zorder=3, alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Y{y}" for y in years], fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, framealpha=0)
    ax.set_title("NOI vs. Debt Service", fontsize=9, fontweight="bold",
                 color=_NAVY_RGB, pad=6)
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0)

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_cumulative_cf(pro_forma: list[ProFormaYear],
                          equity: float, w_px: int, h_px: int) -> io.BytesIO:
    cfs = [-equity / 1e6] + [y.levered_net_cash_flow / 1e6 for y in pro_forma]
    cum, running = [], 0.0
    for c in cfs:
        running += c
        cum.append(running)
    xs = list(range(len(cum)))

    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_facecolor(_LIGHT_HEX)
    ax.set_facecolor(_LIGHT_HEX)

    ax.fill_between(xs, 0, [max(c, 0) for c in cum],
                    color=_GREEN_RGB, alpha=0.25, zorder=2)
    ax.fill_between(xs, [min(c, 0) for c in cum], 0,
                    color=_RED_RGB, alpha=0.25, zorder=2)
    ax.plot(xs, cum, color=_NAVY_RGB, linewidth=1.8, zorder=3,
            marker="o", markersize=3,
            markerfacecolor=_GOLD_RGB, markeredgewidth=0)
    ax.axhline(0, color=_NAVY_RGB, linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Y{i}" for i in xs], fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.tick_params(labelsize=7)
    ax.set_title("Cumulative Levered Cash Flow", fontsize=9, fontweight="bold",
                 color=_NAVY_RGB, pad=6)
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
# Canvas callbacks — fixed page chrome (headers + footer)
# ---------------------------------------------------------------------------

def _draw_cover_masthead(canvas, doc, prop_name: str, prepared_by: str) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - COVER_HDR_H, PAGE_W, COVER_HDR_H, fill=1, stroke=0)

    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawString(MARGIN, PAGE_H - 14 * mm,
                      "INVESTMENT COMMITTEE MEMORANDUM")

    canvas.setFillColor(GOLD)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(MARGIN, PAGE_H - 24 * mm,
                      "STRICTLY CONFIDENTIAL  -  FOR INTERNAL USE ONLY")

    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 24 * mm,
                           date.today().strftime("%B %d, %Y"))
    canvas.restoreState()


def _draw_section_header(canvas, doc,
                          title: str, prop_name: str) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - SECTION_HDR_H, PAGE_W, SECTION_HDR_H,
                fill=1, stroke=0)

    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(MARGIN, PAGE_H - 12 * mm, _s(title))

    canvas.setFillColor(GOLD)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm, _s(prop_name))
    canvas.restoreState()


def _draw_footer(canvas, doc, page_num: int, total_pages: int) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 6.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        MARGIN, 8 * mm,
        "CONFIDENTIAL  -  This document contains proprietary financial projections.",
    )
    canvas.drawRightString(
        PAGE_W - MARGIN, 8 * mm,
        f"Page {page_num} of {total_pages}",
    )
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Reusable table builders
# ---------------------------------------------------------------------------

def _two_col_kv_table(rows: list[tuple[str, str, str, str]]) -> Table:
    """
    4-column [left_key | left_val | right_key | right_val] table.
    BOX + INNERGRID + ROWBACKGROUNDS, no header row.
    """
    lbl_w = BODY_W * 0.27
    val_w = BODY_W * 0.23
    data = [[_s(lk), _s(lv), _s(rk), _s(rv)] for lk, lv, rk, rv in rows]
    t = Table(data, colWidths=[lbl_w, val_w, lbl_w, val_w])
    t.setStyle(TableStyle([
        ("BOX",       (0, 0), (-1, -1), 0.8, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, SLATE),
        ("LINEAFTER", (1, 0), (1, -1),  1.2, NAVY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, WHITE]),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME",  (3, 0), (3, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 0), (1, -1), NAVY),
        ("TEXTCOLOR", (3, 0), (3, -1), NAVY),
        ("ALIGN",     (1, 0), (1, -1), "RIGHT"),
        ("ALIGN",     (3, 0), (3, -1), "RIGHT"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ]))
    return t


def _single_col_kv_table(rows: list[tuple[str, str]],
                          highlight_rows: set[int] | None = None,
                          value_colors: dict[int, Any] | None = None) -> Table:
    """
    2-column [metric | value] table with header styling.
    highlight_rows: row indices to bold.
    value_colors: {row_index: RL_color} overrides.
    """
    lbl_w = BODY_W * 0.58
    val_w = BODY_W - lbl_w
    data = [[_s(k), _s(v)] for k, v in rows]
    cmds = [
        ("BOX",       (0, 0), (-1, -1), 0.8, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, SLATE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, WHITE]),
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), BLACK),
        ("ALIGN",     (0, 0), (0, -1), "LEFT"),
        ("ALIGN",     (1, 0), (1, -1), "RIGHT"),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ]
    for r in (highlight_rows or set()):
        cmds.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
    for r, col in (value_colors or {}).items():
        cmds.append(("TEXTCOLOR", (1, r), (1, r), col))
    t = Table(data, colWidths=[lbl_w, val_w])
    t.setStyle(TableStyle(cmds))
    return t


def _financial_header_style(n_header_rows: int = 1) -> list[tuple]:
    """Standard header-row commands for a financial table."""
    hr = n_header_rows - 1
    return [
        ("BACKGROUND", (0, 0), (-1, hr), NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, hr), WHITE),
        ("FONTNAME",   (0, 0), (-1, hr), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, hr), 7.5),
        ("ALIGN",      (0, 0), (-1, hr), "CENTER"),
        ("LINEBELOW",  (0, hr), (-1, hr), 1.5, GOLD),
    ]


# ---------------------------------------------------------------------------
# Main report class
# ---------------------------------------------------------------------------


class ICMemorandum:
    """
    Institutional Investment Committee Memorandum PDF.

    Parameters
    ----------
    asset         : underwriting model (CommercialAsset or MultifamilyAsset)
    pro_forma     : output of asset.generate_10_year_pro_forma()
    dcf           : output of asset.calculate_dcf(pro_forma)
    report        : output of AssetManager.generate_property_report()
    waterfall     : output of run_waterfall() (optional)
    sensitivity   : output of run_macro_sensitivity_matrix() (optional)
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
        self._asset  = asset
        self._pf     = pro_forma
        self._dcf    = dcf
        self._report = report
        self._wf     = waterfall
        self._sm     = sensitivity
        self._name   = property_name or report.get("property_name") or asset.property_id
        self._by     = prepared_by
        self._ac     = (report.get("asset_class") or "commercial").title()
        self._total_pages = 4 if sensitivity is not None else 3

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        doc = BaseDocTemplate(
            str(out),
            pagesize=LETTER,
            leftMargin=0, rightMargin=0,
            topMargin=0, bottomMargin=0,
        )
        doc.addPageTemplates(self._make_page_templates())

        story = [NextPageTemplate("cover")]
        story += self._cover_story()
        story += [NextPageTemplate("body_p2"), PageBreak()]
        story += self._proforma_story()
        story += [NextPageTemplate("body_p3"), PageBreak()]
        story += self._dcf_story()
        if self._sm is not None:
            story += [NextPageTemplate("body_p4"), PageBreak()]
            story += self._sensitivity_story()

        doc.build(story)
        return out

    # ------------------------------------------------------------------
    # Page template factory
    # ------------------------------------------------------------------

    def _make_page_templates(self) -> list[PageTemplate]:
        name, by, total = self._name, self._by, self._total_pages

        cover_frame = Frame(
            MARGIN, FOOTER_ZONE,
            BODY_W, PAGE_H - COVER_HDR_H - FOOTER_ZONE,
            leftPadding=0, rightPadding=0,
            topPadding=3 * mm, bottomPadding=0,
        )
        body_frame = Frame(
            MARGIN, FOOTER_ZONE,
            BODY_W, PAGE_H - SECTION_HDR_H - FOOTER_ZONE - 2 * mm,
            leftPadding=0, rightPadding=0,
            topPadding=3 * mm, bottomPadding=0,
        )

        def on_cover(canvas, doc):
            _draw_cover_masthead(canvas, doc, name, by)
            _draw_footer(canvas, doc, 1, total)

        def on_p2(canvas, doc):
            _draw_section_header(canvas, doc,
                                  "10-YEAR PRO FORMA PROJECTION", name)
            _draw_footer(canvas, doc, 2, total)

        def on_p3(canvas, doc):
            _draw_section_header(canvas, doc,
                                  "DISCOUNTED CASH FLOW ANALYSIS & RECOMMENDATION",
                                  name)
            _draw_footer(canvas, doc, 3, total)

        def on_p4(canvas, doc):
            _draw_section_header(canvas, doc,
                                  "MACRO SENSITIVITY ANALYSIS", name)
            _draw_footer(canvas, doc, 4, total)

        return [
            PageTemplate("cover",  [cover_frame], onPage=on_cover),
            PageTemplate("body_p2", [body_frame],  onPage=on_p2),
            PageTemplate("body_p3", [body_frame],  onPage=on_p3),
            PageTemplate("body_p4", [body_frame],  onPage=on_p4),
        ]

    # ------------------------------------------------------------------
    # Page 1 — Cover & Executive Summary
    # ------------------------------------------------------------------

    def _cover_story(self) -> list:
        asset, dcf, pf = self._asset, self._dcf, self._pf

        story: list = []

        # Property identity
        story.append(Paragraph(_sp(self._name), P_PROP_NAME))
        story.append(Paragraph(
            f"Asset Class: {_sp(self._ac)}  ·  "
            f"Property ID: {_sp(asset.property_id)}  ·  "
            f"Prepared by: {_sp(self._by)}",
            P_SUBTITLE,
        ))
        story.append(HRFlowable(
            width=BODY_W, color=NAVY, thickness=1, spaceAfter=4 * mm,
        ))

        # Investment Highlights (two-column KV table)
        highlights = _two_col_kv_table([
            ("Purchase Price",    _dollar(asset.purchase_price),
             "Going-In Cap Rate", _pct(dcf.year_1_cap_rate)),
            ("Loan Amount",       _dollar(asset._loan_amount),
             "Year 1 NOI",        _dollar(pf[0].net_operating_income)),
            ("Equity Invested",   _dollar(dcf.equity_invested),
             "DSCR (Year 1)",     _x(dcf.dscr_year_1)),
            ("Loan-to-Value",     _pct(asset.loan_to_value),
             "Debt Yield",        _pct(dcf.debt_yield)),
            ("Interest Rate",     _pct(asset.debt_interest_rate),
             "Exit Cap Rate",     _pct(asset.exit_cap_rate)),
            ("Total Rentable SF", f"{asset.total_sqft:,}",
             "Base Rent PSF",     f"${asset.base_rent_psf:,.2f}"),
        ])
        story.append(KeepTogether([
            Paragraph("INVESTMENT HIGHLIGHTS", P_SECTION),
            highlights,
        ]))

        # DCF Performance
        irr_color  = GREEN if dcf.irr  >= 0 else RED
        npv_color  = GREEN if dcf.npv  >= 0 else RED
        em_color   = GREEN if dcf.equity_multiple >= 1.0 else RED
        dcf_table  = _single_col_kv_table(
            [
                ("Levered IRR",          _pct(dcf.irr)),
                ("NPV @ Discount Rate",  _dollar(dcf.npv)),
                ("Equity Multiple",      _x(dcf.equity_multiple)),
                ("Terminal NOI (Y11)",   _dollar(dcf.terminal_noi)),
                ("Gross Exit Value",     _dollar(dcf.exit_value)),
                ("Net Exit Proceeds",    _dollar(dcf.net_exit_proceeds)),
            ],
            highlight_rows={0, 1, 2},
            value_colors={0: irr_color, 1: npv_color, 2: em_color},
        )
        story.append(KeepTogether([
            Paragraph("DCF PERFORMANCE", P_SECTION),
            dcf_table,
        ]))

        # Alert status banner
        status = self._report.get("overall_status", "OK")
        banner_color, label = {
            "OK":       (GREEN, "OK  -  NO ACTIVE ALERTS  -  ALL METRICS WITHIN TARGETS"),
            "INFO":     (AMBER, "INFO  -  ELEVATED  -  MONITORING RECOMMENDED"),
            "WARNING":  (AMBER, "WARN  -  ONE OR MORE METRICS REQUIRE ATTENTION"),
            "CRITICAL": (RED,   "CRITICAL ALERT  -  IMMEDIATE ACTION REQUIRED"),
            "ERROR":    (RED,   "ERROR  -  CONFIGURATION ISSUE DETECTED"),
        }.get(status, (MUTED, f"STATUS: {status}"))

        banner = Table(
            [[Paragraph(_sp(label), P_BANNER_TXT)]],
            colWidths=[BODY_W],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), banner_color),
            ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(Spacer(0, 3 * mm))
        story.append(banner)

        # Active alerts list
        alerts = self._report.get("alerts", [])
        if alerts:
            story.append(Paragraph("ACTIVE ALERTS", P_ALERT_HDR))
            for a in alerts:
                sev = a["severity"]
                sev_color = RED if sev == "CRITICAL" else AMBER if sev == "WARNING" else MUTED
                row = Table(
                    [[
                        Paragraph(
                            f"<b>[{_sp(sev)}]</b>",
                            _ps(f"asev_{sev}", fontName="Helvetica-Bold",
                                fontSize=7.5, textColor=sev_color, leading=10),
                        ),
                        Paragraph(
                            _sp(a["message"]),
                            _ps(f"amsg_{sev}", fontSize=7.5,
                                textColor=BLACK, leading=10),
                        ),
                    ]],
                    colWidths=[22 * mm, BODY_W - 22 * mm],
                )
                row.setStyle(TableStyle([
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                    ("TOPPADDING",    (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]))
                story.append(row)

        return story

    # ------------------------------------------------------------------
    # Page 2 — 10-Year Pro Forma + Charts
    # ------------------------------------------------------------------

    def _proforma_story(self) -> list:
        pf = self._pf
        story: list = []

        # Column spec (widths sum to BODY_W)
        col_w = [
            0.055 * BODY_W,   # Year
            0.095 * BODY_W,   # Rent PSF
            0.135 * BODY_W,   # PGI
            0.135 * BODY_W,   # EGI
            0.125 * BODY_W,   # OpEx
            0.135 * BODY_W,   # NOI
            0.135 * BODY_W,   # Debt Svc
            0.185 * BODY_W,   # LNCF
        ]
        headers = ["Year", "Rent PSF", "PGI", "EGI", "OpEx", "NOI", "Debt Svc", "LNCF"]

        data = [headers]
        for yr in pf:
            data.append([
                str(yr.year),
                f"${yr.rent_psf:.2f}",
                _m(yr.potential_gross_income),
                _m(yr.effective_gross_income),
                _m(yr.operating_expenses),
                _m(yr.net_operating_income),
                _m(yr.debt_service),
                _m(yr.levered_net_cash_flow),
            ])

        cmds = [
            ("BOX",       (0, 0), (-1, -1), 0.8, NAVY),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, SLATE),
            *_financial_header_style(1),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
            ("FONTNAME",  (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",  (0, 1), (-1, -1), 7),
            ("TEXTCOLOR", (0, 1), (-1, -1), BLACK),
            ("ALIGN",     (0, 1), (0, -1),  "CENTER"),
            ("ALIGN",     (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ]
        # Per-row LNCF color (col 7)
        for i, yr in enumerate(pf, start=1):
            c = GREEN if yr.levered_net_cash_flow >= 0 else RED
            cmds.append(("TEXTCOLOR", (7, i), (7, i), c))

        pf_table = Table(data, colWidths=col_w, repeatRows=1)
        pf_table.setStyle(TableStyle(cmds))

        story.append(KeepTogether([
            Paragraph("10-YEAR CASH FLOW SUMMARY", P_SECTION),
            pf_table,
        ]))

        # Charts
        chart_w_mm = (187.9 - 6) / 2   # ~90.95 mm
        chart_h_mm = 56.0
        chart_w_pt = chart_w_mm * mm
        chart_h_pt = chart_h_mm * mm

        buf1 = _chart_noi_vs_ds(
            pf,
            w_px=int(chart_w_mm * 4.5),
            h_px=int(chart_h_mm * 4.5),
        )
        buf2 = _chart_cumulative_cf(
            pf,
            equity=self._dcf.equity_invested,
            w_px=int(chart_w_mm * 4.5),
            h_px=int(chart_h_mm * 4.5),
        )
        img1 = Image(buf1, width=chart_w_pt, height=chart_h_pt)
        img2 = Image(buf2, width=chart_w_pt, height=chart_h_pt)

        chart_table = Table(
            [[img1, img2]],
            colWidths=[chart_w_pt, chart_w_pt],
            spaceBefore=4 * mm,
        )
        chart_table.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))

        story.append(Spacer(0, 4 * mm))
        story.append(HRFlowable(width=BODY_W, color=NAVY, thickness=0.5,
                                 spaceAfter=2 * mm))
        story.append(KeepTogether([
            Paragraph("CASH FLOW ANALYSIS CHARTS", P_SECTION),
            chart_table,
        ]))

        return story

    # ------------------------------------------------------------------
    # Page 3 — DCF Summary, Equity Waterfall, Lease Roll, Recommendation
    # ------------------------------------------------------------------

    def _dcf_story(self) -> list:
        asset, dcf, pf = self._asset, self._dcf, self._pf
        story: list = []

        # DCF Summary table
        irr_c = GREEN if dcf.irr >= 0 else RED
        npv_c = GREEN if dcf.npv >= 0 else RED
        em_c  = GREEN if dcf.equity_multiple >= 1.0 else RED
        dcf_table = _single_col_kv_table(
            [
                ("Equity Invested",      _dollar(dcf.equity_invested)),
                ("Year 1 NOI",           _dollar(pf[0].net_operating_income)),
                ("Year 10 NOI",          _dollar(pf[9].net_operating_income)),
                ("Terminal NOI (Y11)",   _dollar(dcf.terminal_noi)),
                ("Exit Cap Rate",        _pct(asset.exit_cap_rate)),
                ("Gross Exit Value",     _dollar(dcf.exit_value)),
                ("Loan Balance at Exit", _dollar(dcf.loan_balance_at_exit)),
                ("Net Exit Proceeds",    _dollar(dcf.net_exit_proceeds)),
                ("NPV",                  _dollar(dcf.npv)),
                ("Levered IRR",          _pct(dcf.irr)),
                ("Equity Multiple",      _x(dcf.equity_multiple)),
                ("Discount Rate",        _pct(asset.discount_rate)),
            ],
            highlight_rows={8, 9, 10},
            value_colors={8: npv_c, 9: irr_c, 10: em_c},
        )
        story.append(KeepTogether([
            Paragraph("DCF SUMMARY", P_SECTION),
            dcf_table,
        ]))

        # Equity Waterfall
        if self._wf:
            wf = self._wf
            wf_hdr = [
                Paragraph(
                    "EQUITY WATERFALL  (90% LP / 10% GP  |  Hurdles: 8% / 12%)",
                    P_SECTION,
                )
            ]

            half_w = (BODY_W - 4 * mm) / 2

            def _wf_side(equity, irr, em, distrib):
                irr_c2 = GREEN if irr >= 0 else RED
                return _single_col_kv_table(
                    [
                        ("Equity",        _dollar(equity)),
                        ("IRR",           _pct(irr)),
                        ("Equity Multiple", _x(em)),
                        ("Distributions", _dollar(distrib)),
                    ],
                    highlight_rows={1},
                    value_colors={1: irr_c2},
                )

            lp_tbl = _wf_side(wf.lp_equity, wf.lp_irr,
                               wf.lp_equity_multiple, wf.lp_distributions)
            gp_tbl = _wf_side(wf.gp_equity, wf.gp_irr,
                               wf.gp_equity_multiple, wf.gp_distributions)

            # Side-by-side header row
            side_hdr = Table(
                [[
                    Paragraph("<b>INVESTOR  (LP  90%)</b>",
                               _ps("lp_hdr", fontName="Helvetica-Bold",
                                   fontSize=8, textColor=WHITE,
                                   leading=11, alignment=TA_CENTER)),
                    Paragraph("<b>SPONSOR   (GP  10%)</b>",
                               _ps("gp_hdr", fontName="Helvetica-Bold",
                                   fontSize=8, textColor=WHITE,
                                   leading=11, alignment=TA_CENTER)),
                ]],
                colWidths=[half_w, half_w],
            )
            side_hdr.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
                ("BOX",           (0, 0), (-1, -1), 0.8, NAVY),
                ("LINEAFTER",     (0, 0), (0, -1),  1.2, GOLD),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW",     (0, 0), (-1, -1), 1.5, GOLD),
            ]))

            sides = Table(
                [[lp_tbl, gp_tbl]],
                colWidths=[half_w, half_w],
            )
            sides.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                ("TOPPADDING",    (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LINEAFTER",     (0, 0), (0, -1),  1.2, NAVY),
            ]))

            # Tier summary strip
            tier_data = [[
                f"Tier 1 (0-8%)   {_dollar(wf.tier1_distributed)}",
                f"Tier 2 (8-12%)  {_dollar(wf.tier2_distributed)}",
                f"Tier 3 (12%+)   {_dollar(wf.tier3_distributed)}",
            ]]
            tier_tbl = Table(tier_data, colWidths=[BODY_W / 3] * 3)
            tier_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
                ("BOX",           (0, 0), (-1, -1), 0.8, NAVY),
                ("INNERGRID",     (0, 0), (-1, -1), 0.35, SLATE),
                ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
                ("TEXTCOLOR",     (0, 0), (-1, -1), MUTED),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))

            story.append(KeepTogether(wf_hdr + [side_hdr, sides, tier_tbl]))

        # Lease Roll
        lease_roll = self._report.get("lease_roll", [])
        if lease_roll:
            lr_col_w = [
                0.30 * BODY_W,   # Tenant
                0.11 * BODY_W,   # SF
                0.11 * BODY_W,   # Rent PSF
                0.16 * BODY_W,   # Annual Rent
                0.16 * BODY_W,   # Lease End
                0.16 * BODY_W,   # Delinquent
            ]
            lr_hdrs = ["Tenant", "SF", "Rent PSF", "Annual Rent", "Lease End", "Delinquent"]
            lr_data = [lr_hdrs]
            for t in lease_roll:
                delinq = bool(t.get("is_delinquent"))
                lr_data.append([
                    _s(t["tenant_name"][:28]),
                    f"{t['square_footage']:,}",
                    f"${t['base_rent_psf']:.2f}",
                    _dollar(t["annual_rent"]),
                    str(t["lease_end"]),
                    "YES" if delinq else "-",
                ])

            lr_cmds = [
                ("BOX",       (0, 0), (-1, -1), 0.8, NAVY),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, SLATE),
                *_financial_header_style(1),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
                ("FONTNAME",  (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",  (0, 1), (-1, -1), 7),
                ("TEXTCOLOR", (0, 1), (-1, -1), BLACK),
                ("ALIGN",     (0, 1), (0, -1),  "LEFT"),
                ("ALIGN",     (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN",     (4, 1), (5, -1),  "CENTER"),
                ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ]
            # Delinquent tenant name + flag in red
            for i, t in enumerate(lease_roll, start=1):
                if t.get("is_delinquent"):
                    lr_cmds.append(("TEXTCOLOR", (0, i), (0, i), RED))
                    lr_cmds.append(("TEXTCOLOR", (5, i), (5, i), RED))
                    lr_cmds.append(("FONTNAME",  (5, i), (5, i), "Helvetica-Bold"))

            lr_table = Table(lr_data, colWidths=lr_col_w, repeatRows=1)
            lr_table.setStyle(TableStyle(lr_cmds))

            story.append(KeepTogether([
                Paragraph("ACTIVE LEASE ROLL", P_SECTION),
                lr_table,
            ]))

        # Recommendation banner
        rec, rationale, rec_color = self._recommendation
        rec_table = Table(
            [
                [Paragraph("INVESTMENT COMMITTEE RECOMMENDATION", P_REC_LABEL)],
                [Paragraph(_sp(rec), P_REC_MAIN)],
                [Paragraph(_sp(rationale), P_REC_NOTE)],
            ],
            colWidths=[BODY_W],
        )
        rec_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), rec_color),
            ("BOX",           (0, 0), (-1, -1), 1.0, NAVY),
            ("TOPPADDING",    (0, 0), (0, 0),   6),
            ("TOPPADDING",    (0, 1), (0, 1),   2),
            ("TOPPADDING",    (0, 2), (0, 2),   2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(Spacer(0, 4 * mm))
        story.append(rec_table)

        return story

    # ------------------------------------------------------------------
    # Page 4 — Macro Sensitivity Matrix
    # ------------------------------------------------------------------

    def _sensitivity_story(self) -> list:
        sm = self._sm
        if sm is None:
            return []

        cap_rates = sm.cap_rate_range
        vacancies = sm.vacancy_range
        n_cols    = len(cap_rates)

        story: list = []

        # Subtitle
        story.append(Paragraph(
            f"Parametric stress test: {len(vacancies)} vacancy levels × "
            f"{n_cols} exit cap rate scenarios  —  "
            f"Combined IRR with LP / GP split",
            P_SUBTITLE,
        ))
        story.append(Spacer(0, 3 * mm))

        # Header row
        label_w = 32 * mm
        col_w   = (BODY_W - label_w) / n_cols
        hdr_row = ["VACANCY \\ EXIT CAP"] + [f"{c:.2%} Exit Cap" for c in cap_rates]

        # Data rows: each cell has Combined IRR on top, LP/GP below
        # Encode as Paragraphs inside cells for two-line layout
        data = [hdr_row]
        for v in vacancies:
            row = [f"{v:.1%} Vacancy"]
            for c in cap_rates:
                cell = sm.get(v, c)
                if cell is None:
                    row.append(Paragraph("N/A", P_SMALL))
                    continue
                irr = cell.combined_irr
                irr_col = GREEN if irr >= 0.12 else (AMBER if irr >= 0.08 else RED)
                irr_para = Paragraph(
                    f"<b>{irr:.2%}</b>",
                    _ps(f"irr_{v}_{c}", fontName="Helvetica-Bold", fontSize=10,
                        textColor=irr_col, leading=13, alignment=TA_CENTER),
                )
                sub_para = Paragraph(
                    f"LP {cell.lp_irr:.1%}  /  GP {cell.gp_irr:.1%}",
                    _ps(f"sub_{v}_{c}", fontSize=7, textColor=MUTED,
                        leading=9, alignment=TA_CENTER),
                )
                # Wrap in a mini-table to stack vertically
                cell_inner = Table(
                    [[irr_para], [sub_para]],
                    colWidths=[col_w - 2],
                )
                cell_inner.setStyle(TableStyle([
                    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                    ("TOPPADDING",    (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]))
                row.append(cell_inner)
            data.append(row)

        col_widths = [label_w] + [col_w] * n_cols

        sm_cmds = [
            ("BOX",       (0, 0), (-1, -1), 0.8, NAVY),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, SLATE),
            *_financial_header_style(1),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
            ("FONTNAME",  (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",  (0, 1), (0, -1), 9),
            ("TEXTCOLOR", (0, 1), (0, -1), NAVY),
            ("ALIGN",     (0, 1), (0, -1), "CENTER"),
            ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ]
        sm_table = Table(data, colWidths=col_widths, repeatRows=1)
        sm_table.setStyle(TableStyle(sm_cmds))

        story.append(KeepTogether([
            Paragraph("IRR SENSITIVITY MATRIX", P_SECTION),
            sm_table,
        ]))

        # Legend
        story.append(Spacer(0, 3 * mm))
        leg_col_w = BODY_W / 3
        legend_data = [[
            f"Combined IRR >= 12%  (Outperforms Hurdle)",
            f"Combined IRR  8-12%  (Threshold Zone)",
            f"Combined IRR  < 8%   (Below Hurdle)",
        ]]
        leg_tbl = Table(legend_data, colWidths=[leg_col_w] * 3)
        leg_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), GREEN),
            ("BACKGROUND",    (1, 0), (1, 0), AMBER),
            ("BACKGROUND",    (2, 0), (2, 0), RED),
            ("BOX",           (0, 0), (-1, -1), 0.8, NAVY),
            ("INNERGRID",     (0, 0), (-1, -1), 0.35, SLATE),
            ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
            ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 7),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(leg_tbl)

        # Base-case callout
        base_cell = sm.get(self._asset.vacancy_rate, self._asset.exit_cap_rate)
        if base_cell:
            story.append(Spacer(0, 3 * mm))
            callout = Table(
                [[
                    Paragraph("<b>Base Case:</b>",
                               _ps("bc_lbl", fontName="Helvetica-Bold",
                                   fontSize=8, textColor=NAVY, leading=11)),
                    Paragraph(
                        _sp(
                            f"Vacancy {self._asset.vacancy_rate:.1%}  |  "
                            f"Exit Cap {self._asset.exit_cap_rate:.2%}  ->  "
                            f"Combined IRR {base_cell.combined_irr:.2%}  "
                            f"|  LP IRR {base_cell.lp_irr:.2%}  "
                            f"|  GP IRR {base_cell.gp_irr:.2%}"
                        ),
                        _ps("bc_val", fontSize=8, textColor=BLACK, leading=11),
                    ),
                ]],
                colWidths=[22 * mm, BODY_W - 22 * mm],
            )
            callout.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
                ("BOX",           (0, 0), (-1, -1), 0.8, NAVY),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(callout)

        return story

    # ------------------------------------------------------------------
    # Recommendation logic
    # ------------------------------------------------------------------

    @property
    def _recommendation(self) -> tuple[str, str, Any]:
        irr  = self._dcf.irr
        dscr = self._dcf.dscr_year_1
        em   = self._dcf.equity_multiple
        sev  = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3, "ERROR": 3}.get(
            self._report.get("overall_status", "OK"), 0
        )
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
