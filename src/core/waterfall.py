"""
Equity waterfall model — three-tier IRR hurdle framework.

Distributes a levered equity cash flow stream between an LP (90% equity) and
GP (10% equity) across three IRR hurdle tiers:

    Tier 1   0 - 8%    90 / 10  pari-passu
    Tier 2   8 - 12%   70 / 30  GP promote
    Tier 3   12% +     50 / 50  carried interest

Algorithm: compounding-balance method
---------------------------------------
Two running "hurdle balances" compound at 8% and 12% per period (accrual
before distribution). Each period's available cash is allocated in order:

    1. Up to the 8% balance   -> Tier 1 (90/10)
    2. In the 8%-12% zone     -> Tier 2 (70/30)
    3. Above both balances    -> Tier 3 (50/50)

Every Tier-1 dollar also reduces the Tier-2 balance dollar-for-dollar because
cash that satisfies the 8% hurdle already counts toward the 12% hurdle.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.underwriting import _irr   # reuse the Newton-Raphson / bisection solver


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterfallResult:
    # Initial capital splits
    lp_equity:           float
    gp_equity:           float

    # Full cash flow streams (t=0 outflow through t=10 + exit)
    lp_cash_flows:       tuple[float, ...]
    gp_cash_flows:       tuple[float, ...]

    # Per-party returns
    lp_irr:              float
    gp_irr:              float
    lp_equity_multiple:  float
    gp_equity_multiple:  float
    lp_distributions:    float   # total cash received by LP (excl. initial outflow)
    gp_distributions:    float   # total cash received by GP (excl. initial outflow)

    # Tier totals (LP + GP combined)
    tier1_distributed:   float
    tier2_distributed:   float
    tier3_distributed:   float


# ---------------------------------------------------------------------------
# Main waterfall function
# ---------------------------------------------------------------------------


def run_waterfall(
    equity_cash_flows: list[float],
    lp_fraction:  float = 0.90,
    gp_fraction:  float = 0.10,
    tier1_hurdle: float = 0.08,
    tier2_hurdle: float = 0.12,
) -> WaterfallResult:
    """
    Distribute *equity_cash_flows* through a three-tier IRR hurdle waterfall.

    Parameters
    ----------
    equity_cash_flows : list[float]
        Full equity stream: ``[-equity, lncf_1, ..., lncf_9, lncf_10 + net_exit]``
    lp_fraction : float
        LP share of initial equity commitment (default 0.90).
    gp_fraction : float
        GP share of initial equity commitment (default 0.10).
    tier1_hurdle : float
        Tier 1 / Tier 2 IRR boundary (default 0.08).
    tier2_hurdle : float
        Tier 2 / Tier 3 IRR boundary (default 0.12).

    Returns
    -------
    WaterfallResult
    """
    if abs(lp_fraction + gp_fraction - 1.0) > 1e-9:
        raise ValueError("lp_fraction + gp_fraction must equal 1.0")
    if tier1_hurdle >= tier2_hurdle:
        raise ValueError("tier1_hurdle must be strictly less than tier2_hurdle")
    if equity_cash_flows[0] >= 0:
        raise ValueError("equity_cash_flows[0] must be negative (equity invested)")

    total_equity = -equity_cash_flows[0]
    lp_equity    = lp_fraction * total_equity
    gp_equity    = gp_fraction * total_equity

    # Hurdle balances start at full combined equity and compound each period
    # at their respective rates before distributions are applied.
    bal_t1 = total_equity
    bal_t2 = total_equity

    lp_cfs: list[float] = [-lp_equity]
    gp_cfs: list[float] = [-gp_equity]

    tier1_total = tier2_total = tier3_total = 0.0

    for cf in equity_cash_flows[1:]:
        # Accrue hurdle balances for this period (before distributing)
        bal_t1 = bal_t1 * (1.0 + tier1_hurdle)
        bal_t2 = bal_t2 * (1.0 + tier2_hurdle)

        available = max(0.0, cf)
        lp_recv   = 0.0
        gp_recv   = 0.0

        # ── Tier 1: 90/10 up to 8% hurdle ───────────────────────────────
        if bal_t1 > 0.0 and available > 0.0:
            t1_alloc   = min(available, bal_t1)
            lp_recv   += lp_fraction * t1_alloc
            gp_recv   += gp_fraction * t1_alloc
            bal_t1     = max(0.0, bal_t1 - t1_alloc)
            bal_t2     = max(0.0, bal_t2 - t1_alloc)   # T1 cash counts toward T2 balance
            available -= t1_alloc
            tier1_total += t1_alloc

        # ── Tier 2: 70/30 in the 8%-12% zone ────────────────────────────
        t2_zone = max(0.0, bal_t2 - bal_t1)
        if t2_zone > 0.0 and available > 0.0:
            t2_alloc   = min(available, t2_zone)
            lp_recv   += 0.70 * t2_alloc
            gp_recv   += 0.30 * t2_alloc
            bal_t2     = max(0.0, bal_t2 - t2_alloc)
            available -= t2_alloc
            tier2_total += t2_alloc

        # ── Tier 3: 50/50 above both hurdles ─────────────────────────────
        if available > 0.0:
            lp_recv    += 0.50 * available
            gp_recv    += 0.50 * available
            tier3_total += available

        lp_cfs.append(round(lp_recv, 4))
        gp_cfs.append(round(gp_recv, 4))

    lp_irr  = _irr(lp_cfs)
    gp_irr  = _irr(gp_cfs)
    lp_dist = sum(lp_cfs[1:])
    gp_dist = sum(gp_cfs[1:])

    return WaterfallResult(
        lp_equity          = round(lp_equity,     2),
        gp_equity          = round(gp_equity,     2),
        lp_cash_flows      = tuple(lp_cfs),
        gp_cash_flows      = tuple(gp_cfs),
        lp_irr             = round(lp_irr,        6),
        gp_irr             = round(gp_irr,        6),
        lp_equity_multiple = round(lp_dist / lp_equity, 4) if lp_equity > 0 else float("nan"),
        gp_equity_multiple = round(gp_dist / gp_equity, 4) if gp_equity > 0 else float("nan"),
        lp_distributions   = round(lp_dist,       2),
        gp_distributions   = round(gp_dist,       2),
        tier1_distributed  = round(tier1_total,   2),
        tier2_distributed  = round(tier2_total,   2),
        tier3_distributed  = round(tier3_total,   2),
    )
