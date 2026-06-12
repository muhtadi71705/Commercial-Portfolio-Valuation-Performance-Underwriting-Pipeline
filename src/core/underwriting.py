from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Financial math primitives
# ---------------------------------------------------------------------------


def _amortizing_payment(principal: float, annual_rate: float, amort_years: int) -> float:
    """Monthly P&I payment on a fully-amortizing mortgage."""
    if annual_rate == 0.0:
        return principal / (amort_years * 12)
    r_m = annual_rate / 12
    n   = amort_years * 12
    return principal * (r_m * (1 + r_m) ** n) / ((1 + r_m) ** n - 1)


def _outstanding_balance(
    principal: float,
    annual_rate: float,
    amort_years: int,
    elapsed_years: int,
) -> float:
    """Remaining principal after *elapsed_years* of monthly amortizing payments."""
    if annual_rate == 0.0:
        pmt = principal / (amort_years * 12)
        return principal - pmt * elapsed_years * 12
    r_m = annual_rate / 12
    n   = amort_years * 12
    p   = elapsed_years * 12
    return principal * ((1 + r_m) ** n - (1 + r_m) ** p) / ((1 + r_m) ** n - 1)


def _npv(cash_flows: list[float], rate: float) -> float:
    """Standard discounted cash flow sum; t=0 is the first element."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def _irr(
    cash_flows: list[float],
    guess: float = 0.10,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> float:
    """
    Levered IRR via Newton-Raphson with bisection fallback.

    Expects the canonical equity stream:  [-equity, cf1, cf2, ..., cfN+exit].
    Returns float('nan') if no real root can be bracketed.
    """
    def f(r: float) -> float:
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))

    def df(r: float) -> float:
        return sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cash_flows))

    r = guess
    for _ in range(max_iter):
        fv  = f(r)
        dfv = df(r)
        if abs(dfv) < 1e-14:
            break
        r_new = r - fv / dfv
        if abs(r_new - r) < tol:
            return r_new
        r = max(r_new, -0.9999)   # keep rate in solvable domain

    # Bisection fallback — bracket is wide enough for most CRE scenarios
    lo, hi = -0.50, 5.00
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(300):
        mid = (lo + hi) / 2
        if abs(hi - lo) < tol:
            break
        if f(lo) * f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Lease escalation engine
# ---------------------------------------------------------------------------

_FIXED_RATE_RE = re.compile(r"fixed[- ](\d+(?:\.\d+)?)%", re.IGNORECASE)


@dataclass(frozen=True)
class LeaseInfo:
    """Per-lease descriptor consumed by the pro forma engine."""
    tenant_name:     str
    square_footage:  int
    base_rent_psf:   float
    escalation_type: str = ""
    recovery_type:   str = ""    # NNN | Base-Year-Stop | (empty = gross lease)


def _escalate_rent_psf(
    base_psf:          float,
    escalation_type:   str,
    year:              int,
    general_inflation: float,
    fallback_rate:     float = 0.030,
) -> float:
    """
    Return the rent PSF for *year* given a lease's escalation clause.

    Dispatch table
    --------------
    Fixed-X%     compound at X%/yr from Year 1
    CPI-Linked   compound at general_inflation/yr
    Stepped-Jump flat until Year 4; +$2.00 PSF from Year 5 onward
    None         flat for all years
    (empty)      no explicit clause — fall back to property-level fallback_rate
    """
    esc   = (escalation_type or "").strip()
    lower = esc.lower()

    m = _FIXED_RATE_RE.match(esc)
    if m:
        rate = float(m.group(1)) / 100.0
        return base_psf * (1 + rate) ** (year - 1)

    if lower == "cpi-linked":
        return base_psf * (1 + general_inflation) ** (year - 1)

    if lower == "stepped-jump":
        return base_psf + (2.00 if year >= 5 else 0.0)

    if lower == "none":
        return base_psf

    # Empty string — escalation type not supplied in source data
    return base_psf * (1 + fallback_rate) ** (year - 1)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProFormaYear:
    year:                    int
    rent_psf:                float
    potential_gross_income:  float   # PGI   = base rent GPR
    vacancy_loss:            float   # PGI   × vacancy_rate
    credit_loss:             float   # PGI   × credit_loss_rate
    expense_reimbursement:   float   # NNN / Base-Year-Stop recovery from tenants
    effective_gross_income:  float   # EGI   = PGI - vacancy - credit + reimbursement
    operating_expenses:      float   # total landlord OpEx burden (pre-recovery)
    net_operating_income:    float   # NOI   = EGI - OpEx
    debt_service:            float   # annual P&I on acquisition loan
    levered_net_cash_flow:   float   # LNCF  = NOI - DS


@dataclass(frozen=True)
class DCFResult:
    equity_invested:        float
    equity_cash_flows:      tuple[float, ...]   # Full equity stream t=0..10 (used by waterfall)
    levered_cash_flows:     tuple[float, ...]   # Years 1-10 LNCF
    terminal_noi:           float               # Year 11 NOI (reversion anchor)
    exit_value:             float               # terminal_noi / exit_cap_rate
    loan_balance_at_exit:   float               # outstanding principal at Year 10
    net_exit_proceeds:      float               # exit_value - loan_balance_at_exit
    npv:                    float               # NPV at discount_rate
    irr:                    float               # levered IRR
    equity_multiple:        float               # sum(positive CFs) / equity_invested
    year_1_cap_rate:        float               # going-in cap: Y1 NOI / purchase_price
    debt_yield:             float               # Y1 NOI / loan_amount
    dscr_year_1:            float               # Y1 NOI / annual_debt_service


# ---------------------------------------------------------------------------
# Abstract base asset
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Asset(ABC):
    """
    Base class for all underwritten real estate assets.

    Projection waterfall:
        Base Rent GPR  →  less Vacancy Loss  →  less Credit Loss
                       +   Expense Reimbursement Revenue (subclass hook)
                       =   EGI
                       →   less Total Operating Expenses
                       =   NOI
                       →   less Annual Debt Service
                       =   Levered Net Cash Flow

    Subclasses implement _compute_operating_expenses() for their total OpEx stack
    and _compute_expense_reimbursements() for how much of that OpEx is recovered
    from tenants under NNN or Base-Year-Stop structures.
    """

    # Identity
    property_id:   str
    purchase_price: float

    # Rent roll
    total_sqft:           int
    base_rent_psf:        float
    rent_escalation_rate: float = 0.030   # annual compounding rent bump

    # Revenue haircuts
    vacancy_rate:     float = 0.050
    credit_loss_rate: float = 0.010

    # Capital reserves (used by subclasses; stored here so base helpers can reach it)
    capex_reserve_psf: float = 1.00

    # Inflation / CPI assumption — drives CPI-Linked lease escalations
    general_inflation: float = 0.025

    # Per-lease escalation schedule; when populated, overrides the blended
    # base_rent_psf × rent_escalation_rate approach
    lease_schedule: list[LeaseInfo] = field(default_factory=list)

    # Exit assumption
    exit_cap_rate: float = 0.0   # must be overridden; validated in __post_init__

    # Debt structure
    loan_to_value:      float = 0.65
    debt_interest_rate: float = 0.065
    amortization_years: int   = 30

    # Equity hurdle
    discount_rate: float = 0.090

    def __post_init__(self) -> None:
        if self.purchase_price <= 0:
            raise ValueError("purchase_price must be positive")
        if self.total_sqft <= 0:
            raise ValueError("total_sqft must be positive")
        if self.base_rent_psf <= 0:
            raise ValueError("base_rent_psf must be positive")
        if not (0.0 < self.exit_cap_rate < 1.0):
            raise ValueError("exit_cap_rate must be in (0, 1)")
        if not (0.0 <= self.loan_to_value < 1.0):
            raise ValueError("loan_to_value must be in [0, 1)")

    # ------------------------------------------------------------------
    # Internal helpers — kept private; downstream code uses properties
    # ------------------------------------------------------------------

    @property
    def _loan_amount(self) -> float:
        return self.purchase_price * self.loan_to_value

    @property
    def _equity_invested(self) -> float:
        return self.purchase_price * (1.0 - self.loan_to_value)

    @property
    def _annual_debt_service(self) -> float:
        if self._loan_amount == 0.0:
            return 0.0
        return _amortizing_payment(
            self._loan_amount, self.debt_interest_rate, self.amortization_years
        ) * 12

    def _compute_gpr(self, year: int) -> float:
        """
        Gross Potential Rent for *year*.

        When lease_schedule is populated each tenant's rent is escalated
        independently via _escalate_rent_psf, then summed to a property total.
        When it is empty the legacy blended-PSF path is used, preserving
        backward compatibility for assets built without per-lease data.
        """
        if self.lease_schedule:
            return sum(
                _escalate_rent_psf(
                    lease.base_rent_psf,
                    lease.escalation_type,
                    year,
                    self.general_inflation,
                    self.rent_escalation_rate,
                ) * lease.square_footage
                for lease in self.lease_schedule
            )
        return self.base_rent_psf * (1 + self.rent_escalation_rate) ** (year - 1) * self.total_sqft

    # ------------------------------------------------------------------
    # Expense recovery hook (override in subclasses that support it)
    # ------------------------------------------------------------------

    def _compute_expense_reimbursements(
        self, year: int, total_opex: float, base_year_opex: float
    ) -> float:
        """
        Return the total expense reimbursement revenue collected from tenants.

        Base implementation returns 0.0 (gross lease — landlord absorbs all OpEx).
        CommercialAsset overrides this to implement NNN and Base-Year-Stop structures.

        Parameters
        ----------
        year           : projection year (1-based)
        total_opex     : full operating expenses for this year (pre-recovery)
        base_year_opex : Year 1 operating expenses — the Base-Year-Stop threshold
        """
        return 0.0

    # ------------------------------------------------------------------
    # Abstract hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_operating_expenses(self, year: int, pgi: float, egi: float) -> float:
        """
        Return the total operating expense burden the *landlord* bears in *year*.

        Parameters
        ----------
        year : int
            Projection year (1-based); used for expense escalation factors.
        pgi  : float
            Potential Gross Income for the year.
        egi  : float
            Effective Gross Income after vacancy and credit loss.
        """

    # ------------------------------------------------------------------
    # Pro forma engine
    # ------------------------------------------------------------------

    def generate_10_year_pro_forma(self) -> list[ProFormaYear]:
        """
        Loop through years 1-10, compounding rent at rent_escalation_rate each year
        and applying the landlord's expense structure via _compute_operating_expenses().

        Returns a list of immutable ProFormaYear records — one per projection year.
        Pass this list directly to calculate_dcf().
        """
        ds             = self._annual_debt_service
        base_year_opex = 0.0
        output: list[ProFormaYear] = []

        for yr in range(1, 11):
            pgi      = self._compute_gpr(yr)
            rent_psf = pgi / self.total_sqft

            # Vacancy and credit loss applied to base rent only
            vac      = pgi * self.vacancy_rate
            crd      = pgi * self.credit_loss_rate
            base_egi = pgi - vac - crd

            # OpEx computed on base-rent EGI to avoid circular dependency with
            # the management fee — reimbursements are not subject to the mgmt %
            opex = self._compute_operating_expenses(year=yr, pgi=pgi, egi=base_egi)

            if yr == 1:
                base_year_opex = opex

            # Tenant expense recoveries added on top of base rent revenue
            reimb = self._compute_expense_reimbursements(yr, opex, base_year_opex)
            egi   = base_egi + reimb
            noi   = egi - opex
            lncf  = noi - ds

            output.append(ProFormaYear(
                year                   = yr,
                rent_psf               = round(rent_psf, 4),
                potential_gross_income = round(pgi,      2),
                vacancy_loss           = round(vac,      2),
                credit_loss            = round(crd,      2),
                expense_reimbursement  = round(reimb,    2),
                effective_gross_income = round(egi,      2),
                operating_expenses     = round(opex,     2),
                net_operating_income   = round(noi,      2),
                debt_service           = round(ds,       2),
                levered_net_cash_flow  = round(lncf,     2),
            ))

        return output

    # ------------------------------------------------------------------
    # Year 11 NOI — terminal value anchor
    # ------------------------------------------------------------------

    def _project_year_11_noi(self, _pro_forma: list[ProFormaYear]) -> float:
        """
        Project one additional year beyond the hold period.

        This is used exclusively as the reversion anchor:
            Exit Value = Year 11 NOI / exit_cap_rate

        Expenses are escalated by the same logic as the hold-period years so the
        terminal NOI margin is internally consistent.
        """
        yr       = 11
        pgi      = self._compute_gpr(yr)
        vac      = pgi * self.vacancy_rate
        crd      = pgi * self.credit_loss_rate
        base_egi = pgi - vac - crd
        opex     = self._compute_operating_expenses(year=yr, pgi=pgi, egi=base_egi)
        base_year_opex = _pro_forma[0].operating_expenses if _pro_forma else 0.0
        reimb    = self._compute_expense_reimbursements(yr, opex, base_year_opex)
        return (base_egi + reimb) - opex

    # ------------------------------------------------------------------
    # DCF / terminal value
    # ------------------------------------------------------------------

    def calculate_dcf(self, pro_forma: list[ProFormaYear]) -> DCFResult:
        """
        Compute levered NPV and IRR from a completed 10-year pro forma.

        Terminal value
        --------------
        Exit Value  =  Year 11 NOI / exit_cap_rate
        Loan payoff =  outstanding amortized balance at Year 10
        Net Exit    =  Exit Value − Loan Balance

        Equity cash flow stream  (used for both NPV and IRR)
        -------------------------------------------------------
        t=0      : −equity_invested
        t=1..9   :  levered_net_cash_flow (years 1-9)
        t=10     :  levered_net_cash_flow_year_10 + net_exit_proceeds

        NPV is discounted at discount_rate (levered equity hurdle).
        IRR is the rate that drives NPV to zero (Newton-Raphson + bisection).
        """
        equity     = self._equity_invested
        y11_noi    = self._project_year_11_noi(pro_forma)
        exit_val   = y11_noi / self.exit_cap_rate
        loan_bal   = _outstanding_balance(
            self._loan_amount, self.debt_interest_rate,
            self.amortization_years, elapsed_years=10,
        )
        net_exit   = exit_val - loan_bal
        lncf_list  = [y.levered_net_cash_flow for y in pro_forma]

        # Equity stream: initial outflow then annual levered CFs; exit reversion added to Year 10
        equity_cfs: list[float] = (
            [-equity]
            + lncf_list[:9]
            + [lncf_list[9] + net_exit]
        )

        npv_val  = _npv(equity_cfs, self.discount_rate)
        irr_val  = _irr(equity_cfs)
        y1_noi   = pro_forma[0].net_operating_income
        ds       = self._annual_debt_service

        total_in       = sum(cf for cf in equity_cfs[1:] if cf > 0)
        equity_mult    = total_in / equity if equity > 0 else float("nan")

        return DCFResult(
            equity_invested      = round(equity,       2),
            equity_cash_flows    = tuple(equity_cfs),
            levered_cash_flows   = tuple(lncf_list),
            terminal_noi         = round(y11_noi,      2),
            exit_value           = round(exit_val,     2),
            loan_balance_at_exit = round(loan_bal,     2),
            net_exit_proceeds    = round(net_exit,     2),
            npv                  = round(npv_val,      2),
            irr                  = round(irr_val,      6),
            equity_multiple      = round(equity_mult,  4),
            year_1_cap_rate      = round(y1_noi / self.purchase_price, 6),
            debt_yield           = round(y1_noi / self._loan_amount, 6) if self._loan_amount else float("nan"),
            dscr_year_1          = round(y1_noi / ds, 4) if ds > 0 else float("inf"),
        )


# ---------------------------------------------------------------------------
# Commercial (Triple Net / NNN) asset
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CommercialAsset(Asset):
    """
    Triple Net leased office, retail, or industrial property.

    NNN Expense Reimbursement Logic
    --------------------------------
    Under a true triple-net structure, tenants are responsible for their pro-rata
    share of property taxes, insurance, and common-area maintenance (CAM).  The
    landlord collects these costs from tenants and simultaneously remits them to
    vendors, so they net to zero in the landlord P&L.

    The landlord's *residual* expense burden — the only costs that compress NOI —
    are:
        1. Property management fee   (% of EGI collected)
        2. Discretionary CapEx reserve  (per-SF, escalated for inflation)

    This produces the characteristically high NOI margins of NNN assets relative
    to gross-lease alternatives of identical rent.
    """

    management_fee_rate: float = 0.040   # % of base-rent EGI paid to third-party manager
    expense_growth_rate: float = 0.020   # annual CapEx reserve inflation factor

    def _compute_operating_expenses(self, year: int, pgi: float, egi: float) -> float:
        mgmt_fee   = self.management_fee_rate * egi
        capex_rsrv = (
            self.capex_reserve_psf
            * self.total_sqft
            * (1 + self.expense_growth_rate) ** (year - 1)
        )
        return mgmt_fee + capex_rsrv

    def _compute_expense_reimbursements(
        self, year: int, total_opex: float, base_year_opex: float
    ) -> float:
        """
        Aggregate tenant expense reimbursements for *year*.

        NNN          — tenant pays pro-rata share of total OpEx every year.
        Base-Year-Stop — tenant pays pro-rata share of the OpEx *increase* above
                         the Year 1 baseline; $0 if OpEx has not exceeded baseline.
        All other    — no reimbursement (gross lease treatment).

        Pro-rata share = Tenant SF / Total Asset SF.
        """
        if not self.lease_schedule:
            return 0.0

        reimb = 0.0
        for lease in self.lease_schedule:
            rt       = (lease.recovery_type or "").strip().upper()
            pro_rata = lease.square_footage / self.total_sqft

            if rt == "NNN":
                reimb += pro_rata * total_opex
            elif rt == "BASE-YEAR-STOP":
                reimb += pro_rata * max(0.0, total_opex - base_year_opex)

        return reimb


# ---------------------------------------------------------------------------
# Multifamily asset
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class MultifamilyAsset(Asset):
    """
    Apartment community where the landlord bears the full operating cost stack.

    Expense Model
    -------------
    Operating expenses (taxes, insurance, maintenance, management, utilities,
    and turnover costs) are modeled as a ratio of Year 1 PGI, then escalated
    at expense_growth_rate independently of rent growth.

    Decoupling expense escalation from rent escalation is intentional: on
    older vintage assets or in high-inflation environments, expenses routinely
    outpace rent growth, compressing NOI margins over the hold period.
    """

    operating_expense_ratio: float = 0.40   # Year 1 OpEx as % of Year 1 PGI
    expense_growth_rate:     float = 0.025  # independent annual OpEx escalation
    unit_count:              int   = 0      # optional; surfaced in per-unit metrics

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 < self.operating_expense_ratio < 1.0):
            raise ValueError("operating_expense_ratio must be in (0, 1)")

    def _compute_operating_expenses(self, year: int, pgi: float, egi: float) -> float:
        base_pgi  = self.base_rent_psf * self.total_sqft   # Year 1 PGI — escalation anchor
        base_opex = self.operating_expense_ratio * base_pgi
        return base_opex * (1 + self.expense_growth_rate) ** (year - 1)


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def pro_forma_to_dataframe(pro_forma: list[ProFormaYear]) -> pd.DataFrame:
    """Return a formatted DataFrame suitable for display or export."""
    return pd.DataFrame(
        [
            {
                "Year":                          y.year,
                "Rent PSF ($)":                  y.rent_psf,
                "Potential Gross Income ($)":    y.potential_gross_income,
                "Vacancy Loss ($)":              y.vacancy_loss,
                "Credit Loss ($)":               y.credit_loss,
                "Expense Reimbursement ($)":     y.expense_reimbursement,
                "Effective Gross Income ($)":    y.effective_gross_income,
                "Operating Expenses ($)":        y.operating_expenses,
                "Net Operating Income ($)":      y.net_operating_income,
                "Debt Service ($)":              y.debt_service,
                "Levered Net Cash Flow ($)":     y.levered_net_cash_flow,
            }
            for y in pro_forma
        ]
    ).set_index("Year")


def dcf_summary(result: DCFResult) -> pd.DataFrame:
    """Return a single-column summary DataFrame of key DCF metrics."""
    return pd.DataFrame(
        {
            "Metric": [
                "Equity Invested ($)",
                "Exit Value ($)",
                "Loan Balance at Exit ($)",
                "Net Exit Proceeds ($)",
                "NPV ($)",
                "Levered IRR",
                "Equity Multiple (x)",
                "Year 1 Cap Rate",
                "Debt Yield",
                "DSCR (Year 1)",
            ],
            "Value": [
                f"{result.equity_invested:,.0f}",
                f"{result.exit_value:,.0f}",
                f"{result.loan_balance_at_exit:,.0f}",
                f"{result.net_exit_proceeds:,.0f}",
                f"{result.npv:,.0f}",
                f"{result.irr:.2%}",
                f"{result.equity_multiple:.2f}x",
                f"{result.year_1_cap_rate:.2%}",
                f"{result.debt_yield:.2%}",
                f"{result.dscr_year_1:.2f}x",
            ],
        }
    ).set_index("Metric")
