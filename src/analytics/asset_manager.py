from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.core.underwriting import Asset, ProFormaYear
from src.database.db_manager import (
    Expense,
    Lease,
    Property,
    VIEW_PORTFOLIO_CF_INPUTS,
    VIEW_PROPERTY_SUMMARY,
    _DB_PATH,
    get_engine,
    get_session_factory,
)


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

DSCR_THRESHOLD        = 1.25   # minimum acceptable debt service coverage ratio
DSCR_WARNING_BAND     = 1.40   # soft warning below this, above hard threshold
DELINQUENCY_THRESHOLD = 0.05   # 5 % of contracted rent uncollected
DELINQUENCY_WARNING   = 0.03   # soft warning before hard breach


# ---------------------------------------------------------------------------
# Custom exceptions — raised internally, caught and converted to alert dicts
# ---------------------------------------------------------------------------


class DSCRBreach(Exception):
    """Raised when live NOI / Debt Service falls below DSCR_THRESHOLD."""

    def __init__(self, property_id: str, actual: float, threshold: float) -> None:
        self.property_id = property_id
        self.actual      = actual
        self.threshold   = threshold
        super().__init__(
            f"{property_id}: DSCR {actual:.2f}x is below minimum {threshold:.2f}x"
        )


class DelinquencySpike(Exception):
    """Raised when the share of delinquent contracted rent exceeds DELINQUENCY_THRESHOLD."""

    def __init__(self, property_id: str, actual: float, threshold: float) -> None:
        self.property_id = property_id
        self.actual      = actual
        self.threshold   = threshold
        super().__init__(
            f"{property_id}: delinquency rate {actual:.1%} exceeds {threshold:.1%} threshold"
        )


# ---------------------------------------------------------------------------
# AssetManager
# ---------------------------------------------------------------------------


class AssetManager:
    """
    Portfolio reporting layer that bridges the live database and the underwriting
    pro forma.

    For each managed property it:
      • queries active leases and YTD expenses from the SQLite database
      • annualizes actual cash flows and compares them against the Year-1 budget
        derived from the Asset object's pro forma
      • evaluates two systemic alert conditions —
            DSCRBreach       (actual NOI / DS < 1.25x)
            DelinquencySpike (delinquent rent / gross rent > 5 %)
        by raising the corresponding exception, catching it, and converting it
        into a structured alert dictionary ready for a UI to consume
      • aggregates individual property reports into a portfolio-level summary

    Parameters
    ----------
    assets   : dict mapping property_id → Asset (CommercialAsset or MultifamilyAsset)
    db_path  : path to the SQLite database (defaults to data/secure_vault/portfolio.db)
    """

    def __init__(
        self,
        assets: dict[str, Asset],
        db_path: Path = _DB_PATH,
    ) -> None:
        self._assets   = assets
        self._engine   = get_engine(db_path)
        self._Session  = get_session_factory(self._engine)

    # ------------------------------------------------------------------
    # DB query helpers
    # ------------------------------------------------------------------

    def _fetch_property_row(self, property_id: str) -> Property | None:
        with self._Session() as s:
            return s.get(Property, property_id)

    def _fetch_active_leases(self, property_id: str, as_of: date) -> list[Lease]:
        """Return all leases active on *as_of* (started ≤ today ≤ ended)."""
        with self._Session() as s:
            stmt = (
                select(Lease)
                .where(
                    Lease.property_id == property_id,
                    Lease.lease_start <= str(as_of),
                    Lease.lease_end   >= str(as_of),
                )
                .order_by(Lease.tenant_name)
            )
            return s.execute(stmt).scalars().all()

    def _fetch_annualized_expenses(
        self, property_id: str, year: int, as_of_month: int
    ) -> float:
        """
        Sum operating expenses for *property_id* in *year* and annualize when the
        year is incomplete.

        Annualization logic
        -------------------
        • Rows with expense_month IS NULL are treated as already-annual figures
          and are never scaled.
        • When monthly rows exist and the data covers fewer than 12 months, the
          monthly total is scaled to a full year using the highest month observed
          as the denominator (conservative: assumes costs arrived uniformly).
        • If the dataset already spans all 12 months, the raw sum is used.
        """
        with self._Session() as s:
            stmt = (
                select(Expense)
                .where(
                    Expense.property_id  == property_id,
                    Expense.expense_year == year,
                )
            )
            rows = s.execute(stmt).scalars().all()

        if not rows:
            return 0.0

        annual_total  = sum(r.amount for r in rows if r.expense_month is None)
        monthly_rows  = [r for r in rows if r.expense_month is not None]
        monthly_total = sum(r.amount for r in monthly_rows)

        if not monthly_rows:
            return annual_total

        months_observed = len({r.expense_month for r in monthly_rows})
        if months_observed >= 12:
            annualized_monthly = monthly_total
        else:
            annualized_monthly = monthly_total * (12 / months_observed)

        return annual_total + annualized_monthly

    def _fetch_property_summary(self, property_id: str) -> dict[str, Any] | None:
        """
        Fetch one row from view_property_summary for *property_id*.

        The view pre-aggregates active-lease counts, occupied SF, gross scheduled
        rent, delinquent rent, weighted delinquency rate, and the most recent
        full-year operating expense total — all evaluated against DATE('now').

        Returns None when the property has no database row.
        """
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT * FROM {VIEW_PROPERTY_SUMMARY} WHERE property_id = :pid"),
                {"pid": property_id},
            ).mappings().fetchone()
        return dict(row) if row else None

    def _fetch_portfolio_cf_inputs(
        self, asset_class: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Fetch rows from view_portfolio_cash_flow_inputs.

        Returns all asset classes when *asset_class* is None, or a single-element
        list when filtered.  Each row contains clean-income totals, tenant counts,
        baseline expenses, and a portfolio-level delinquency rate grouped by class.
        """
        with self._engine.connect() as conn:
            if asset_class:
                rows = conn.execute(
                    text(
                        f"SELECT * FROM {VIEW_PORTFOLIO_CF_INPUTS}"
                        " WHERE asset_class = :ac"
                    ),
                    {"ac": asset_class},
                ).mappings().fetchall()
            else:
                rows = conn.execute(
                    text(f"SELECT * FROM {VIEW_PORTFOLIO_CF_INPUTS}")
                ).mappings().fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cash flow computation
    # ------------------------------------------------------------------

    def _compute_actual_cashflows(
        self, property_id: str, asset: Asset, as_of: date
    ) -> dict[str, Any]:
        """
        Derive actual annual cash flows from view_property_summary.

        Revenue and delinquency metrics come directly from the view aggregate,
        eliminating per-lease Python summation.  Operating expenses use the
        most recent full-year total stored in the view (falls back to the
        annualized expenses query when no database row exists).

        Revenue
        -------
        gross_scheduled_rent  — contracted rent from active leases (view)
        delinquent_rent       — share withheld by delinquent tenants (view)
        collected_revenue     = gross_scheduled_rent − delinquent_rent

        Expenses
        --------
        total_operating_expenses — most recent complete year total from view

        NOI / LNCF
        ----------
        net_operating_income  = collected_revenue − operating_expenses
        debt_service          = asset._annual_debt_service  (deterministic)
        levered_net_cash_flow = net_operating_income − debt_service
        """
        summary = self._fetch_property_summary(property_id)

        if summary is not None:
            gross_rent    = float(summary["gross_scheduled_rent"])
            delinq_rent   = float(summary["delinquent_rent"])
            opex          = float(summary["total_operating_expenses"])
            occupied_sqft = int(summary["occupied_sqft"])
            delinq_sqft   = int(summary["delinquent_sqft"])
            tenant_count  = int(summary["active_tenant_count"])
            delinq_count  = int(summary["delinquent_tenant_count"])
        else:
            # Property absent from DB — derive from asset model and raw expense query
            gross_rent    = asset.base_rent_psf * asset.total_sqft
            delinq_rent   = 0.0
            opex          = self._fetch_annualized_expenses(property_id, as_of.year, as_of.month)
            occupied_sqft = asset.total_sqft
            delinq_sqft   = 0
            tenant_count  = len(asset.lease_schedule)
            delinq_count  = 0

        collected_rev  = gross_rent - delinq_rent
        noi            = collected_rev - opex
        ds             = asset._annual_debt_service
        lncf           = noi - ds
        delinq_rate    = (delinq_rent / gross_rent) if gross_rent > 0 else 0.0
        occupancy_rate = (occupied_sqft / asset.total_sqft) if asset.total_sqft > 0 else 0.0
        dscr           = (noi / ds) if ds > 0 else float("inf")

        return {
            "gross_rent_billed":        round(gross_rent,    2),
            "delinquent_rent":          round(delinq_rent,   2),
            "collected_revenue":        round(collected_rev, 2),
            "operating_expenses":       round(opex,          2),
            "net_operating_income":     round(noi,           2),
            "debt_service":             round(ds,            2),
            "levered_net_cash_flow":    round(lncf,          2),
            "dscr":                     round(dscr,          4),
            "delinquency_rate":         round(delinq_rate,   4),
            "occupancy_rate":           round(occupancy_rate,4),
            "total_sqft":               asset.total_sqft,
            "occupied_sqft":            occupied_sqft,
            "delinquent_sqft":          delinq_sqft,
            "total_tenant_count":       tenant_count,
            "delinquent_tenant_count":  delinq_count,
        }

    def _compute_budget_cashflows(self, asset: Asset) -> dict[str, Any]:
        """Extract Year-1 pro forma targets from the Asset underwriting model."""
        pf: list[ProFormaYear] = asset.generate_10_year_pro_forma()
        y1 = pf[0]
        ds = asset._annual_debt_service
        return {
            "gross_rent_billed":    y1.potential_gross_income,
            "effective_gross_income": y1.effective_gross_income,
            "operating_expenses":   y1.operating_expenses,
            "net_operating_income": y1.net_operating_income,
            "debt_service":         y1.debt_service,
            "levered_net_cash_flow":y1.levered_net_cash_flow,
            # assumed rates baked into underwriting
            "dscr":             round(y1.net_operating_income / ds, 4) if ds > 0 else float("inf"),
            "delinquency_rate": asset.credit_loss_rate,
            "occupancy_rate":   round(1.0 - asset.vacancy_rate, 4),
        }

    # ------------------------------------------------------------------
    # Variance helper
    # ------------------------------------------------------------------

    @staticmethod
    def _variance_line(budget: float, actual: float, favorable: str = "up") -> dict:
        """
        Build a budget-vs-actual line item dict.

        Parameters
        ----------
        favorable : "up"   → higher actual is better (revenue lines)
                    "down" → lower actual is better  (expense lines)
        """
        variance     = actual - budget
        variance_pct = round(variance / abs(budget) * 100, 2) if budget != 0 else None
        on_track     = (variance >= 0) if favorable == "up" else (variance <= 0)
        return {
            "budget":       round(budget,   2),
            "actual":       round(actual,   2),
            "variance":     round(variance, 2),
            "variance_pct": variance_pct,
            "on_track":     on_track,
        }

    # ------------------------------------------------------------------
    # Alert constructors
    # ------------------------------------------------------------------

    @staticmethod
    def _make_alert(
        alert_id: str,
        severity: str,
        metric: str,
        threshold: float,
        actual_value: float,
        message: str,
        recommended_action: str,
    ) -> dict:
        return {
            "alert_id":             alert_id,
            "severity":             severity,
            "metric":               metric,
            "threshold":            threshold,
            "actual_value":         round(actual_value, 4),
            "delta_from_threshold": round(actual_value - threshold, 4),
            "message":              message,
            "recommended_action":   recommended_action,
            "triggered_at":         datetime.utcnow().isoformat() + "Z",
        }

    def _evaluate_dscr(
        self, property_id: str, noi: float, debt_service: float
    ) -> list[dict]:
        """
        Raise DSCRBreach if DSCR < DSCR_THRESHOLD; catch it and return an alert list.
        Also emits a softer WARNING when DSCR is below DSCR_WARNING_BAND.
        Returns an empty list when no thresholds are breached.
        """
        alerts: list[dict] = []
        if debt_service <= 0:
            return alerts

        dscr = noi / debt_service

        try:
            if dscr < DSCR_THRESHOLD:
                raise DSCRBreach(property_id, dscr, DSCR_THRESHOLD)
        except DSCRBreach as exc:
            alerts.append(self._make_alert(
                alert_id           = "DSCR_BREACH",
                severity           = "CRITICAL",
                metric             = "dscr",
                threshold          = exc.threshold,
                actual_value       = exc.actual,
                message            = (
                    f"DSCR of {exc.actual:.2f}x is "
                    f"{abs(exc.actual - exc.threshold) / exc.threshold:.1%} below "
                    f"the {exc.threshold:.2f}x minimum coverage threshold."
                ),
                recommended_action = (
                    "Review debt structure, accelerate lease-up, or identify "
                    "operating expense reductions to restore debt coverage."
                ),
            ))
            return alerts   # CRITICAL already set; skip warning check

        if dscr < DSCR_WARNING_BAND:
            alerts.append(self._make_alert(
                alert_id           = "DSCR_WARNING",
                severity           = "WARNING",
                metric             = "dscr",
                threshold          = DSCR_WARNING_BAND,
                actual_value       = dscr,
                message            = (
                    f"DSCR of {dscr:.2f}x is within the warning band "
                    f"({DSCR_THRESHOLD:.2f}x – {DSCR_WARNING_BAND:.2f}x). "
                    f"Coverage is adequate but declining headroom warrants monitoring."
                ),
                recommended_action = (
                    "Monitor rent roll for upcoming lease expirations and track "
                    "expense trends to prevent further DSCR compression."
                ),
            ))

        return alerts

    def _evaluate_delinquency(
        self, property_id: str, delinquency_rate: float
    ) -> list[dict]:
        """
        Raise DelinquencySpike if delinquency > DELINQUENCY_THRESHOLD; catch and
        return an alert list.  Emits a soft WARNING when rate is in the warning band.
        Returns an empty list when no thresholds are breached.
        """
        alerts: list[dict] = []

        try:
            if delinquency_rate > DELINQUENCY_THRESHOLD:
                raise DelinquencySpike(property_id, delinquency_rate, DELINQUENCY_THRESHOLD)
        except DelinquencySpike as exc:
            alerts.append(self._make_alert(
                alert_id           = "DELINQUENCY_SPIKE",
                severity           = "WARNING",
                metric             = "delinquency_rate",
                threshold          = exc.threshold,
                actual_value       = exc.actual,
                message            = (
                    f"Tenant delinquency of {exc.actual:.1%} exceeds the "
                    f"{exc.threshold:.0%} threshold — contracted rent is being "
                    f"withheld by one or more active tenants."
                ),
                recommended_action = (
                    "Initiate demand letters for delinquent tenants, review lease "
                    "cure provisions, and assess credit reserves against uncollected rent."
                ),
            ))
            return alerts

        if delinquency_rate > DELINQUENCY_WARNING:
            alerts.append(self._make_alert(
                alert_id           = "DELINQUENCY_ELEVATED",
                severity           = "INFO",
                metric             = "delinquency_rate",
                threshold          = DELINQUENCY_WARNING,
                actual_value       = delinquency_rate,
                message            = (
                    f"Delinquency of {delinquency_rate:.1%} is elevated "
                    f"(warning band: {DELINQUENCY_WARNING:.0%} – {DELINQUENCY_THRESHOLD:.0%}). "
                    f"No threshold breached yet."
                ),
                recommended_action = (
                    "Flag delinquent tenants for collections follow-up before "
                    "the rate crosses the 5% threshold."
                ),
            ))

        return alerts

    # ------------------------------------------------------------------
    # Public reporting API
    # ------------------------------------------------------------------

    def generate_property_report(
        self,
        property_id: str,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """
        Produce a fully structured property report comparing live DB performance
        against the Year-1 underwriting budget.

        Parameters
        ----------
        property_id : str
            Must match a key in the assets registry passed to __init__.
        as_of       : date, optional
            Snapshot date for active-lease and expense queries. Defaults to today.

        Returns
        -------
        dict with keys:
            property_id, property_name, asset_class, as_of_date, report_timestamp,
            overall_status, alerts, budget_vs_actual, performance_metrics, lease_roll,
            delinquency_detail  (only present when delinquency_rate > 0)

        overall_status is the highest severity across all alerts:
            "OK" → "INFO" → "WARNING" → "CRITICAL"
        """
        if property_id not in self._assets:
            return {
                "property_id":   property_id,
                "error":         f"No Asset registered for property_id '{property_id}'.",
                "overall_status":"ERROR",
                "alerts":        [],
            }

        as_of  = as_of or date.today()
        asset  = self._assets[property_id]
        prop   = self._fetch_property_row(property_id)
        leases = self._fetch_active_leases(property_id, as_of)

        actual = self._compute_actual_cashflows(property_id, asset, as_of)
        budget = self._compute_budget_cashflows(asset)

        # --- Alert evaluation ---
        alerts: list[dict] = []
        alerts += self._evaluate_dscr(property_id, actual["net_operating_income"], actual["debt_service"])
        alerts += self._evaluate_delinquency(property_id, actual["delinquency_rate"])

        severity_rank = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
        overall_status = max(
            (a["severity"] for a in alerts),
            key=lambda s: severity_rank.get(s, 0),
            default="OK",
        )

        # --- Budget vs Actual ---
        bva = {
            "gross_rent_billed":    self._variance_line(budget["gross_rent_billed"],    actual["gross_rent_billed"],    "up"),
            "effective_gross_income":self._variance_line(budget["effective_gross_income"],actual["collected_revenue"],   "up"),
            "operating_expenses":   self._variance_line(budget["operating_expenses"],   actual["operating_expenses"],   "down"),
            "net_operating_income": self._variance_line(budget["net_operating_income"], actual["net_operating_income"], "up"),
            "debt_service":         self._variance_line(budget["debt_service"],         actual["debt_service"],         "down"),
            "levered_net_cash_flow":self._variance_line(budget["levered_net_cash_flow"],actual["levered_net_cash_flow"],"up"),
        }

        # --- Performance metrics ---
        perf = {
            "dscr": {
                "value":     actual["dscr"],
                "budget":    budget["dscr"],
                "threshold": DSCR_THRESHOLD,
                "status":    "BREACH" if actual["dscr"] < DSCR_THRESHOLD
                             else "WARNING" if actual["dscr"] < DSCR_WARNING_BAND
                             else "OK",
            },
            "delinquency_rate": {
                "value":     actual["delinquency_rate"],
                "budget":    budget["delinquency_rate"],
                "threshold": DELINQUENCY_THRESHOLD,
                "status":    "BREACH"  if actual["delinquency_rate"] > DELINQUENCY_THRESHOLD
                             else "WARNING" if actual["delinquency_rate"] > DELINQUENCY_WARNING
                             else "OK",
            },
            "occupancy_rate": {
                "value":  actual["occupancy_rate"],
                "budget": budget["occupancy_rate"],
            },
            "noi_margin": {
                "value":  round(actual["net_operating_income"] / actual["gross_rent_billed"], 4)
                          if actual["gross_rent_billed"] > 0 else None,
                "budget": round(budget["net_operating_income"] / budget["gross_rent_billed"], 4)
                          if budget["gross_rent_billed"] > 0 else None,
            },
            "total_sqft":              actual["total_sqft"],
            "occupied_sqft":           actual["occupied_sqft"],
            "delinquent_sqft":         actual["delinquent_sqft"],
            "active_tenant_count":     actual["total_tenant_count"],
            "delinquent_tenant_count": actual["delinquent_tenant_count"],
        }

        # --- Lease roll ---
        lease_roll = [
            {
                "tenant_name":    l.tenant_name,
                "square_footage": l.square_footage,
                "base_rent_psf":  l.base_rent_psf,
                "annual_rent":    round(l.base_rent_psf * l.square_footage, 2),
                "lease_start":    str(l.lease_start),
                "lease_end":      str(l.lease_end),
                "is_delinquent":  bool(l.is_delinquent),
            }
            for l in leases
        ]

        report: dict[str, Any] = {
            "property_id":      property_id,
            "property_name":    prop.property_name if prop else None,
            "asset_class":      prop.asset_class   if prop else None,
            "as_of_date":       str(as_of),
            "report_timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_status":   overall_status,
            "alerts":           alerts,
            "budget_vs_actual": bva,
            "performance_metrics": perf,
            "lease_roll":       lease_roll,
        }

        # Delinquency detail block — only emitted when delinquency is non-zero
        if actual["delinquency_rate"] > 0:
            delinq_leases = [l for l in leases if l.is_delinquent]
            report["delinquency_detail"] = {
                "delinquent_tenant_count": len(delinq_leases),
                "total_tenant_count":      len(leases),
                "delinquent_rent":         actual["delinquent_rent"],
                "delinquency_rate":        actual["delinquency_rate"],
                "delinquent_tenants": [
                    {
                        "tenant_name":    l.tenant_name,
                        "square_footage": l.square_footage,
                        "annual_rent":    round(l.base_rent_psf * l.square_footage, 2),
                        "lease_end":      str(l.lease_end),
                    }
                    for l in delinq_leases
                ],
            }

        return report

    def generate_portfolio_cf_baseline(
        self,
        asset_class: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return pre-aggregated cash flow input baselines from the database view.

        Reads view_portfolio_cash_flow_inputs, which groups all active,
        non-delinquent income streams and the most recent full-year baseline
        expense records across the entire portfolio by asset class.

        Parameters
        ----------
        asset_class : optional filter; if None all asset classes are returned.

        Returns
        -------
        list of dicts (one per asset class) with keys:
            asset_class, property_count, total_portfolio_sqft,
            active_clean_rent, total_scheduled_rent,
            active_clean_tenant_count, total_active_tenant_count,
            baseline_operating_expenses, portfolio_delinquency_rate
        """
        return self._fetch_portfolio_cf_inputs(asset_class)

    def generate_portfolio_report(
        self,
        property_ids: list[str] | None = None,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate report across all managed properties (or a supplied subset).

        Returns
        -------
        dict with keys:
            report_timestamp, as_of_date, portfolio_summary, property_reports,
            flagged_properties
        """
        as_of      = as_of or date.today()
        target_ids = property_ids or list(self._assets.keys())

        property_reports = [
            self.generate_property_report(pid, as_of) for pid in target_ids
        ]

        total_noi_budget  = sum(
            r["budget_vs_actual"]["net_operating_income"]["budget"]
            for r in property_reports if "budget_vs_actual" in r
        )
        total_noi_actual  = sum(
            r["budget_vs_actual"]["net_operating_income"]["actual"]
            for r in property_reports if "budget_vs_actual" in r
        )
        total_ds          = sum(
            r["budget_vs_actual"]["debt_service"]["actual"]
            for r in property_reports if "budget_vs_actual" in r
        )
        portfolio_dscr    = round(total_noi_actual / total_ds, 4) if total_ds > 0 else float("inf")

        total_delinq_rent  = sum(
            r.get("delinquency_detail", {}).get("delinquent_rent", 0.0)
            for r in property_reports
        )
        total_gross_rent   = sum(
            r["budget_vs_actual"]["gross_rent_billed"]["actual"]
            for r in property_reports if "budget_vs_actual" in r
        )
        portfolio_delinq   = round(total_delinq_rent / total_gross_rent, 4) if total_gross_rent > 0 else 0.0

        flagged = [
            {
                "property_id":   r["property_id"],
                "property_name": r.get("property_name"),
                "overall_status":r["overall_status"],
                "alerts":        r["alerts"],
            }
            for r in property_reports
            if r.get("alerts")
        ]

        severity_rank  = {"OK": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
        portfolio_status = max(
            (r["overall_status"] for r in property_reports),
            key=lambda s: severity_rank.get(s, 0),
            default="OK",
        )

        return {
            "report_timestamp": datetime.utcnow().isoformat() + "Z",
            "as_of_date":       str(as_of),
            "portfolio_summary": {
                "total_properties":        len(property_reports),
                "properties_with_alerts":  len(flagged),
                "total_noi_budget":        round(total_noi_budget,  2),
                "total_noi_actual":        round(total_noi_actual,  2),
                "noi_variance":            round(total_noi_actual - total_noi_budget, 2),
                "total_debt_service":      round(total_ds,           2),
                "portfolio_dscr":          portfolio_dscr,
                "portfolio_delinquency_rate": portfolio_delinq,
                "overall_status":          portfolio_status,
            },
            "property_reports":  property_reports,
            "flagged_properties":flagged,
        }
