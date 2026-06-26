"""
Macro sensitivity simulator — parametric stress testing for CRE underwriting.

Clones an initialized Asset model across a grid of vacancy rates and exit cap
rates, runs the full 10-year pro forma + DCF + equity waterfall for every
cross-combination, and returns a structured SensitivityMatrix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.core.underwriting import Asset
from src.core.waterfall import run_waterfall


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityCell:
    """Single scenario result for one (vacancy_rate, exit_cap_rate) pair."""
    vacancy_rate:  float
    exit_cap_rate: float
    combined_irr:  float
    lp_irr:        float
    gp_irr:        float


@dataclass(frozen=True)
class SensitivityMatrix:
    """
    Full cross-product stress test result.

    Cells are stored in row-major order: vacancy outer loop, cap_rate inner loop,
    matching a conventional sensitivity table layout where vacancy rows vary
    top-to-bottom and exit cap rate columns vary left-to-right.
    """
    vacancy_range:  tuple[float, ...]
    cap_rate_range: tuple[float, ...]
    cells:          tuple[SensitivityCell, ...]

    def get(self, vacancy: float, cap_rate: float) -> SensitivityCell | None:
        """Return the cell for (vacancy, cap_rate), or None if not in the grid."""
        for cell in self.cells:
            if (
                abs(cell.vacancy_rate  - vacancy)  < 1e-9
                and abs(cell.exit_cap_rate - cap_rate) < 1e-9
            ):
                return cell
        return None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_VACANCY_RANGE  = [0.05, 0.08, 0.11, 0.14]
_DEFAULT_CAP_RATE_RANGE = [0.0525, 0.0625, 0.0725]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def run_macro_sensitivity_matrix(
    asset:          Asset,
    vacancy_range:  list[float] | None = None,
    cap_rate_range: list[float] | None = None,
) -> SensitivityMatrix:
    """
    Parametric stress test across vacancy and exit cap rate scenarios.

    For every (vacancy_rate, exit_cap_rate) cross-combination:
      1. Clones the asset via dataclasses.replace(), overriding only the two
         stressed parameters — rent schedule, LTV, debt structure, and all other
         assumptions remain identical to the base model.
      2. Runs the full 10-year pro forma and DCF calculation.
      3. Passes the equity cash flow stream through the three-tier equity waterfall.
      4. Records Combined IRR, LP IRR, and GP IRR per scenario cell.

    Parameters
    ----------
    asset          : initialized CommercialAsset or MultifamilyAsset
    vacancy_range  : vacancy rate decimals (default: 5% / 8% / 11% / 14%)
    cap_rate_range : exit cap rates (default: 5.25% / 6.25% / 7.25%)

    Returns
    -------
    SensitivityMatrix — row-major (vacancy outer, cap_rate inner)
    """
    vac_grid = vacancy_range  if vacancy_range  is not None else _DEFAULT_VACANCY_RANGE
    cap_grid = cap_rate_range if cap_rate_range is not None else _DEFAULT_CAP_RATE_RANGE

    cells: list[SensitivityCell] = []

    for v in vac_grid:
        for c in cap_grid:
            clone = replace(asset, vacancy_rate=v, exit_cap_rate=c)
            pf    = clone.generate_10_year_pro_forma()
            dcf   = clone.calculate_dcf(pf)
            wf    = run_waterfall(list(dcf.equity_cash_flows))
            cells.append(SensitivityCell(
                vacancy_rate  = v,
                exit_cap_rate = c,
                combined_irr  = dcf.irr,
                lp_irr        = wf.lp_irr,
                gp_irr        = wf.gp_irr,
            ))

    return SensitivityMatrix(
        vacancy_range  = tuple(vac_grid),
        cap_rate_range = tuple(cap_grid),
        cells          = tuple(cells),
    )
