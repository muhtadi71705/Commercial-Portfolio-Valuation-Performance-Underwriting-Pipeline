from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    event,
    text,
)
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "secure_vault" / "portfolio.db"

# ---------------------------------------------------------------------------
# ORM base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------


class Property(Base):
    """One row per investable asset (building / portfolio property)."""

    __tablename__ = "properties"

    property_id:          Mapped[str]            = mapped_column(String,  primary_key=True)
    property_name:        Mapped[str]            = mapped_column(String,  nullable=False)
    address:              Mapped[Optional[str]]  = mapped_column(String)
    city:                 Mapped[Optional[str]]  = mapped_column(String)
    state:                Mapped[Optional[str]]  = mapped_column(String(2))
    asset_class:          Mapped[str]            = mapped_column(String,  nullable=False)
    total_sqft:           Mapped[Optional[int]]  = mapped_column(Integer)
    year_built:           Mapped[Optional[int]]  = mapped_column(Integer)
    acquisition_date:     Mapped[Optional[date]] = mapped_column(Date)
    acquisition_price:    Mapped[Optional[float]]= mapped_column(Float)
    target_exit_cap_rate: Mapped[float]          = mapped_column(Float,   nullable=False)
    created_at:           Mapped[datetime]       = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:           Mapped[datetime]       = mapped_column(DateTime, default=datetime.utcnow,
                                                                  onupdate=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("target_exit_cap_rate > 0", name="ck_properties_cap_rate_positive"),
        CheckConstraint(
            "asset_class IN ('office','retail','multifamily','industrial','mixed_use','hotel','land')",
            name="ck_properties_asset_class_valid",
        ),
    )

    leases:   Mapped[list[Lease]]   = relationship("Lease",   back_populates="property",
                                                    cascade="all, delete-orphan")
    expenses: Mapped[list[Expense]] = relationship("Expense", back_populates="property",
                                                    cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Property {self.property_id} | {self.asset_class} | cap={self.target_exit_cap_rate:.2%}>"


class Lease(Base):
    """Standardized tenant / rent records (maps 1-to-1 with LeaseRecord schema)."""

    __tablename__ = "leases"

    lease_id:       Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    property_id:    Mapped[str]   = mapped_column(String,  ForeignKey("properties.property_id",
                                                                       ondelete="CASCADE"), nullable=False)
    tenant_name:    Mapped[str]   = mapped_column(String,  nullable=False)
    square_footage: Mapped[int]   = mapped_column(Integer, nullable=False)
    base_rent_psf:  Mapped[float] = mapped_column(Float,   nullable=False)
    lease_start:    Mapped[date]  = mapped_column(Date,    nullable=False)
    lease_end:      Mapped[date]  = mapped_column(Date,    nullable=False)
    is_delinquent:  Mapped[bool]  = mapped_column(Boolean, nullable=False, default=False)
    created_at:     Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:     Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                      onupdate=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("square_footage > 0", name="ck_leases_sqft_positive"),
        CheckConstraint("base_rent_psf > 0",  name="ck_leases_rent_positive"),
        CheckConstraint("lease_end > lease_start", name="ck_leases_dates_ordered"),
        Index("ix_leases_property_id", "property_id"),
        Index("ix_leases_lease_start",  "lease_start"),
        Index("ix_leases_is_delinquent","is_delinquent"),
    )

    property: Mapped[Property] = relationship("Property", back_populates="leases")

    def __repr__(self) -> str:
        return f"<Lease {self.lease_id} | {self.property_id} | {self.tenant_name} | {self.lease_start}–{self.lease_end}>"


class Expense(Base):
    """Historical operating costs at the property level."""

    __tablename__ = "expenses"

    expense_id:   Mapped[int]           = mapped_column(Integer, primary_key=True, autoincrement=True)
    property_id:  Mapped[str]           = mapped_column(String,  ForeignKey("properties.property_id",
                                                                             ondelete="CASCADE"), nullable=False)
    expense_year: Mapped[int]           = mapped_column(Integer, nullable=False)
    expense_month:Mapped[Optional[int]] = mapped_column(Integer)
    category:     Mapped[str]           = mapped_column(String,  nullable=False)
    amount:       Mapped[float]         = mapped_column(Float,   nullable=False)
    description:  Mapped[Optional[str]] = mapped_column(String)
    created_at:   Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:   Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow,
                                                         onupdate=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("amount >= 0", name="ck_expenses_amount_nonneg"),
        CheckConstraint("expense_month IS NULL OR (expense_month >= 1 AND expense_month <= 12)",
                        name="ck_expenses_month_range"),
        CheckConstraint(
            "category IN ('taxes','insurance','repairs_maintenance','management_fees',"
            "'utilities','capital_expenditures','other')",
            name="ck_expenses_category_valid",
        ),
        Index("ix_expenses_property_id",   "property_id"),
        Index("ix_expenses_year_month",    "expense_year", "expense_month"),
        Index("ix_expenses_category",      "category"),
    )

    property: Mapped[Property] = relationship("Property", back_populates="expenses")

    def __repr__(self) -> str:
        period = f"{self.expense_year}-{self.expense_month:02d}" if self.expense_month else str(self.expense_year)
        return f"<Expense {self.expense_id} | {self.property_id} | {self.category} | {period} | ${self.amount:,.2f}>"


# ---------------------------------------------------------------------------
# Required columns per table (subset that must be present in any DataFrame insert)
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "properties": {"property_id", "property_name", "asset_class", "target_exit_cap_rate"},
    "leases":     {"property_id", "tenant_name", "square_footage",
                   "base_rent_psf", "lease_start", "lease_end", "is_delinquent"},
    "expenses":   {"property_id", "category", "amount", "expense_year"},
}

_VALID_TABLES = set(_REQUIRED_COLUMNS.keys())


# ---------------------------------------------------------------------------
# Engine / session factory
# ---------------------------------------------------------------------------


def get_engine(db_path: Path = _DB_PATH) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # SQLite disables FK enforcement by default; enable it per connection.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")
        dbapi_conn.execute("PRAGMA journal_mode = WAL")

    return engine


def get_session_factory(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_db(db_path: Path = _DB_PATH) -> Engine:
    """Create all tables (no-op if they already exist) and return the engine."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# DataFrame loader
# ---------------------------------------------------------------------------


def load_dataframe_to_db(
    df: pd.DataFrame,
    table_name: str,
    db_path: Path = _DB_PATH,
    if_exists: str = "append",
) -> int:
    """
    Append a validated Pandas DataFrame into *table_name*.

    Returns the number of rows written.

    Raises
    ------
    ValueError
        If table_name is unknown or required columns are missing from *df*.
    """
    if table_name not in _VALID_TABLES:
        raise ValueError(
            f"Unknown table '{table_name}'. Valid tables: {sorted(_VALID_TABLES)}"
        )

    required = _REQUIRED_COLUMNS[table_name]
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns for '{table_name}': {sorted(missing)}"
        )

    df = df.copy()

    # Stamp timestamps if absent so SQLite gets explicit values rather than NULL.
    now = datetime.utcnow()
    if "created_at" not in df.columns:
        df["created_at"] = now
    if "updated_at" not in df.columns:
        df["updated_at"] = now

    # Normalise date columns to ISO strings (SQLite TEXT affinity for Date).
    for col in ("lease_start", "lease_end", "acquisition_date", "expense_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date.astype(str)

    # Normalise boolean to int for SQLite (0/1).
    if "is_delinquent" in df.columns:
        df["is_delinquent"] = df["is_delinquent"].astype(bool).astype(int)

    engine = get_engine(db_path)

    # Ensure schema exists before writing.
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists=if_exists,
            index=False,
            method="multi",
        )

    return len(df)
