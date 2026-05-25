from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


PositiveInt   = Annotated[int,   Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]


class LeaseRecord(BaseModel):
    property_id:    str
    tenant_name:    str
    square_footage: PositiveInt
    base_rent_psf:  PositiveFloat
    lease_start:    date
    lease_end:      date
    is_delinquent:  bool

    @model_validator(mode="after")
    def _lease_end_after_start(self) -> "LeaseRecord":
        if self.lease_end <= self.lease_start:
            raise ValueError(
                f"lease_end {self.lease_end} must be strictly after lease_start {self.lease_start}"
            )
        return self

    @property
    def lease_term_years(self) -> float:
        return (self.lease_end - self.lease_start).days / 365.25

    @property
    def annual_base_rent(self) -> float:
        return self.square_footage * self.base_rent_psf
