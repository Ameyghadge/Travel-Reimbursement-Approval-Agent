"""Pydantic models for claim input."""



from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ExpenseCategory(str, Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"
    MEALS = "meals"
    TRANSPORT = "transport"
    CONFERENCE = "conference"
    MISCELLANEOUS = "miscellaneous"


class ReceiptInfo(BaseModel):
    has_receipt: bool
    receipt_format: Optional[str] = None
    receipt_amount_matches: Optional[bool] = None


class ExpenseItem(BaseModel):
    date: date
    category: ExpenseCategory
    amount: float = Field(gt=0)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    vendor: str = Field(min_length=1)
    description: str = ""
    receipt: ReceiptInfo


class EmployeeInfo(BaseModel):
    employee_id: str
    name: str
    department: str
    level: str


class TripDetails(BaseModel):
    destination: str
    purpose: str
    start_date: date
    end_date: date
    is_international: bool = False


class ClaimRequest(BaseModel):
    claim_id: str
    employee: EmployeeInfo
    trip: TripDetails
    expenses: List[ExpenseItem] = Field(min_length=1)
    total_amount: float = Field(gt=0)
    notes: str = ""
    submitted_at: str
