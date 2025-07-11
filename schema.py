from pydantic import BaseModel, Field
from typing import List, Optional, Any

class PatientInfo(BaseModel):
    name: Optional[str] = Field(None, description="Patient's full name")
    age: Optional[int] = Field(None, description="Patient's age in years")
    sex: Optional[str] = Field(None, description="Patient's sex")
    notes: Optional[str] = Field(None, description="Relevant clinical notes about the patient")

class TreatmentPlan(BaseModel):
    diagnosis: str = Field(..., description="The primary cancer diagnosis")
    cancer_subtype: Optional[str] = Field(None, description="The specific subtype of the cancer, if available")
    location: Optional[str] = Field(None, description="The primary location of the tumor")
    metastatic: bool = Field(False, description="Whether the cancer is metastatic")
    steps: List[str] = Field(..., description="A list of proposed treatment steps")
    version: Optional[str] = Field(None)

class IntakeOutput(BaseModel):
    """Structured output from the IntakeAgent."""
    patient: PatientInfo
    plan: TreatmentPlan

class ReviewResult(BaseModel):
    """Structured output from the PlanReviewAgent."""
    status: str = Field(..., description="The concordance status: 'match', 'partial_match', or 'mismatch'")
    missing_steps: List[str] = Field([], description="A list of guideline steps missing from the patient's plan")
    extra_steps: List[str] = Field([], description="A list of steps in the patient's plan that are not in the guidelines")
    explanation: str = Field(..., description="A clear, concise explanation for the review decision")