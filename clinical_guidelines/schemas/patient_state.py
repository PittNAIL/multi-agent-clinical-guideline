from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class TreatmentPhase(str, Enum):
    INITIAL = "initial"
    POST_IMAGING = "post_imaging"
    POST_BIOPSY = "post_biopsy"
    POST_DIAGNOSIS = "post_diagnosis"
    PRE_TREATMENT = "pre_treatment"
    IN_TREATMENT = "in_treatment"
    POST_SURGERY = "post_surgery"
    POST_RADIATION = "post_radiation"
    POST_CHEMO = "post_chemo"
    SURVEILLANCE = "surveillance"

class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    SEMI_URGENT = "semi_urgent"
    ROUTINE = "routine"
    FOLLOW_UP = "follow_up"

class AgentDecisionHistory(BaseModel):
    """Record of an agent's decision"""
    timestamp: datetime
    agent_role: str
    decision: str
    next_steps: List[str]
    supporting_data: Dict[str, Any]

class PatientState(BaseModel):
    """Shared state object passed between agents"""
    
    # Basic Information (from Intake)
    patient_id: str = Field(default_factory=lambda: f"P{datetime.now().strftime('%Y%m%d%H%M%S')}")
    age: Optional[int] = None
    gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    current_symptoms: List[str] = Field(default_factory=list)
    symptom_duration: Optional[str] = None
    medical_history: Optional[str] = None
    family_history: Optional[str] = None
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vital_signs: Dict[str, Any] = Field(default_factory=dict)

    # Workflow State
    treatment_phase: TreatmentPhase = TreatmentPhase.INITIAL
    current_agent: Optional[str] = None
    urgency_level: Optional[UrgencyLevel] = None
    requires_human_review: bool = False
    last_updated: datetime = Field(default_factory=datetime.now)

    # Clinical Information
    diagnosis: Optional[str] = None
    differential_diagnoses: List[str] = Field(default_factory=list)
    clinical_notes: List[str] = Field(default_factory=list)
    
    # Imaging Information
    imaging_required: bool = False
    imaging_protocol: Optional[str] = None
    imaging_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Pathology Information
    biopsy_results: Optional[str] = None
    tumor_grade: Optional[str] = None
    molecular_markers: List[str] = Field(default_factory=list)
    molecular_tests: List[str] = Field(default_factory=list)
    
    # Treatment Information
    treatment_plan: Optional[Dict[str, Any]] = None
    surgical_plan: Optional[Dict[str, Any]] = None
    chemotherapy_protocol: Optional[Dict[str, Any]] = None
    radiation_plan: Optional[Dict[str, Any]] = None
    
    # Follow-up Information
    next_follow_up: Optional[datetime] = None
    follow_up_instructions: List[str] = Field(default_factory=list)
    monitoring_parameters: List[str] = Field(default_factory=list)
    
    # Decision History
    agent_decisions: List[AgentDecisionHistory] = Field(default_factory=list)
    clinical_flags: List[str] = Field(default_factory=list)
    
    def update_from_agent(
        self,
        agent_role: str,
        updates: Dict[str, Any],
        decision: Optional[AgentDecisionHistory] = None
    ) -> None:
        """Update state with agent's findings and decisions"""
        # Update fields
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Record decision if provided
        if decision:
            self.agent_decisions.append(decision)
        
        # Update metadata
        self.current_agent = agent_role
        self.last_updated = datetime.now()
    
    def get_agent_history(self, agent_role: Optional[str] = None) -> List[AgentDecisionHistory]:
        """Get decision history, optionally filtered by agent role"""
        if agent_role:
            return [d for d in self.agent_decisions if d.agent_role == agent_role]
        return self.agent_decisions
    
    def add_clinical_flag(self, flag: str) -> None:
        """Add a clinical flag/warning"""
        if flag not in self.clinical_flags:
            self.clinical_flags.append(flag)
    
    def add_clinical_note(self, note: str, agent_role: str) -> None:
        """Add a clinical note with timestamp"""
        timestamped_note = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')} - {agent_role}] {note}"
        self.clinical_notes.append(timestamped_note)
    
    def progress_treatment_phase(self, new_phase: TreatmentPhase) -> None:
        """Progress treatment phase and record the transition"""
        old_phase = self.treatment_phase
        self.treatment_phase = new_phase
        self.add_clinical_note(
            f"Treatment phase progressed from {old_phase} to {new_phase}",
            self.current_agent or "System"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientState":
        """Create state from dictionary"""
        return cls(**data) 