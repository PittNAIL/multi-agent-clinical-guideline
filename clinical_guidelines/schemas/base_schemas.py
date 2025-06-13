from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from dataclasses import dataclass, field

class ClinicalRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    PATHOLOGIST = "pathologist"
    RADIOLOGIST = "radiologist"
    SURGEON = "surgeon"
    ONCOLOGIST = "oncologist"
    HUMAN_REVIEWER = "human_reviewer"
    INTAKE = "intake"
    SURVEILLANCE = "surveillance"

class GuidelineMatch(BaseModel):
    """Represents a matched clinical guideline"""
    guideline_id: str
    content: str
    applicable_roles: List[ClinicalRole]

@dataclass
class GuidelineReference:
    """Reference to specific NCCN guideline section"""
    guideline_id: str
    section: str
    version: str
    page: Optional[int] = None
    confidence_score: float = 0.0

@dataclass
class AgentDecision:
    """Decision made by an agent"""
    role: ClinicalRole
    timestamp: datetime
    decision: str
    next_steps: List[str]
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    guideline_references: List[GuidelineReference] = field(default_factory=list)
    confidence_score: float = 0.0
    rationale: str = ""
    alternative_options: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)

class AgentResponse(BaseModel):
    """Represents a response from an agent, including the updated state"""
    decision: AgentDecision
    updated_state: 'PatientState'
    applied_guidelines: List[GuidelineMatch] = []
    next_role: Optional[ClinicalRole] = None

class PatientState(BaseModel):
    """Represents the current state of a patient case"""
    patient_id: str = "PENDING"
    age: Optional[int] = None
    current_symptoms: List[str] = []
    diagnosis: Optional[str] = None
    treatment_phase: str = "initial"
    current_agent: ClinicalRole = ClinicalRole.INTAKE
    last_updated: datetime = datetime.now()
    requires_human_review: bool = False
    
    # Clinical Data
    biopsy_results: Optional[str] = None
    molecular_tests: List[str] = []
    imaging_findings: Optional[str] = None
    imaging_required: bool = False
    surgical_candidate: bool = False
    comorbidities: List[str] = []
    
    # History
    agent_decisions: List[AgentDecision] = []
    clinical_notes: List[Dict[str, Any]] = []
    
    def add_decision(self, decision: AgentDecision):
        """Add a new agent decision to history"""
        self.agent_decisions.append(decision)
        self.last_updated = datetime.now()
        
    def add_clinical_note(self, note: str, source: str):
        """Add a clinical note"""
        self.clinical_notes.append({
            "note": note,
            "source": source,
            "timestamp": datetime.now()
        })
        self.last_updated = datetime.now()

# Add forward reference for AgentResponse
AgentResponse.update_forward_refs() 