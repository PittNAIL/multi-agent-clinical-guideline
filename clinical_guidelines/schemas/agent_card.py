from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum

class AgentType(str, Enum):
    """Represents different types of agents"""
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    REVIEWER = "reviewer"
    ASSISTANT = "assistant"

class AgentCapability(str, Enum):
    """Represents different capabilities an agent can have"""
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    IMAGING = "imaging"
    PATHOLOGY = "pathology"
    SURGERY = "surgery"
    REVIEW = "review"
    ORCHESTRATION = "orchestration"
    SURVEILLANCE = "surveillance"

class AgentConstraint(BaseModel):
    type: str
    description: str
    threshold: Optional[float] = None

class AgentProtocol(BaseModel):
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    communication_pattern: str

class ModelInfo(BaseModel):
    """Information about the underlying model used by the agent"""
    name: str
    provider: str
    version: str
    capabilities: List[str]
    limitations: List[str]
    performance_metrics: Dict[str, float]

class AgentCard(BaseModel):
    """Represents an agent's capabilities and requirements"""
    name: str
    role: str
    capabilities: List[AgentCapability]
    required_inputs: List[str]
    provided_outputs: List[str]
    description: Optional[str] = None
    metadata: Dict[str, str] = {}
    agent_id: str
    version: str
    constraints: List[AgentConstraint]
    protocols: List[AgentProtocol]
    performance_metrics: Dict[str, float]
    trust_score: float
    specialization: str
    interaction_patterns: List[str]
    agent_type: Optional[AgentType] = None
    model_info: Optional[ModelInfo] = None 