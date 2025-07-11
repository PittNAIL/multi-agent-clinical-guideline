# common/types.py

from typing import List, Optional, Any
from pydantic import BaseModel


class AgentCapabilities(BaseModel):
    streaming: bool = False


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: Optional[List[str]] = []
    examples: Optional[List[str]] = []


class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str
    capabilities: AgentCapabilities
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[AgentSkill]
    custom: Optional[dict[str, Any]] = {}


class TaskRequest(BaseModel):
    id: str
    sessionId: str
    message: dict
    metadata: Optional[dict] = {}
    acceptedOutputModes: Optional[List[str]] = []


class TaskResponse(BaseModel):
    status: str = "completed"
    message: dict
