from typing import Dict, List, Optional, Callable, Any
import uuid
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

from ..schemas.base_schemas import PatientState, AgentResponse, ClinicalRole, GuidelineMatch
from ..schemas.agent_card import AgentCard, AgentCapability, AgentType, AgentConstraint, AgentProtocol
from .base_agent import BaseAgent
from langchain_core.language_models import BaseLLM

class RemoteSpecialistAgent(BaseAgent):
    """Base class for remote specialist agents.
    
    Each specialist agent:
    1. Has specific capabilities and constraints
    2. Can process patient cases independently
    3. Can communicate with the host agent
    4. Can access and interpret relevant guidelines
    5. Can make specialist-specific decisions
    """
    
    def __init__(
        self,
        role: ClinicalRole,
        guideline_chunks: List[str],
        llm: Optional[BaseLLM] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
        host_url: Optional[str] = None
    ):
        super().__init__(
            role=role,
            guideline_chunks=guideline_chunks,
            llm=llm,
            stream_handler=stream_handler
        )
        self.host_url = host_url
        self.active_tasks: Dict[str, Any] = {}
        
    def _create_agent_card(self) -> AgentCard:
        """Create the agent's capability card"""
        return AgentCard(
            agent_id=f"{self.role.value}-{uuid.uuid4()}",
            name=f"{self.role.value.capitalize()} Specialist",
            role=self.role.value,
            description=self._get_description(),
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=self._get_capabilities(),
            constraints=self._get_constraints(),
            protocols=[
                AgentProtocol(
                    input_format={"patient_state": "PatientState"},
                    output_format={"decision": "AgentResponse"},
                    communication_pattern="request-response"
                )
            ],
            required_inputs=self._get_required_inputs(),
            provided_outputs=self._get_provided_outputs(),
            performance_metrics={
                "accuracy": 0.9,
                "response_time": 2.0
            },
            trust_score=0.85,
            specialization=self.role.value
        )
        
    @abstractmethod
    def _get_description(self) -> str:
        """Get specialist-specific description"""
        pass
        
    @abstractmethod
    def _get_constraints(self) -> List[AgentConstraint]:
        """Get specialist-specific constraints"""
        pass
        
    async def process_task(
        self,
        task_id: str,
        state: PatientState
    ) -> AgentResponse:
        """Process a task from the host agent"""
        # Validate input
        if not await self.validate_input(state):
            return AgentResponse(
                agent_id=self.agent_card.agent_id,
                task_id=task_id,
                timestamp=datetime.now(),
                decision="Invalid input state",
                confidence_score=0.0,
                requires_human_review=True,
                human_review_reason="Invalid input state"
            )
            
        # Process guidelines
        relevant_guidelines = await self.process_guidelines(state)
        
        # Make specialist decision
        decision = await self.make_decision(state, relevant_guidelines)
        
        # Store task result
        self.active_tasks[task_id] = decision
        
        return decision
        
    async def handle_update(
        self,
        task_id: str,
        update: Dict[str, Any]
    ) -> AgentResponse:
        """Handle an update from the host agent"""
        if task_id not in self.active_tasks:
            return AgentResponse(
                agent_id=self.agent_card.agent_id,
                task_id=task_id,
                timestamp=datetime.now(),
                decision="Unknown task",
                confidence_score=0.0,
                requires_human_review=True,
                human_review_reason="Unknown task ID"
            )
            
        # Process the update
        state = PatientState(**update.get("state", {}))
        return await self.process_task(task_id, state)
        
    async def start_server(self, host: str = "localhost", port: int = 8000):
        """Start the agent's server to receive requests"""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        app = FastAPI()
        
        class TaskRequest(BaseModel):
            task_id: str
            state: Dict[str, Any]
            
        @app.post("/process_task")
        async def process_task_endpoint(request: TaskRequest):
            try:
                state = PatientState(**request.state)
                response = await self.process_task(request.task_id, state)
                return response.dict()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.get("/agent_card")
        async def get_agent_card():
            return self.agent_card.dict()
            
        import uvicorn
        await uvicorn.run(app, host=host, port=port) 