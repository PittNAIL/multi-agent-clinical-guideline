from typing import Dict, List, Optional, Callable
import uuid
from datetime import datetime

from ..schemas.base_schemas import PatientState, AgentResponse, ClinicalRole
from ..schemas.agent_card import AgentCard, AgentCapability, AgentType
from .base_agent import BaseAgent
from langchain_core.language_models import BaseLLM

class TaskCallback:
    """Callback for task updates from remote agents"""
    def __init__(self, on_update: Callable[[str, AgentResponse], None]):
        self.on_update = on_update

    def __call__(self, task_id: str, response: AgentResponse):
        self.on_update(task_id, response)

class HostAgent(BaseAgent):
    """Host agent that orchestrates communication between specialist agents.
    
    This agent is responsible for:
    1. Managing connections to remote specialist agents
    2. Routing patient cases to appropriate specialists
    3. Coordinating multi-specialist decisions
    4. Maintaining conversation state and history
    """
    
    def __init__(
        self,
        guideline_chunks: List[str],
        llm: Optional[BaseLLM] = None,
        stream_handler: Optional[Callable[[str], None]] = None
    ):
        super().__init__(
            role=ClinicalRole.ORCHESTRATOR,
            guideline_chunks=guideline_chunks,
            llm=llm,
            stream_handler=stream_handler
        )
        self.remote_agents: Dict[str, AgentCard] = {}
        self.active_tasks: Dict[str, str] = {}  # task_id -> agent_id
        self.session_id = str(uuid.uuid4())
        
    def register_agent(self, agent_card: AgentCard) -> None:
        """Register a new remote agent"""
        self.remote_agents[agent_card.agent_id] = agent_card
        
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister a remote agent"""
        if agent_id in self.remote_agents:
            del self.remote_agents[agent_id]
            
    def list_agents(self) -> List[AgentCard]:
        """List all registered remote agents"""
        return list(self.remote_agents.values())
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get orchestrator capabilities"""
        return [AgentCapability.ORCHESTRATION]
        
    def _get_required_inputs(self) -> List[str]:
        """Get required input fields"""
        return ["patient_id", "treatment_phase"]
        
    def _get_provided_outputs(self) -> List[str]:
        """Get provided output fields"""
        return ["orchestration_plan", "next_steps"]
        
    async def validate_input(self, state: PatientState) -> bool:
        """Validate input state"""
        return await self._validate_state(state)
        
    async def process_guidelines(self, state: PatientState) -> List[str]:
        """Process guidelines to determine which specialists to involve"""
        # Use LLM to analyze guidelines and determine required specialists
        prompt = f"""Based on the patient state and guidelines, determine which specialists should be involved.
        
        Patient State:
        {state.model_dump_json(indent=2)}
        
        Guidelines:
        {self.guidelines}
        
        Return the list of required specialists and rationale for each.
        """
        
        response = await self.llm.agenerate([prompt])
        return response.generations[0].text.split("\n")
        
    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[str]
    ) -> AgentResponse:
        """Coordinate decisions between specialists"""
        # Create a new task
        task_id = str(uuid.uuid4())
        
        # Determine required specialists
        specialist_decisions = []
        for agent_card in self.remote_agents.values():
            if self._is_specialist_needed(agent_card, state, relevant_guidelines):
                # Send task to specialist
                response = await self._send_task_to_specialist(
                    task_id,
                    agent_card,
                    state
                )
                specialist_decisions.append(response)
                
        # Synthesize specialist decisions
        final_decision = await self._synthesize_decisions(
            specialist_decisions,
            state,
            relevant_guidelines
        )
        
        return AgentResponse(
            agent_id=self.agent_card.agent_id,
            task_id=task_id,
            timestamp=datetime.now(),
            decision=final_decision,
            confidence_score=0.9,
            specialist_decisions=specialist_decisions
        )
        
    def _is_specialist_needed(
        self,
        agent_card: AgentCard,
        state: PatientState,
        guidelines: List[str]
    ) -> bool:
        """Determine if a specialist is needed based on state and guidelines"""
        # TODO: Implement specialist selection logic
        return True
        
    async def _send_task_to_specialist(
        self,
        task_id: str,
        agent_card: AgentCard,
        state: PatientState
    ) -> AgentResponse:
        """Send a task to a specialist agent"""
        # TODO: Implement remote agent communication
        pass
        
    async def _synthesize_decisions(
        self,
        specialist_decisions: List[AgentResponse],
        state: PatientState,
        guidelines: List[str]
    ) -> str:
        """Synthesize decisions from multiple specialists"""
        # TODO: Implement decision synthesis logic
        pass 