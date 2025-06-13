from typing import Dict, Any, List, Type
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from .base_agent import BaseAgent
from .intake_agent import IntakeAgent
from .radiologist_agent import RadiologistAgent
from .pathologist_agent import PathologistAgent
from .oncologist_agent import OncologistAgent
from .surgical_agent import SurgicalAgent
from .surveillance_agent import SurveillanceAgent
from ..schemas.base_schemas import (
    PatientState,
    AgentDecision,
    ClinicalRole,
    GuidelineMatch,
    AgentResponse
)
from ..schemas.agent_card import AgentCapability
from ..core.guideline_loader import GuidelineLoader

class OrchestratorAgent(BaseAgent):
    def __init__(
        self,
        guidelines: List[str]
    ):
        super().__init__(
            role=ClinicalRole.ORCHESTRATOR,
            guideline_chunks=guidelines
        )
        self.specialists: Dict[str, BaseAgent] = {}
        self.guideline_loader = GuidelineLoader()
        self._initialize_specialists()

    def _initialize_specialists(self):
        """Initialize all specialist agents with their specific guidelines"""
        specialist_classes: Dict[str, Type[BaseAgent]] = {
            "intake": IntakeAgent,
            "radiologist": RadiologistAgent,
            "pathologist": PathologistAgent,
            "oncologist": OncologistAgent,
            "surgical": SurgicalAgent,
            "surveillance": SurveillanceAgent
        }
        
        # Load all guidelines
        all_guidelines = self.guideline_loader.load_all_guidelines()
        
        for name, specialist_class in specialist_classes.items():
            # Convert name to match guideline file naming (e.g., "intake" -> "IntakeAgent")
            guideline_name = f"{name.capitalize()}Agent"
            agent_guidelines_dict = all_guidelines.get(guideline_name, {})
            # Convert dictionary of guidelines to list
            agent_guidelines = list(agent_guidelines_dict.values())
            
            self.specialists[name] = specialist_class(
                guideline_chunks=agent_guidelines
            )

    async def validate_input(self, state: PatientState) -> bool:
        """Validate basic patient information"""
        basic_valid = super()._validate_state(state)
        return (
            basic_valid and
            hasattr(state, "current_symptoms") and
            len(state.current_symptoms) > 0
        )

    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Match relevant guidelines for orchestration decisions."""
        relevant_guidelines = []
        
        for guideline in self.guidelines:
            # Simple keyword matching - replace with proper semantic search
            if ((state.treatment_phase and state.treatment_phase.lower() in guideline.lower()) or
                (state.diagnosis and state.diagnosis.lower() in guideline.lower())):
                relevant_guidelines.append(
                    GuidelineMatch(
                        guideline_id=f"ORCH_{len(relevant_guidelines)}",
                        relevance_score=0.85,  # Would be actual similarity score
                        content=guideline,
                        applicable_roles=[ClinicalRole.ORCHESTRATOR]
                    )
                )
        
        return relevant_guidelines

    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Determine next specialist based on guidelines."""
        # Prepare context for decision
        guidelines_text = "\n".join([g.content for g in relevant_guidelines])
        
        # Generate orchestration prompt
        prompt = f"""Given the current patient state:
        Treatment Phase: {getattr(state, "treatment_phase", "initial")}
        Diagnosis: {getattr(state, "diagnosis", "pending")}
        Recent Updates: {', '.join(str(key) for key in state.__dict__ if key not in ['age', 'current_symptoms'])}
        
        Guidelines to consider:
        {guidelines_text}
        
        Determine which specialist should handle the case next.
        Available specialists: intake, radiologist, pathologist, oncologist, surgical, surveillance
        
        Consider:
        1. Current phase of treatment
        2. Pending actions or evaluations
        3. Natural progression of care
        
        Format your response as:
        DECISION: [chosen specialist]
        NEXT STEPS:
        - [reason for choice]
        - [expected actions]"""
        
        # Use LangChain for decision generation
        messages = ChatPromptTemplate.from_messages([
            ("system", "You are an orchestrator coordinating clinical workflow."),
            ("user", prompt)
        ]).format_messages()
        
        response = await self.llm.agenerate([m.content for m in messages])
        decision_text = response.generations[0][0].text
        
        # Parse decision and next steps
        decision_parts = decision_text.split("NEXT STEPS:")
        main_decision = decision_parts[0].replace("DECISION:", "").strip()
        
        next_steps = []
        if len(decision_parts) > 1:
            steps_text = decision_parts[1]
            next_steps = [
                step.strip("- ").strip()
                for step in steps_text.split("\n")
                if step.strip().startswith("-")
            ]
        
        return AgentDecision(
            role=self.role,
            timestamp=datetime.now(),
            decision=main_decision,
            next_steps=next_steps,
            supporting_data={"guidelines_used": [g.guideline_id for g in relevant_guidelines]}
        )

    async def run(self, state: PatientState) -> AgentResponse:
        """Orchestrate the flow between specialists."""
        # First, get the orchestration decision
        response = await super().run(state)
        
        # Get the chosen specialist
        next_specialist = response.decision.decision.lower()
        if next_specialist not in self.specialists:
            raise ValueError(f"Invalid specialist determined: {next_specialist}")
        
        specialist = self.specialists[next_specialist]
        
        # Validate specialist can handle current state
        if not await specialist.validate_input(state):
            raise ValueError(f"State validation failed for {next_specialist}")
        
        # Process with chosen specialist
        specialist_response = await specialist.run(state)
        
        # Update state with specialist's decision
        state = specialist_response.updated_state
        state.last_updated = datetime.now()
        
        return AgentResponse(
            decision=response.decision,
            updated_state=state,
            applied_guidelines=response.applied_guidelines + specialist_response.applied_guidelines,
            next_role=specialist_response.next_role
        )

    def _get_capabilities(self) -> List[AgentCapability]:
        """Get orchestrator-specific capabilities"""
        return [
            AgentCapability.ORCHESTRATION,
            AgentCapability.DIAGNOSIS
        ]

    def _get_required_inputs(self) -> List[str]:
        """Get required input fields"""
        return [
            "current_symptoms",
            "treatment_phase",
            "diagnosis",
            "age"
        ]

    def _get_provided_outputs(self) -> List[str]:
        """Get provided output fields"""
        return [
            "next_specialist",
            "workflow_state",
            "escalation_status",
            "coordination_plan"
        ]

    def _get_field_description(self, field: str) -> str:
        """Get description for a required field"""
        descriptions = {
            "current_symptoms": "Current symptoms the patient is experiencing",
            "treatment_phase": "Current phase of treatment (e.g., initial, ongoing, follow-up)",
            "diagnosis": "Current diagnosis if available",
            "age": "Patient's age"
        }
        return descriptions.get(field, f"Value for {field}") 