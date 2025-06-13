from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from google.adk.tools.tool_context import ToolContext
from google.adk.tools.memory import MemoryConfig
from google.adk.tools.interaction import InteractionConfig

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .base_agent import BaseAgent
from ..schemas.base_schemas import (
    PatientState,
    AgentDecision,
    ClinicalRole,
    GuidelineMatch
)
from ..schemas.agent_card import AgentCapability
from .base_adk_agent import BaseADKAgent

class OncologistAgent(BaseADKAgent):
    """Oncologist specialist agent for bone sarcoma treatment using Google ADK"""
    
    def __init__(
        self,
        guideline_chunks: List[str],
        stream_handler: Optional[Callable[[str], None]] = None,
        memory_config: Optional[MemoryConfig] = None,
        interaction_config: Optional[InteractionConfig] = None
    ):
        super().__init__(
            role=ClinicalRole.ONCOLOGIST,
            guideline_chunks=guideline_chunks,
            name="Oncologist Specialist",
            description="Expert oncologist specializing in bone sarcoma treatment",
            stream_handler=stream_handler,
            memory_config=memory_config,
            interaction_config=interaction_config
        )

    async def validate_input(self, state: PatientState) -> bool:
        """Validate required oncological information"""
        basic_valid = super()._validate_state(state)
        return (
            basic_valid and
            hasattr(state, "diagnosis") and
            state.diagnosis is not None and
            hasattr(state, "treatment_phase") and
            state.treatment_phase is not None
        )

    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Match relevant guidelines based on diagnosis and treatment phase.
        Uses semantic matching to find guidelines that match the clinical context,
        rather than requiring exact keyword matches."""
        relevant_guidelines = []
        
        # Build a clinical context from the state
        clinical_context = {
            "diagnosis": state.diagnosis or "",
            "phase": state.treatment_phase or "",
            "imaging": state.imaging_findings if hasattr(state, "imaging_findings") else "",
            "pathology": state.biopsy_results if hasattr(state, "biopsy_results") else "",
            "symptoms": ", ".join(state.current_symptoms) if state.current_symptoms else ""
        }
        
        for guideline in self.guidelines:
            # Check if guideline's clinical context matches our case
            matches_context = False
            
            # Parse guideline criteria and conditions
            guideline_lower = guideline.lower()
            
            # Check if the guideline's diagnostic criteria align
            if (clinical_context["diagnosis"] and 
                any(term in guideline_lower for term in clinical_context["diagnosis"].lower().split())):
                matches_context = True
            
            # Check if treatment phase aligns
            if (clinical_context["phase"] and 
                clinical_context["phase"].lower() in guideline_lower):
                matches_context = True
                
            # Check if imaging or pathology findings are relevant
            if ((clinical_context["imaging"] and 
                 any(finding.lower() in guideline_lower 
                     for finding in clinical_context["imaging"].lower().split(", "))) or
                (clinical_context["pathology"] and 
                 any(finding.lower() in guideline_lower 
                     for finding in clinical_context["pathology"].lower().split(", ")))):
                matches_context = True
            
            # Check if symptoms are addressed
            if (clinical_context["symptoms"] and 
                any(symptom.lower() in guideline_lower 
                    for symptom in clinical_context["symptoms"].split(", "))):
                matches_context = True
            
            if matches_context:
                relevant_guidelines.append(
                    GuidelineMatch(
                        guideline_id=guideline[:20],  # Use start of guideline as ID
                        content=guideline,
                        applicable_roles=[ClinicalRole.ONCOLOGIST]
                    )
                )
        
        return relevant_guidelines

    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Make treatment decisions strictly based on matched guidelines."""
        if not relevant_guidelines:
            return AgentDecision(
                role=self.role,
                timestamp=datetime.now(),
                decision="No matching guidelines found for this case",
                next_steps=["Escalate to human review"],
                requires_escalation=True,
                escalation_reason="No matching guidelines found",
                supporting_data={}
            )
        
        # Combine all relevant guideline content
        guidelines_text = "\n".join(g.content for g in relevant_guidelines)
        
        # Determine next steps based on treatment phase
        next_steps = []
        if state.treatment_phase == "initial":
            if not state.imaging_findings:
                next_steps.append("Request imaging studies")
            if not state.biopsy_results:
                next_steps.append("Request pathology assessment")
        elif state.treatment_phase == "treatment":
            if not hasattr(state, "treatment_response"):
                next_steps.append("Evaluate treatment response")
        elif state.treatment_phase == "follow_up":
            next_steps.append("Schedule follow-up imaging")
            next_steps.append("Monitor for recurrence")
        
        if not next_steps:
            next_steps = ["Refer to human specialist for guideline interpretation"]
        
        return AgentDecision(
            role=self.role,
            timestamp=datetime.now(),
            decision=f"Following guidelines: {guidelines_text}",
            next_steps=next_steps,
            supporting_data={
                "guidelines_used": [g.guideline_id for g in relevant_guidelines],
                "treatment_phase": state.treatment_phase
            }
        )

    def _get_field_description(self, field: str) -> str:
        """Get description for required oncology fields"""
        descriptions = {
            "diagnosis": "the confirmed diagnosis",
            "treatment_phase": "current phase of treatment (initial/treatment/follow_up)",
            "imaging_findings": "relevant imaging results",
            "biopsy_results": "pathology findings",
            "treatment_response": "response to current treatment"
        }
        return descriptions.get(field, f"the {field.replace('_', ' ')}")

    def _get_capabilities(self) -> List[str]:
        return [
            AgentCapability.TREATMENT_PLANNING.value,
            AgentCapability.CHEMOTHERAPY.value,
            AgentCapability.CLINICAL_TRIALS.value,
            AgentCapability.MONITORING.value
        ]
        
    def _get_capabilities_description(self) -> str:
        return """- Systemic therapy planning and management
- Neoadjuvant and adjuvant chemotherapy
- Clinical trial evaluation and enrollment
- Treatment response monitoring
- Survivorship care planning"""
        
    def _get_constraints(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "treatment_modalities",
                "description": "Available treatment modalities",
                "constraint_type": "domain",
                "value": ["chemotherapy", "immunotherapy", "targeted_therapy", "clinical_trials"]
            },
            {
                "name": "patient_factors",
                "description": "Required patient considerations",
                "constraint_type": "domain",
                "value": ["age", "performance_status", "comorbidities", "prior_treatments"]
            }
        ]
        
    def _get_protocols(self) -> List[Dict[str, Any]]:
        return [{
            "input_format": {"patient_data": "PatientData"},
            "output_format": {"plan": "TreatmentPlan"},
            "communication_pattern": "request-response"
        }]
        
    def _get_required_inputs(self) -> List[str]:
        return [
            "patient_id",
            "diagnosis",
            "stage",
            "performance_status",
            "prior_treatments",
            "molecular_profile"
        ]
        
    def _get_provided_outputs(self) -> List[str]:
        return [
            "treatment_plan",
            "chemotherapy_regimen",
            "clinical_trial_options",
            "monitoring_schedule",
            "toxicity_management"
        ]
        
    def _get_tools(self) -> List[Any]:
        return [
            self.plan_treatment,
            self.evaluate_clinical_trials,
            self.monitor_response,
            self.check_guidelines
        ]
        
    async def plan_treatment(
        self,
        patient_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Plan systemic therapy for bone sarcoma patient.
        
        Args:
            patient_data: Dictionary containing patient information
            tool_context: Tool context
            
        Returns:
            Dictionary containing treatment plan
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not patient_data.get("fertility_discussed"):
            fertility = await self.ask_user(
                "Has fertility preservation been discussed with the patient?",
                tool_context,
                options=["Yes", "No", "Not Applicable"]
            )
            patient_data["fertility_discussed"] = fertility
            
        prompt = f"""Plan systemic therapy for this bone sarcoma patient:
        
        Patient Data:
        {patient_data}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Treatment strategy (neoadjuvant vs adjuvant)
        2. Specific chemotherapy regimen
        3. Duration and cycles
        4. Supportive care needs
        5. Monitoring requirements
        """
        
        # Stream the response
        response_chunks = []
        async for chunk in self.stream_response(prompt, tool_context):
            response_chunks.append(chunk)
            
        # Save to memory
        response_text = "".join(response_chunks)
        self._save_to_memory(
            tool_context.context.state['session_id'],
            response_text
        )
        
        return {
            "treatment_plan": response_text,
            "requires_tumor_board": True if "tumor board" in response_text.lower() else False
        }
        
    async def evaluate_clinical_trials(
        self,
        patient_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Evaluate clinical trial options for the patient.
        
        Args:
            patient_data: Dictionary containing patient information
            tool_context: Tool context
            
        Returns:
            Dictionary containing trial recommendations
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not patient_data.get("travel_capability"):
            travel = await self.ask_user(
                "Is the patient able to travel for trial participation?",
                tool_context,
                options=["Yes", "Limited", "No"]
            )
            patient_data["travel_capability"] = travel
            
        prompt = f"""Evaluate clinical trial options for this patient:
        
        Patient Data:
        {patient_data}
        
        Previous Context:
        {memories}
        
        Consider:
        1. Trial eligibility criteria
        2. Available trials matching profile
        3. Travel and logistics
        4. Alternative standard options
        """
        
        # Stream the response
        response_chunks = []
        async for chunk in self.stream_response(prompt, tool_context):
            response_chunks.append(chunk)
            
        # Save to memory
        response_text = "".join(response_chunks)
        self._save_to_memory(
            tool_context.context.state['session_id'],
            response_text
        )
        
        return {
            "trial_recommendations": response_text,
            "eligible_trials": True if "eligible" in response_text.lower() else False
        }
        
    async def monitor_response(
        self,
        treatment_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Monitor treatment response and manage toxicities.
        
        Args:
            treatment_data: Dictionary containing treatment information
            tool_context: Tool context
            
        Returns:
            Dictionary containing monitoring assessment
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not treatment_data.get("toxicity_grade"):
            toxicity = await self.ask_user(
                "What is the highest grade of toxicity observed?",
                tool_context,
                options=["Grade 1", "Grade 2", "Grade 3", "Grade 4", "None"]
            )
            treatment_data["toxicity_grade"] = toxicity
            
        prompt = f"""Assess treatment response and toxicity:
        
        Treatment Data:
        {treatment_data}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Response assessment
        2. Toxicity management
        3. Dose modifications if needed
        4. Supportive care recommendations
        """
        
        # Stream the response
        response_chunks = []
        async for chunk in self.stream_response(prompt, tool_context):
            response_chunks.append(chunk)
            
        # Save to memory
        response_text = "".join(response_chunks)
        self._save_to_memory(
            tool_context.context.state['session_id'],
            response_text
        )
        
        return {
            "monitoring_assessment": response_text,
            "requires_dose_modification": True if "dose modification" in response_text.lower() else False,
            "toxicity_grade": treatment_data.get("toxicity_grade", "None")
        } 