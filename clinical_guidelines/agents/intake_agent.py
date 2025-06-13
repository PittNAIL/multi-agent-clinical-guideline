from typing import List, Dict, Any, Optional
from datetime import datetime

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

class IntakeAgent(BaseAgent):
    """Special agent for handling initial patient intake"""
    
    # Class-level definition of required information
    required_initial_info = {
        "age": "patient's age in years",
        "gender": "patient's gender",
        "chief_complaint": "main reason for seeking medical attention",
        "current_symptoms": "list of current symptoms (comma-separated)",
        "symptom_duration": "how long symptoms have been present",
        "medical_history": "relevant medical history",
        "family_history": "relevant family history",
        "current_medications": "current medications (if any)",
        "allergies": "known allergies or adverse reactions",
        "vital_signs": "current vital signs if available"
    }
    
    def __init__(
        self,
        guideline_chunks: List[str],
        stream_handler: Optional[callable] = None
    ):
        super().__init__(
            role=ClinicalRole.INTAKE,
            guideline_chunks=guideline_chunks,
            stream_handler=stream_handler
        )

    def _get_field_description(self, field: str) -> str:
        """Get description for intake fields"""
        return self.required_initial_info.get(field, f"the {field.replace('_', ' ')}")

    def _get_capabilities(self) -> List[AgentCapability]:
        """Get intake-specific capabilities"""
        return [
            AgentCapability.DIAGNOSIS  # Initial assessment
        ]

    def _get_required_inputs(self) -> List[str]:
        """Get required input fields"""
        return list(self.required_initial_info.keys())

    def _get_provided_outputs(self) -> List[str]:
        """Get provided output fields"""
        return [
            "initial_assessment",
            "risk_factors",
            "recommended_specialists",
            "urgency_level",
            "preliminary_plan"
        ]

    async def gather_initial_information(self) -> PatientState:
        """Gather all required initial information from user"""
        if not hasattr(self, 'user_input_handler'):
            raise ValueError("No user input handler configured")

        state = PatientState()
        
        # Stream welcome message
        if self.stream_handler:
            await self.stream_handler(
                "Welcome to the Clinical Assessment System.\n"
                "I'll need to gather some initial information about the patient.\n"
                "Please provide the following details:\n\n"
            )

        # Gather information in a structured way
        for field, description in self.required_initial_info.items():
            if self.stream_handler:
                await self.stream_handler(f"\nRequesting {field}...\n")
            
            value = await self.user_input_handler(
                prompt=f"Please provide {description}:",
                field_name=field,
                agent_role=self.role.value
            )
            setattr(state, field, value)

            # Provide feedback
            if self.stream_handler:
                await self.stream_handler(f"Recorded {field}.\n")

        return state

    async def validate_input(self, state: PatientState) -> bool:
        """Validate required intake information"""
        return all(
            hasattr(state, field) and getattr(state, field) is not None
            for field in self.required_initial_info
        )

    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Match initial symptoms and complaints with guidelines"""
        relevant_guidelines = []
        
        for guideline in self.guidelines:
            # Match based on symptoms and chief complaint
            if ((state.chief_complaint and state.chief_complaint.lower() in guideline.lower()) or
                any(symptom.lower() in guideline.lower() 
                    for symptom in state.current_symptoms)):
                relevant_guidelines.append(
                    GuidelineMatch(
                        guideline_id=f"INTAKE_{len(relevant_guidelines)}",
                        relevance_score=0.85,
                        content=guideline,
                        applicable_roles=[ClinicalRole.INTAKE]
                    )
                )
        
        return relevant_guidelines

    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Make initial assessment and specialist recommendations"""
        # Prepare context for decision
        guidelines_text = "\n".join([g.content for g in relevant_guidelines])
        
        # Generate initial assessment prompt
        prompt = f"""Given the following patient information:
        Age: {state.age}
        Gender: {state.gender}
        Chief Complaint: {state.chief_complaint}
        Current Symptoms: {', '.join(state.current_symptoms)}
        Duration: {state.symptom_duration}
        Medical History: {state.medical_history}
        Family History: {state.family_history}
        Current Medications: {state.current_medications}
        Allergies: {state.allergies}
        Vital Signs: {state.vital_signs}
        
        Guidelines to consider:
        {guidelines_text}
        
        Provide an initial assessment with:
        1. Preliminary evaluation
        2. Risk factors identified
        3. Recommended specialists
        4. Urgency level
        5. Initial plan
        
        Format your response as:
        ASSESSMENT: [initial assessment]
        NEXT STEPS:
        - [step 1]
        - [step 2]
        etc."""
        
        # Use LangChain for decision generation
        messages = ChatPromptTemplate.from_messages([
            ("system", "You are an intake specialist performing initial patient assessment."),
            ("user", prompt)
        ]).format_messages()
        
        response = await self.llm.agenerate([m.content for m in messages])
        decision_text = response.generations[0][0].text
        
        # Parse decision and next steps
        decision_parts = decision_text.split("NEXT STEPS:")
        main_decision = decision_parts[0].replace("ASSESSMENT:", "").strip()
        
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
            supporting_data={
                "guidelines_used": [g.guideline_id for g in relevant_guidelines],
                "risk_factors": self._extract_risk_factors(state)
            }
        )

    def _extract_risk_factors(self, state: PatientState) -> List[str]:
        """Extract risk factors from patient state"""
        risk_factors = []
        
        # Age-related risks
        try:
            age = int(state.age)
            if age > 65:
                risk_factors.append("Advanced age")
            elif age < 18:
                risk_factors.append("Pediatric case")
        except (ValueError, TypeError):
            pass

        # Medical history risks
        if state.medical_history:
            risk_factors.extend([
                history.strip()
                for history in state.medical_history.split(",")
                if history.strip()
            ])

        # Family history risks
        if state.family_history:
            risk_factors.extend([
                f"Family history of {history.strip()}"
                for history in state.family_history.split(",")
                if history.strip()
            ])

        return risk_factors 