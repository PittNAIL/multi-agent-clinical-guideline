from typing import List, Dict, Any
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from .base_agent import BaseAgent
from ..schemas.base_schemas import (
    PatientState,
    AgentDecision,
    ClinicalRole,
    GuidelineMatch
)
from ..schemas.agent_card import AgentCapability

class SurveillanceAgent(BaseAgent):
    def __init__(
        self,
        guideline_chunks: List[str],
    ):
        super().__init__(
            role=ClinicalRole.SURVEILLANCE,
            guideline_chunks=guideline_chunks,
        )

    async def validate_input(self, state: PatientState) -> bool:
        """Validate required surveillance information"""
        basic_valid = super()._validate_state(state)
        return (
            basic_valid and
            hasattr(state, "treatment_phase") and
            state.treatment_phase in ["post_surgery", "post_radiation"] and
            hasattr(state, "diagnosis") and
            state.diagnosis is not None
        )

    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Match relevant guidelines based on diagnosis and treatment phase."""
        relevant_guidelines = []
        
        for guideline in self.guidelines:
            # Simple keyword matching - replace with proper semantic search
            if ((state.diagnosis and state.diagnosis.lower() in guideline.lower()) or
                (state.treatment_phase and state.treatment_phase.lower() in guideline.lower())):
                relevant_guidelines.append(
                    GuidelineMatch(
                        guideline_id=f"NCCN_SURV_{len(relevant_guidelines)}",
                        relevance_score=0.85,  # Would be actual similarity score
                        content=guideline,
                        applicable_roles=[ClinicalRole.SURVEILLANCE]
                    )
                )
        
        return relevant_guidelines

    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Make follow-up recommendations based on guidelines."""
        # Prepare context for decision
        guidelines_text = "\n".join([g.content for g in relevant_guidelines])
        
        # Generate surveillance plan prompt
        prompt = f"""Given the following patient information:
        Diagnosis: {state.diagnosis}
        Treatment Phase: {state.treatment_phase}
        Surgical Margins: {getattr(state, "surgical_plan", {}).get("margins", "N/A")}
        
        Guidelines to consider:
        {guidelines_text}
        
        Provide a surveillance plan with:
        1. Follow-up schedule
        2. Imaging requirements
        3. Lab tests needed
        4. Clinical assessment frequency
        5. Long-term monitoring considerations
        
        Format your response as:
        DECISION: [surveillance plan]
        NEXT STEPS:
        - [step 1]
        - [step 2]
        etc."""
        
        # Use LangChain for decision generation
        messages = ChatPromptTemplate.from_messages([
            ("system", "You are a surveillance specialist creating follow-up plans."),
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

    async def process(self, state: PatientState) -> tuple[PatientState, Dict[str, Any]]:
        """Process post-treatment status and create follow-up plan"""
        # Prepare context for guideline checking
        context = {
            "diagnosis": state.diagnosis,
            "treatment_phase": state.treatment_phase,
            "surgical_margins": getattr(state, "surgical_plan", {}).get("margins", None)
        }
        
        # Get relevant guidelines
        guidelines = await self.process_guidelines(state)
        
        # Generate surveillance plan
        prompt = f"""Given the following patient information:
        Diagnosis: {state.diagnosis}
        Treatment Phase: {state.treatment_phase}
        Surgical Margins: {getattr(state, "surgical_plan", {}).get("margins", "N/A")}
        
        Guidelines to consider:
        {guidelines if guidelines else 'No specific guidelines found'}
        
        Provide a surveillance plan with:
        1. Follow-up schedule
        2. Imaging requirements
        3. Lab tests needed
        4. Clinical assessment frequency
        5. Long-term monitoring considerations
        
        Format your response as:
        SCHEDULE: [follow-up timeline]
        IMAGING: [required imaging studies]
        LABS: [required lab tests]
        ASSESSMENTS: [clinical assessment details]
        DURATION: [total surveillance duration]
        RISK_FACTORS: [specific concerns to monitor]"""
        
        decision = await self.make_decision(state, guidelines)
        
        # Parse decision
        lines = decision.decision.split("\n")
        output = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                output[key.strip().lower()] = value.strip()
        
        # Update state
        updates = {
            "surveillance_plan": {
                "schedule": output.get("schedule", "").split(","),
                "imaging": output.get("imaging", "").split(","),
                "labs": output.get("labs", "").split(","),
                "assessments": output.get("assessments", "").split(","),
                "duration": output.get("duration", ""),
                "risk_factors": output.get("risk_factors", "").split(",")
            },
            "treatment_phase": "surveillance",
            "next_follow_up": output.get("schedule", "").split(",")[0] if output.get("schedule") else None
        }
        
        updated_state = self._update_state(state, updates)
        
        return updated_state, {
            "follow_up_plan": {
                "schedule": output.get("schedule", "").split(","),
                "imaging": output.get("imaging", "").split(","),
                "labs": output.get("labs", "").split(",")
            },
            "monitoring_duration": output.get("duration", ""),
            "risk_factors": output.get("risk_factors", "").split(","),
            "next_appointment": updates["next_follow_up"]
        }

    def _get_field_description(self, field: str) -> str:
        """Get description for required surveillance fields"""
        descriptions = {
            "diagnosis": "the confirmed diagnosis being monitored",
            "treatment_phase": "current phase of treatment (post_surgery/post_radiation)",
            "surgical_margins": "status of surgical margins if applicable",
            "treatment_response": "response to previous treatments"
        }
        return descriptions.get(field, f"the {field.replace('_', ' ')}")

    def _get_capabilities(self) -> List[AgentCapability]:
        """Get surveillance-specific capabilities"""
        return [
            AgentCapability.SURVEILLANCE,
            AgentCapability.TREATMENT
        ]

    def _get_required_inputs(self) -> List[str]:
        """Get required input fields"""
        return [
            "diagnosis",
            "treatment_phase",
            "surgical_margins",
            "treatment_response"
        ]

    def _get_provided_outputs(self) -> List[str]:
        """Get provided output fields"""
        return [
            "follow_up_schedule",
            "imaging_requirements",
            "lab_tests",
            "clinical_assessments",
            "monitoring_duration",
            "risk_factors"
        ] 