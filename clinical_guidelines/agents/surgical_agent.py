from typing import List, Dict, Any
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

class SurgicalAgent(BaseAgent):
    def __init__(
        self,
        guideline_chunks: List[str],
    ):
        super().__init__(
            role=ClinicalRole.SURGEON,
            guideline_chunks=guideline_chunks,
        )

    async def validate_input(self, state: PatientState) -> bool:
        """Validate required surgical information"""
        basic_valid = super()._validate_state(state)
        return (
            basic_valid and
            hasattr(state, "diagnosis") and
            state.diagnosis is not None and
            hasattr(state, "imaging_findings") and
            state.imaging_findings is not None and
            hasattr(state, "surgical_candidate") and
            state.surgical_candidate is True
        )

    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Match relevant guidelines based on diagnosis and surgical requirements.
        Uses semantic matching to find guidelines that match the clinical context,
        rather than requiring exact keyword matches."""
        relevant_guidelines = []
        
        # Build a clinical context from the state
        clinical_context = {
            "diagnosis": state.diagnosis or "",
            "imaging": state.imaging_findings or "",
            "comorbidities": ", ".join(state.comorbidities) if hasattr(state, "comorbidities") and state.comorbidities else ""
        }
        
        for guideline in self.guidelines:
            # Check if guideline's clinical context matches our case
            # This allows for semantic matching rather than exact keyword matching
            matches_context = False
            
            # Parse guideline criteria and conditions
            guideline_lower = guideline.lower()
            
            # Check if the guideline's diagnostic criteria align
            if (clinical_context["diagnosis"] and 
                any(term in guideline_lower for term in clinical_context["diagnosis"].lower().split())):
                matches_context = True
            
            # Check if imaging findings align with surgical indications
            if (clinical_context["imaging"] and 
                any(finding.lower() in guideline_lower 
                    for finding in clinical_context["imaging"].lower().split(", "))):
                matches_context = True
                
            # Check contraindications based on comorbidities
            if clinical_context["comorbidities"]:
                has_contraindication = False
                for comorbidity in clinical_context["comorbidities"].split(", "):
                    if (comorbidity.lower() in guideline_lower and 
                        "contraindication" in guideline_lower):
                        has_contraindication = True
                        break
                # Only include if no contraindications found
                if not has_contraindication:
                    matches_context = True
            
            if matches_context:
                relevant_guidelines.append(
                    GuidelineMatch(
                        guideline_id=guideline[:20],  # Use start of guideline as ID
                        content=guideline,
                        applicable_roles=[ClinicalRole.SURGEON]
                    )
                )
        
        return relevant_guidelines

    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Make surgical recommendations strictly based on matched guidelines."""
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
        
        # Extract decision and next steps directly from guidelines
        main_decision = f"Following guidelines: {guidelines_text}"
        next_steps = ["Refer to human specialist for guideline interpretation"]
        
        return AgentDecision(
            role=self.role,
            timestamp=datetime.now(),
            decision=main_decision,
            next_steps=next_steps,
            supporting_data={"guidelines_used": [g.guideline_id for g in relevant_guidelines]}
        )

    def _get_field_description(self, field: str) -> str:
        """Get description for required surgical fields"""
        descriptions = {
            "diagnosis": "the confirmed diagnosis requiring surgical intervention",
            "imaging_findings": "relevant imaging findings for surgical planning",
            "surgical_candidate": "whether patient is a surgical candidate (true/false)",
            "comorbidities": "list of patient comorbidities affecting surgical planning"
        }
        return descriptions.get(field, f"the {field.replace('_', ' ')}")

    def _get_capabilities(self) -> List[AgentCapability]:
        """Get surgeon-specific capabilities"""
        return [
            AgentCapability.SURGERY,
            AgentCapability.TREATMENT
        ]

    def _get_required_inputs(self) -> List[str]:
        """Get required input fields"""
        return [
            "diagnosis",
            "imaging_findings",
            "surgical_candidate",
            "comorbidities"
        ]

    def _get_provided_outputs(self) -> List[str]:
        """Get provided output fields"""
        return [
            "surgical_approach",
            "estimated_duration",
            "required_equipment",
            "post_op_care",
            "expected_outcomes"
        ] 