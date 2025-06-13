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

class PathologistAgent(BaseADKAgent):
    """Pathologist specialist agent for bone sarcoma tissue analysis using Google ADK"""
    
    def __init__(
        self,
        guideline_chunks: List[str],
        stream_handler: Optional[Callable[[str], None]] = None,
        memory_config: Optional[MemoryConfig] = None,
        interaction_config: Optional[InteractionConfig] = None
    ):
        super().__init__(
            role=ClinicalRole.PATHOLOGIST,
            guideline_chunks=guideline_chunks,
            name="Pathologist Specialist",
            description="Expert pathologist specializing in bone sarcoma tissue analysis",
            stream_handler=stream_handler,
            memory_config=memory_config,
            interaction_config=interaction_config
        )

    async def validate_input(self, state: PatientState) -> bool:
        """Validate required pathological information"""
        basic_valid = super()._validate_state(state)
        return (
            basic_valid and
            hasattr(state, "biopsy_results") and
            state.biopsy_results is not None and
            hasattr(state, "molecular_tests") and
            isinstance(state.molecular_tests, list)
        )

    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Match relevant guidelines based on biopsy and molecular results.
        Uses semantic matching to find guidelines that match the clinical context,
        rather than requiring exact keyword matches."""
        relevant_guidelines = []
        
        # Build a clinical context from the state
        clinical_context = {
            "biopsy": state.biopsy_results or "",
            "molecular_tests": ", ".join(state.molecular_tests) if state.molecular_tests else "",
            "clinical_concern": state.clinical_concern if hasattr(state, "clinical_concern") else ""
        }
        
        for guideline in self.guidelines:
            # Check if guideline's clinical context matches our case
            # This allows for semantic matching rather than exact keyword matching
            matches_context = False
            
            # Parse guideline criteria and conditions
            guideline_lower = guideline.lower()
            
            # Check if the guideline's diagnostic criteria align with our findings
            if (clinical_context["biopsy"] and 
                any(term in guideline_lower for term in clinical_context["biopsy"].lower().split())):
                matches_context = True
            
            # Check if molecular test types or results align
            if (clinical_context["molecular_tests"] and 
                any(test.lower() in guideline_lower for test in clinical_context["molecular_tests"].split(", "))):
                matches_context = True
                
            # Check clinical concern alignment
            if (clinical_context["clinical_concern"] and 
                any(term in guideline_lower for term in clinical_context["clinical_concern"].lower().split())):
                matches_context = True
            
            if matches_context:
                relevant_guidelines.append(
                    GuidelineMatch(
                        guideline_id=guideline[:20],  # Use start of guideline as ID
                        content=guideline,
                        applicable_roles=[ClinicalRole.PATHOLOGIST]
                    )
                )
        
        return relevant_guidelines

    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Make pathological assessment strictly based on matched guidelines."""
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
        """Get description for required pathology fields"""
        descriptions = {
            "biopsy_results": "the initial biopsy findings or pathology report",
            "molecular_tests": "list of molecular tests performed or requested",
            "clinical_concern": "the primary clinical concern or reason for pathology review"
        }
        return descriptions.get(field, f"the {field.replace('_', ' ')}")

    def _get_capabilities(self) -> List[str]:
        return [
            AgentCapability.DIAGNOSIS.value,
            AgentCapability.TISSUE_ANALYSIS.value,
            AgentCapability.MOLECULAR_TESTING.value
        ]
        
    def _get_capabilities_description(self) -> str:
        return """- Histopathological analysis of tissue samples
- Molecular and genetic testing interpretation
- Tumor grading and classification
- Biomarker assessment
- Treatment response evaluation"""
        
    def _get_constraints(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "sample_types",
                "description": "Limited to bone and soft tissue samples",
                "constraint_type": "domain",
                "value": ["biopsy", "resection", "fine-needle-aspirate"]
            },
            {
                "name": "testing_methods",
                "description": "Available molecular testing methods",
                "constraint_type": "domain",
                "value": ["IHC", "FISH", "NGS", "PCR"]
            }
        ]
        
    def _get_protocols(self) -> List[Dict[str, Any]]:
        return [{
            "input_format": {"sample_data": "SampleData"},
            "output_format": {"analysis": "PathologyReport"},
            "communication_pattern": "request-response"
        }]
        
    def _get_required_inputs(self) -> List[str]:
        return [
            "sample_id",
            "sample_type",
            "collection_date",
            "anatomical_site",
            "clinical_history"
        ]
        
    def _get_provided_outputs(self) -> List[str]:
        return [
            "histological_diagnosis",
            "tumor_grade",
            "molecular_findings",
            "biomarker_status",
            "margin_status"
        ]
        
    def _get_tools(self) -> List[Any]:
        return [
            self.analyze_sample,
            self.interpret_molecular_tests,
            self.assess_margins,
            self.check_guidelines
        ]
        
    async def analyze_sample(
        self,
        sample_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Analyze tissue sample and provide pathological findings.
        
        Args:
            sample_data: Dictionary containing sample information
            tool_context: Tool context
            
        Returns:
            Dictionary containing analysis results
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not sample_data.get("sample_quality"):
            quality = await self.ask_user(
                "What is the quality of the tissue sample?",
                tool_context,
                options=["Adequate", "Suboptimal", "Inadequate"]
            )
            sample_data["sample_quality"] = quality
            
        prompt = f"""Analyze the following tissue sample for bone sarcoma:
        
        Sample Data:
        {sample_data}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Histological features
        2. Tumor type and grade
        3. Mitotic activity
        4. Necrosis assessment
        5. Additional testing recommendations
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
            "findings": response_text,
            "confidence_score": 0.9,
            "requires_additional_testing": "additional testing" in response_text.lower()
        }
        
    async def interpret_molecular_tests(
        self,
        test_results: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Interpret molecular and genetic test results.
        
        Args:
            test_results: Dictionary containing test results
            tool_context: Tool context
            
        Returns:
            Dictionary containing interpretation
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not test_results.get("test_platform"):
            platform = await self.ask_user(
                "Which testing platform was used?",
                tool_context,
                options=["NGS", "FISH", "PCR", "IHC"]
            )
            test_results["test_platform"] = platform
            
        prompt = f"""Interpret the following molecular test results:
        
        Test Results:
        {test_results}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Key molecular findings
        2. Genetic alterations
        3. Clinical significance
        4. Treatment implications
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
            "interpretation": response_text,
            "actionable_findings": True if "actionable" in response_text.lower() else False
        }
        
    async def assess_margins(
        self,
        resection_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Assess surgical margins in resection specimens.
        
        Args:
            resection_data: Dictionary containing resection information
            tool_context: Tool context
            
        Returns:
            Dictionary containing margin assessment
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not resection_data.get("specimen_orientation"):
            oriented = await self.ask_user(
                "Is the specimen properly oriented?",
                tool_context,
                options=["Yes", "Partially", "No"]
            )
            resection_data["specimen_orientation"] = oriented
            
        prompt = f"""Assess the surgical margins of this resection specimen:
        
        Resection Data:
        {resection_data}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Margin status (all anatomical planes)
        2. Closest margin measurement
        3. Areas of concern
        4. Recommendations for additional sampling
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
            "margin_assessment": response_text,
            "margins_negative": True if "negative margins" in response_text.lower() else False,
            "requires_additional_sampling": "additional sampling" in response_text.lower()
        } 