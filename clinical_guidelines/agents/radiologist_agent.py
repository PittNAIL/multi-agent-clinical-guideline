from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import uuid

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.memory import MemoryConfig
from google.adk.tools.interaction import InteractionConfig

from ..schemas.base_schemas import PatientState, AgentResponse, ClinicalRole, GuidelineMatch
from ..schemas.agent_card import AgentCard, AgentCapability, AgentType, AgentConstraint
from common.types import Task, TaskSendParams, TaskState, Message, TextPart
from .base_adk_agent import BaseADKAgent

class RadiologistAgent(BaseADKAgent):
    """Radiologist specialist agent for bone sarcoma imaging analysis using Google ADK"""
    
    def __init__(
        self,
        guideline_chunks: List[str],
        stream_handler: Optional[Callable[[str], None]] = None,
        memory_config: Optional[MemoryConfig] = None,
        interaction_config: Optional[InteractionConfig] = None
    ):
        super().__init__(
            role=ClinicalRole.RADIOLOGIST,
            guideline_chunks=guideline_chunks,
            name="Radiologist Specialist",
            description="Expert radiologist specializing in bone sarcoma imaging analysis",
            stream_handler=stream_handler,
            memory_config=memory_config,
            interaction_config=interaction_config
        )
        
    def _get_capabilities(self) -> List[str]:
        return [
            AgentCapability.IMAGING.value,
            AgentCapability.DIAGNOSIS.value
        ]
        
    def _get_capabilities_description(self) -> str:
        return """- Initial imaging selection and interpretation
- Metastatic disease assessment
- Treatment response evaluation
- Local recurrence monitoring"""
        
    def _get_constraints(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "imaging_types",
                "description": "Limited to radiological imaging interpretation",
                "constraint_type": "domain",
                "value": ["X-ray", "MRI", "CT", "PET-CT"]
            },
            {
                "name": "anatomical_focus",
                "description": "Specialized in musculoskeletal imaging",
                "constraint_type": "domain",
                "value": ["bone", "soft tissue", "joint"]
            }
        ]
        
    def _get_protocols(self) -> List[Dict[str, Any]]:
        return [{
            "input_format": {"patient_state": "PatientState"},
            "output_format": {"decision": "AgentResponse"},
            "communication_pattern": "request-response"
        }]
        
    def _get_required_inputs(self) -> List[str]:
        return [
            "patient_id",
            "age",
            "imaging_history",
            "current_symptoms",
            "anatomical_location"
        ]
        
    def _get_provided_outputs(self) -> List[str]:
        return [
            "imaging_findings",
            "tumor_characteristics",
            "staging_assessment",
            "response_evaluation"
        ]
        
    def _get_tools(self) -> List[Any]:
        return [
            self.analyze_imaging,
            self.recommend_imaging,
            self.assess_response,
            self.check_guidelines
        ]
        
    async def analyze_imaging(
        self,
        imaging_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Analyze imaging studies and provide findings.
        
        Args:
            imaging_data: Dictionary containing imaging information
            tool_context: Tool context
            
        Returns:
            Dictionary containing analysis results
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not imaging_data.get("image_quality"):
            image_quality = await self.ask_user(
                "What is the quality of the imaging study?",
                tool_context,
                options=["High", "Medium", "Low"]
            )
            imaging_data["image_quality"] = image_quality
            
        prompt = f"""Analyze the following imaging studies for bone sarcoma:
        
        Imaging Data:
        {imaging_data}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Imaging findings
        2. Tumor characteristics
        3. Staging assessment
        4. Areas of concern
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
            "requires_review": False
        }
        
    async def recommend_imaging(
        self,
        patient_state: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Recommend appropriate imaging protocols.
        
        Args:
            patient_state: Dictionary containing patient information
            tool_context: Tool context
            
        Returns:
            Dictionary containing imaging recommendations
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not patient_state.get("prior_imaging"):
            has_prior = await self.ask_user(
                "Has the patient had any prior imaging studies?",
                tool_context,
                options=["Yes", "No"]
            )
            patient_state["prior_imaging"] = has_prior == "Yes"
            
        prompt = f"""Based on the patient state, recommend appropriate imaging protocols:
        
        Patient State:
        {patient_state}
        
        Previous Context:
        {memories}
        
        Consider:
        1. Initial workup vs follow-up
        2. Local vs systemic assessment
        3. Contrast requirements
        4. Specific sequences/protocols
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
            "recommendations": response_text,
            "priority": "urgent" if patient_state.get("is_new_diagnosis") else "routine"
        }
        
    async def assess_response(
        self,
        previous_imaging: Dict[str, Any],
        current_imaging: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Assess treatment response by comparing imaging studies.
        
        Args:
            previous_imaging: Previous imaging data
            current_imaging: Current imaging data
            tool_context: Tool context
            
        Returns:
            Dictionary containing response assessment
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not current_imaging.get("time_since_previous"):
            time_since_prev = await self.ask_user(
                "How long has it been since the previous imaging study?",
                tool_context
            )
            current_imaging["time_since_previous"] = time_since_prev
            
        prompt = f"""Compare previous and current imaging to assess treatment response:
        
        Previous Imaging:
        {previous_imaging}
        
        Current Imaging:
        {current_imaging}
        
        Previous Context:
        {memories}
        
        Assess:
        1. Changes in tumor size
        2. Changes in tumor characteristics
        3. New findings
        4. Overall response category
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
            "response_assessment": response_text,
            "requires_tumor_board": True if "concerning" in response_text.lower() else False
        }
        
    async def check_guidelines(
        self,
        clinical_scenario: str,
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Check relevant imaging guidelines for the clinical scenario.
        
        Args:
            clinical_scenario: Description of the clinical scenario
            tool_context: Tool context
            
        Returns:
            Dictionary containing relevant guidelines
        """
        prompt = f"""Find relevant imaging guidelines for this clinical scenario:
        
        Scenario:
        {clinical_scenario}
        
        Guidelines:
        {self.guidelines}
        
        Return specific guideline recommendations and references.
        """
        
        response = await tool_context.model.generate_content(prompt)
        
        return {
            "guideline_matches": response.text,
            "confidence": 0.9
        } 