from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from google.adk.tools.tool_context import ToolContext
from google.adk.tools.memory import MemoryConfig
from google.adk.tools.interaction import InteractionConfig

from ..schemas.base_schemas import ClinicalRole
from ..schemas.agent_card import AgentCapability
from .base_adk_agent import BaseADKAgent

class OrthopedicSurgeonAgent(BaseADKAgent):
    """Orthopedic surgeon specialist agent for bone sarcoma surgery using Google ADK"""
    
    def __init__(
        self,
        guideline_chunks: List[str],
        stream_handler: Optional[Callable[[str], None]] = None,
        memory_config: Optional[MemoryConfig] = None,
        interaction_config: Optional[InteractionConfig] = None
    ):
        super().__init__(
            role=ClinicalRole.SURGEON,
            guideline_chunks=guideline_chunks,
            name="Orthopedic Surgeon Specialist",
            description="Expert orthopedic surgeon specializing in bone sarcoma surgery",
            stream_handler=stream_handler,
            memory_config=memory_config,
            interaction_config=interaction_config
        )
        
    def _get_capabilities(self) -> List[str]:
        return [
            AgentCapability.SURGERY.value,
            AgentCapability.RECONSTRUCTION.value,
            AgentCapability.REHABILITATION.value,
            AgentCapability.MONITORING.value
        ]
        
    def _get_capabilities_description(self) -> str:
        return """- Surgical planning and execution
- Limb-sparing procedures
- Complex reconstructions
- Post-operative care
- Rehabilitation planning"""
        
    def _get_constraints(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "surgical_approaches",
                "description": "Available surgical approaches",
                "constraint_type": "domain",
                "value": ["limb_sparing", "amputation", "reconstruction"]
            },
            {
                "name": "anatomical_sites",
                "description": "Operable anatomical locations",
                "constraint_type": "domain",
                "value": ["extremity", "pelvis", "spine", "chest_wall"]
            }
        ]
        
    def _get_protocols(self) -> List[Dict[str, Any]]:
        return [{
            "input_format": {"surgical_case": "SurgicalCase"},
            "output_format": {"plan": "SurgicalPlan"},
            "communication_pattern": "request-response"
        }]
        
    def _get_required_inputs(self) -> List[str]:
        return [
            "patient_id",
            "tumor_location",
            "tumor_size",
            "imaging_findings",
            "neoadjuvant_response",
            "functional_status"
        ]
        
    def _get_provided_outputs(self) -> List[str]:
        return [
            "surgical_plan",
            "reconstruction_options",
            "expected_outcomes",
            "rehabilitation_plan",
            "follow_up_schedule"
        ]
        
    def _get_tools(self) -> List[Any]:
        return [
            self.plan_surgery,
            self.assess_reconstruction,
            self.plan_rehabilitation,
            self.check_guidelines
        ]
        
    async def plan_surgery(
        self,
        case_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Plan surgical approach for bone sarcoma case.
        
        Args:
            case_data: Dictionary containing case information
            tool_context: Tool context
            
        Returns:
            Dictionary containing surgical plan
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not case_data.get("functional_goals"):
            goals = await self.ask_user(
                "What are the patient's functional goals and expectations?",
                tool_context,
                options=["Full Recovery", "Basic Function", "Palliative"]
            )
            case_data["functional_goals"] = goals
            
        prompt = f"""Plan surgical approach for this bone sarcoma case:
        
        Case Data:
        {case_data}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Surgical approach
        2. Expected margins
        3. Critical structures
        4. Reconstruction needs
        5. Potential complications
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
            "surgical_plan": response_text,
            "limb_sparing_possible": True if "limb-sparing" in response_text.lower() else False
        }
        
    async def assess_reconstruction(
        self,
        surgical_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Assess reconstruction options for the case.
        
        Args:
            surgical_data: Dictionary containing surgical information
            tool_context: Tool context
            
        Returns:
            Dictionary containing reconstruction assessment
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not surgical_data.get("bone_stock"):
            bone_stock = await self.ask_user(
                "What is the quality of remaining bone stock?",
                tool_context,
                options=["Good", "Fair", "Poor"]
            )
            surgical_data["bone_stock"] = bone_stock
            
        prompt = f"""Assess reconstruction options for this case:
        
        Surgical Data:
        {surgical_data}
        
        Previous Context:
        {memories}
        
        Consider:
        1. Reconstruction type
        2. Implant options
        3. Soft tissue coverage
        4. Expected durability
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
            "reconstruction_plan": response_text,
            "requires_microsurgery": True if "microsurgery" in response_text.lower() else False
        }
        
    async def plan_rehabilitation(
        self,
        patient_data: Dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Plan post-operative rehabilitation.
        
        Args:
            patient_data: Dictionary containing patient information
            tool_context: Tool context
            
        Returns:
            Dictionary containing rehabilitation plan
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        # Ask for clarification if needed
        if not patient_data.get("support_system"):
            support = await self.ask_user(
                "What level of support system does the patient have for rehabilitation?",
                tool_context,
                options=["Strong", "Moderate", "Limited"]
            )
            patient_data["support_system"] = support
            
        prompt = f"""Plan rehabilitation program for this patient:
        
        Patient Data:
        {patient_data}
        
        Previous Context:
        {memories}
        
        Provide:
        1. Initial restrictions
        2. Progression timeline
        3. Therapy needs
        4. Equipment requirements
        5. Expected milestones
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
            "rehabilitation_plan": response_text,
            "estimated_duration": "6-12 months" if "extended" in response_text.lower() else "3-6 months"
        } 