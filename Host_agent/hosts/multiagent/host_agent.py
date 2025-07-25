import base64
import json
import uuid
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from common.client import A2ACardResolver
from common.types import (
    AgentCard,
    DataPart,
    Message,
    Part,
    Task,
    TaskSendParams,
    TaskState,
    TextPart,
    InternalError,
    SendTaskResponse
)
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from lite_llm import LiteLlm

from .remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback

import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class HostAgent:
    """The host agent responsible for coordinating the clinical workflow."""

    # Define the forced workflow sequence
    AGENT_SEQUENCE = [
        "IntakeAgent",
        "RadiologistAgent",
        "PathologistAgent",
        "OncologistAgent",    # Not yet made
        "SurgicalAgent",      # Not yet made
        "SurveillanceAgent"   # Not yet made
    ]

    def __init__(
        self,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}

        # Read all agent cards and order them according to AGENT_SEQUENCE
        ordered_cards = []
        unordered_cards = []
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(address)
            card = card_resolver.get_agent_card()
            if card.name in self.AGENT_SEQUENCE:
                ordered_cards.append(card)
            else:
                unordered_cards.append(card)
            self.remote_agent_connections[card.name] = RemoteAgentConnections(card)
            self.cards[card.name] = card

        # Sort ordered_cards by AGENT_SEQUENCE
        ordered_cards.sort(key=lambda c: self.AGENT_SEQUENCE.index(c.name) if c.name in self.AGENT_SEQUENCE else 999)
        self.ordered_agent_names = [c.name for c in ordered_cards] + [c.name for c in unordered_cards]

        agent_info = []
        for card in ordered_cards + unordered_cards:
            agent_info.append(json.dumps({'name': card.name, 'description': card.description}))
        self.agents = '\n'.join(agent_info)

    def get_shared_state_path(self) -> str:
        """Get the path to the shared state file."""
        # Get the workspace root directory (A2A)
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        # Create shared_state directory if it doesn't exist
        shared_state_dir = os.path.join(workspace_root, "shared_state")
        os.makedirs(shared_state_dir, exist_ok=True)
        # Return path to shared_state.json
        return os.path.join(shared_state_dir, "shared_state.json")

    def load_shared_state(self) -> dict:
        """Load the shared state from file."""
        try:
            state_path = self.get_shared_state_path()
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading shared state: {e}")
            return {}

    def get_imaging_workflow_summary(self) -> dict:
        """Get a summary of the current imaging workflow status from shared state."""
        state = self.load_shared_state()
        if not state:
            return {"status": "no_state", "message": "No shared state found"}
        
        imaging_recommendations = state.get("imaging_recommendations")
        imaging_results = state.get("imaging_results", [])
        
        if not imaging_recommendations:
            # Fallback to old format
            if imaging_results:
                return {
                    "status": "completed_legacy",
                    "message": f"Imaging completed (legacy format): {len(imaging_results)} studies recorded",
                    "total_studies": len(imaging_results)
                }
            else:
                return {
                    "status": "not_started",
                    "message": "Imaging workflow not yet started"
                }
        
        # Parse structured imaging recommendations
        workflow_status = imaging_recommendations.get("workflow_status", {})
        imaging_studies = imaging_recommendations.get("imaging_studies", {})
        clinical_context = imaging_recommendations.get("clinical_context", {})
        
        recommended_count = len(imaging_studies.get("recommended", []))
        completed_count = len(imaging_results)
        
        summary = {
            "status": workflow_status.get("completion_status", "unknown"),
            "current_step": workflow_status.get("current_step", "unknown"),
            "next_action": workflow_status.get("next_action", "unknown"),
            "studies_recommended": recommended_count,
            "studies_completed": completed_count,
            "studies_pending": max(0, recommended_count - completed_count),
            "matched_guideline": imaging_recommendations.get("matched_guideline", {}).get("name", "unknown"),
            "staging_status": clinical_context.get("staging_status", "unknown"),
            "symptoms": clinical_context.get("symptoms", [])
        }
        
        if summary["status"] == "pending":
            summary["message"] = f"Imaging in progress: {completed_count}/{recommended_count} studies completed"
        elif summary["status"] == "complete":
            summary["message"] = f"Imaging workflow complete: {completed_count} studies completed"
        else:
            summary["message"] = f"Imaging status: {summary['status']}"
        
        return summary

    def get_pathology_workflow_summary(self) -> dict:
        """Get a summary of the current pathology workflow status from shared state."""
        state = self.load_shared_state()
        if not state:
            return {"status": "no_state", "message": "No shared state found"}
        
        pathology_analysis = state.get("pathology_analysis")
        pathology_results = state.get("pathology_results", [])
        diagnosis = state.get("diagnosis")
        
        if diagnosis:
            return {
                "status": "completed",
                "message": f"Diagnosis confirmed: {diagnosis.get('primary_diagnosis', 'unknown')}",
                "diagnosis": diagnosis.get('primary_diagnosis'),
                "subtype": diagnosis.get('subtype'),
                "grade": diagnosis.get('grade'),
                "confidence": diagnosis.get('confidence_level')
            }
        
        if not pathology_analysis:
            return {
                "status": "not_started",
                "message": "Pathology workflow not yet started"
            }
        
        # Parse structured pathology analysis
        workflow_status = pathology_analysis.get("workflow_status", {})
        pathology_workflow = pathology_analysis.get("pathology_workflow", {})
        
        required_count = len(pathology_workflow.get("required_tests", []))
        completed_count = len(pathology_results)
        
        summary = {
            "status": workflow_status.get("completion_status", "unknown"),
            "current_step": workflow_status.get("current_step", "unknown"),
            "next_action": workflow_status.get("next_action", "unknown"),
            "tests_required": required_count,
            "tests_completed": completed_count,
            "tests_pending": max(0, required_count - completed_count),
            "suspected_diagnoses": [d.get("tumor_type") for d in pathology_analysis.get("suspected_diagnoses", [])]
        }
        
        if summary["status"] == "pending":
            summary["message"] = f"Pathology in progress: {completed_count}/{required_count} tests completed"
        elif summary["status"] == "ready_for_diagnosis":
            summary["message"] = f"Pathology complete: {completed_count} tests completed, ready for diagnosis"
        else:
            summary["message"] = f"Pathology status: {summary['status']}"
        
        return summary

    def determine_next_agent(self, message: str) -> str:
        """
        Force the workflow sequence: IntakeAgent → RadiologistAgent → PathologistAgent → OncologistAgent → SurgicalAgent → SurveillanceAgent.
        Always route to the next agent in the sequence.
        """
        state = self.load_shared_state()
        # Find the current agent in the sequence
        current_agent = state.get("agent")
        if not current_agent:
            # Start with the first agent
            return self.AGENT_SEQUENCE[0]

        # Find the index of the current agent
        try:
            idx = self.AGENT_SEQUENCE.index(current_agent)
        except ValueError:
            # If not found, start from the beginning
            return self.AGENT_SEQUENCE[0]

        # Move to the next agent in the sequence
        if idx < len(self.AGENT_SEQUENCE) - 1:
            return self.AGENT_SEQUENCE[idx + 1]
        else:
            # If at the end, stay with SurveillanceAgent
            return self.AGENT_SEQUENCE[-1]

    def register_agent_card(self, card: AgentCard):
        remote_connection = RemoteAgentConnections(card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def create_agent(self) -> Agent:
        return Agent(
            model=LiteLlm(model="ollama_chat/llama3.2"),
            name='host_agent_Ollama',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                'This agent orchestrates the clinical workflow by routing tasks to the appropriate specialist agents.'
            ),
            tools=[
                self.list_remote_agents,
                self.send_task,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_state(context)
        imaging_summary = self.get_imaging_workflow_summary()
        pathology_summary = self.get_pathology_workflow_summary()
        
        return f"""You are an expert clinical workflow coordinator that routes tasks to the appropriate specialist agents.

Workflow Sequence:
1. IntakeAgent: Initial patient assessment and data collection
2. RadiologistAgent: Imaging recommendations and analysis
3. PathologistAgent: Biopsy and tissue analysis for diagnostic confirmation
4. OncologistAgent: Treatment planning and management
5. SurgicalAgent: Surgical intervention
6. SurveillanceAgent: Follow-up and monitoring

Current Workflow Status:
- Active Agent: {current_agent['active_agent']}
- Imaging Status: {imaging_summary.get('message', 'Unknown')}
- Pathology Status: {pathology_summary.get('message', 'Unknown')}
- Recommended Imaging Studies: {imaging_summary.get('studies_recommended', 0)}
- Completed Imaging Studies: {imaging_summary.get('studies_completed', 0)}
- Required Pathology Tests: {pathology_summary.get('tests_required', 0)}
- Completed Pathology Tests: {pathology_summary.get('tests_completed', 0)}
- Current Diagnosis: {pathology_summary.get('diagnosis', 'Pending')}
- Staging: {imaging_summary.get('staging_status', 'Unknown')}

Available Agents:
{self.agents}

Your responsibilities:
1. Analyze the user's request to determine which specialist agent should handle it
2. Route the task to the appropriate agent based on current workflow status
3. Ensure the workflow follows the correct sequence
4. Track the progress of each case, especially imaging and pathology completion

Workflow Rules:
- If imaging recommendations are pending collection, stay with RadiologistAgent
- If imaging is complete but results are missing, stay with RadiologistAgent  
- After imaging is complete, proceed to PathologistAgent for diagnostic confirmation
- If pathology tests are pending collection, stay with PathologistAgent
- If pathology is complete but no diagnosis confirmed, stay with PathologistAgent
- Only proceed to OncologistAgent after diagnosis is confirmed

When receiving a new request:
1. First determine if it's an initial checkup/consultation → IntakeAgent
2. Check imaging workflow status before proceeding to PathologistAgent
3. Check pathology workflow status before proceeding to OncologistAgent
4. Route based on clinical workflow sequence and current completion status
5. Wait for specialist response before proceeding to the next step

Always use the send_task tool to route tasks to the appropriate agent.
"""

    def check_state(self, context: ReadonlyContext):
        state = context.state
        if (
            'session_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'agent' in state
        ):
            return {'active_agent': f'{state["agent"]}'}
        return {'active_agent': 'None'}

    def before_model_callback(
        self, callback_context: CallbackContext, llm_request
    ):
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            if 'session_id' not in state:
                state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_task(
        self, message: str, tool_context: ToolContext
    ):
        agent_name = self.determine_next_agent(message)
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f'Agent {agent_name} not found')
        state = tool_context.state
        state['agent'] = agent_name
        card = self.cards[agent_name]
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        if 'task_id' in state:
            taskId = state['task_id']
        else:
            taskId = str(uuid.uuid4())
        sessionId = state['session_id']
        messageId = ''
        metadata = {}
        if 'input_message_metadata' in state:
            metadata.update(**state['input_message_metadata'])
            if 'message_id' in state['input_message_metadata']:
                messageId = state['input_message_metadata']['message_id']
        if not messageId:
            messageId = str(uuid.uuid4())
        metadata.update(conversation_id=sessionId, message_id=messageId)
        request: TaskSendParams = TaskSendParams(
            id=taskId,
            sessionId=sessionId,
            message=Message(
                role='user',
                parts=[TextPart(text=message)],
                metadata=metadata,
            ),
            acceptedOutputModes=['text', 'text/plain', 'image/png'],
            metadata={'conversation_id': sessionId},
        )
        task = await client.send_task(request, self.task_callback)
        state['session_active'] = task.status.state not in [
            TaskState.COMPLETED,
            TaskState.CANCELED,
            TaskState.FAILED,
            TaskState.UNKNOWN,
        ]
        return task

def convert_parts(parts: list[Part], tool_context: ToolContext):
    """Convert parts to the appropriate format."""
    return [convert_part(part, tool_context) for part in parts]

def convert_part(part: Part, tool_context: ToolContext):
    """Convert a single part to the appropriate format."""
    if isinstance(part, TextPart):
        return types.Part.from_text(text=part.text)
    elif isinstance(part, DataPart):
        return types.Part.from_data(mime_type=part.mime_type, data=part.data)
    else:
        raise ValueError(f"Unsupported part type: {type(part)}")
