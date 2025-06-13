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

    def __init__(
        self,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self._intake_completed = False  # Track if IntakeAgent has run
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(address)
            card = card_resolver.get_agent_card()
            remote_connection = RemoteAgentConnections(card)
            self.remote_agent_connections[card.name] = remote_connection
            self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
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

    def determine_next_agent(self, message: str) -> str:
        # Always start with IntakeAgent on the very first step
        state = self.load_shared_state()
        if not state or not state.get("intake_completed"):
            return "IntakeAgent"
        # After IntakeAgent, use state-based routing
        if not state.get("imaging_results"):
            return "RadiologistAgent"
        if not state.get("diagnosis"):
            return "PathologistAgent"
        if not state.get("treatment_plan"):
            return "OncologistAgent"
        if not state.get("surgery_status"):
            return "SurgicalAgent"
        if not state.get("surveillance_plan"):
            return "SurveillanceAgent"
        return "SurveillanceAgent"

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
        return f"""You are an expert clinical workflow coordinator that routes tasks to the appropriate specialist agents.

Workflow Sequence:
1. IntakeAgent: Initial patient assessment and data collection
2. RadiologistAgent: Imaging recommendations and analysis
3. PathologistAgent: Biopsy and tissue analysis
4. OncologistAgent: Treatment planning and management
5. SurgicalAgent: Surgical intervention
6. SurveillanceAgent: Follow-up and monitoring

Your responsibilities:
1. Analyze the user's request to determine which specialist agent should handle it
2. Route the task to the appropriate agent
3. Ensure the workflow follows the correct sequence
4. Track the progress of each case

Available Agents:
{self.agents}

Current active agent: {current_agent['active_agent']}

When receiving a new request:
1. First determine if it's an initial checkup/consultation
2. If not, check if it's about imaging, pathology, treatment, surgery, or surveillance
3. Route the task to the appropriate specialist agent
4. Wait for their response before proceeding to the next step

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
