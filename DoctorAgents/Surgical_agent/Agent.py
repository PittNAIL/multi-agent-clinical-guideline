import os
import json
import logging
from datetime import datetime
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content, Part
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from task_manager import AgentWithTaskManager

logging.basicConfig(level=logging.ERROR)

def load_guidelines():
    """Load surgical guidelines from the text file."""
    guidelines_path = os.path.join(os.path.dirname(__file__), 'SurgicalAgent_Guidelines.txt')
    try:
        with open(guidelines_path, 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading guidelines: {e}")
        return ""

def get_shared_state_path():
    agents_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(agents_dir, "shared_state.json")

def load_shared_state():
    state_path = get_shared_state_path()
    try:
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.error(f"Error loading shared state: {e}")
        return {}

def save_shared_state(state):
    state_path = get_shared_state_path()
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error saving shared state: {e}")
        return False

async def plan_resection_strategy(input_data: dict, *args, **kwargs) -> dict:
    """
    Determines resection type, margin goals, and specialist needs per NCCN surgical guideline.
    INPUT:
        - tumor_location
        - imaging_summary
        - diagnosis
        - post_chemo_response
        - pathology_report
    OUTPUT:
        {
            "status": "success",
            "resection_type": "...",
            "margin_goal": "...",
            "specialist_needed": "...",
            "justification": "...",
            "reference": "NCCN Surgical Guidelines (section)",
        }
    """
    print(f"- - - Tool: plan_resection_strategy called with input - - -")
    print(f"Input: {input_data}")

    # Validate input
    required_fields = ["tumor_location", "imaging_summary", "diagnosis", "post_chemo_response", "pathology_report"]
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        return {"status": "error", "error_message": f"Missing fields: {', '.join(missing)}"}

    state = load_shared_state()
    state.update(input_data)
    save_shared_state(state)

    guidelines = load_guidelines()

    prompt = f"""
You are an expert surgical oncologist. Using the NCCN guidelines below, provide a resection strategy for this patient.

PATIENT DATA:
Tumor location: {input_data['tumor_location']}
Imaging summary: {input_data['imaging_summary']}
Diagnosis: {input_data['diagnosis']}
Post-chemo response: {input_data['post_chemo_response']}
Pathology report: {input_data['pathology_report']}

GUIDELINES:
{guidelines}

TASK:
1. Define the recommended resection type (e.g., "Wide en bloc resection").
2. State the surgical margin goal (e.g., "≥2 cm margin; avoid positive margins near neurovascular structures").
3. Specify if a surgical or other specialist referral is indicated (e.g., pelvic involvement, close to major vessels).
4. Provide a brief justification and cite the relevant NCCN guideline section.

Output ONLY a JSON object in this format:
{{
    "status": "success",
    "resection_type": "...",
    "margin_goal": "...",
    "specialist_needed": "...",
    "justification": "...",
    "reference": "NCCN Surgical Guidelines (section)"
}}
"""

    runner = Runner(
        app_name="surgical_strategy_planner",
        agent=Agent(
            name="surgical_strategy_planner",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Defines resection strategy and specialist needs per NCCN guideline",
            instruction="""
You are a surgical oncologist assistant. Recommend resection type, margin goal, and specialist referral per NCCN. Return ONLY valid JSON.
"""
        ),
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService(),
    )

    content = Content(parts=[Part(text=prompt)], role="user")
    session = await runner.session_service.get_session(
        app_name="surgical_strategy_planner",
        user_id="surgical_user",
        session_id="surgical_session"
    )
    if session is None:
        session = await runner.session_service.create_session(
            app_name="surgical_strategy_planner",
            user_id="surgical_user",
            session_id="surgical_session",
            state={}
        )
    events = []
    async for event in runner.run_async(
        user_id="surgical_user",
        session_id=session.id,
        new_message=content
    ):
        events.append(event)
    if not events:
        return {"status": "error", "error_message": "No response from surgical strategy planner"}
    last_event = events[-1]
    if not last_event.content or not last_event.content.parts:
        return {"status": "error", "error_message": "No content in surgical planner response"}
    response_content = last_event.content.parts[0].text.strip()
    if response_content.startswith('```json'):
        response_content = response_content[7:]
    if response_content.endswith('```'):
        response_content = response_content[:-3]
    response_content = response_content.strip()
    try:
        result = json.loads(response_content)
    except Exception as e:
        return {"status": "error", "error_message": f"Invalid JSON from LLM: {str(e)}", "raw": response_content}
    # Save to shared state
    state['surgical_plan'] = result
    save_shared_state(state)
    return {"status": "success", **result}

class SurgicalAgent(AgentWithTaskManager):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "application/json"]

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "surgical_agent_user"
        self._guidelines = load_guidelines()
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> Agent:
        return Agent(
            name="surgical_agent_Ollama",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Defines resection strategy and specialist needs per NCCN guideline",
            instruction="""
You are a surgical oncology assistant following NCCN Bone Tumor guidelines. Your workflow is:

1. Use the 'plan_resection_strategy' tool to recommend resection type, margin goal, and if a specialist referral is needed, based strictly on NCCN guidelines and patient data.
2. For each case, output your recommendation with justification and guideline reference.
3. Base all steps strictly on NCCN guidelines.
4. Wait for user input before proceeding to next step.
5. Trust the tools to manage workflow and state.
"""
        , tools=[plan_resection_strategy])

    def get_processing_message(self) -> str:
        return "Planning surgical resection strategy per NCCN guidelines..."

