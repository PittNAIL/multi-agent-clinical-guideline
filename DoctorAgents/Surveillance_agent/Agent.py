import os
import json
import logging
from datetime import datetime, timedelta
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
    """Load surveillance guidelines from the text file."""
    guidelines_path = os.path.join(os.path.dirname(__file__), 'SurveillanceAgent_Guidelines.txt')
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

def parse_date(date_str):
    """Parse YYYY-MM-DD to datetime, else None."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None

async def plan_surveillance_schedule(input_data: dict, *args, **kwargs) -> dict:
    """
    Generates a guideline-driven NCCN surveillance schedule for a post-treatment bone sarcoma patient.
    INPUT: diagnosis, date_of_treatment_completion, age, prior_imaging (list), prior_labs (list)
    OUTPUT: structured schedule and next actions, justification, reference.
    """
    print(f"- - - Tool: plan_surveillance_schedule called with input - - -")
    print(f"Input: {input_data}")

    required_fields = ["diagnosis", "date_of_treatment_completion", "age"]
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        return {"status": "error", "error_message": f"Missing fields: {', '.join(missing)}"}

    # Save input to shared state
    state = load_shared_state()
    state.update(input_data)
    save_shared_state(state)

    guidelines = load_guidelines()

    # Compose LLM prompt
    prompt = f"""
You are a post-treatment surveillance coordinator for bone sarcoma survivors. Using the following NCCN surveillance guidelines, generate a complete schedule of required follow-up imaging, labs, and visits for the next 5 years for this patient.

PATIENT DATA:
Diagnosis: {input_data['diagnosis']}
Date of treatment completion: {input_data['date_of_treatment_completion']}
Age: {input_data['age']}
Prior imaging: {json.dumps(input_data.get('prior_imaging', []), indent=2)}
Prior labs: {json.dumps(input_data.get('prior_labs', []), indent=2)}

GUIDELINES:
{guidelines}

TASK:
1. Provide a surveillance schedule covering imaging, labs, and clinical visits, specifying recommended interval (e.g., every 3 months for 2 years), and modality (MRI, chest CT, etc.).
2. List any studies that are currently overdue or should be scheduled soon, based on prior studies.
3. Provide structured reminders for next steps.
4. Give a brief justification and cite relevant NCCN guideline section.
5. Output ONLY a JSON object with this structure:

{{
  "status": "success",
  "surveillance_schedule": [
    {{
      "interval": "...",      // e.g. "Every 3 months for 2 years, then every 6 months for 3 years"
      "next_due": "...",      // ISO date
      "modality": "...",      // e.g. "MRI of primary site, chest CT"
    }}
  ],
  "overdue_items": [
    {{"modality": "...", "last_done": "...", "due_since": "..."}}
  ],
  "reminders": [
    "Schedule MRI of primary site (overdue)",
    "Schedule chest CT (due in 2 weeks)"
  ],
  "justification": "...",
  "reference": "NCCN Surveillance Guidelines (section)"
}}

IMPORTANT: Base all recommendations strictly on the guideline content above. Return ONLY valid JSON in the format shown.
"""

    runner = Runner(
        app_name="surveillance_planner",
        agent=Agent(
            name="surveillance_planner",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Creates NCCN-compliant surveillance schedules for bone sarcoma survivors",
            instruction="""
You are a post-treatment surveillance assistant. Generate a 5-year imaging/lab/visit schedule per NCCN, flag overdue studies, and provide actionable reminders. Output ONLY valid JSON.
"""
        ),
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService(),
    )

    content = Content(parts=[Part(text=prompt)], role="user")
    session = await runner.session_service.get_session(
        app_name="surveillance_planner",
        user_id="surveillance_user",
        session_id="surveillance_session"
    )
    if session is None:
        session = await runner.session_service.create_session(
            app_name="surveillance_planner",
            user_id="surveillance_user",
            session_id="surveillance_session",
            state={}
        )
    events = []
    async for event in runner.run_async(
        user_id="surveillance_user",
        session_id=session.id,
        new_message=content
    ):
        events.append(event)
    if not events:
        return {"status": "error", "error_message": "No response from surveillance planner"}
    last_event = events[-1]
    if not last_event.content or not last_event.content.parts:
        return {"status": "error", "error_message": "No content in surveillance planner response"}
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
    state['surveillance_plan'] = result
    save_shared_state(state)
    return {"status": "success", **result}

class SurveillanceAgent(AgentWithTaskManager):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "application/json"]

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "surveillance_agent_user"
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
            name="surveillance_agent_Ollama",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Manages NCCN-based post-treatment surveillance for bone sarcoma",
            instruction="""
You are a follow-up and survivorship care assistant for bone tumor patients. Your job is to:
- Generate and update an NCCN-compliant surveillance schedule (imaging, labs, visits)
- Flag overdue or missing studies
- Remind clinicians of next steps and cite guideline references
Base all actions strictly on NCCN guidelines and persist to shared state.
""",
            tools=[plan_surveillance_schedule]
        )

    def get_processing_message(self) -> str:
        return "Planning post-treatment surveillance per NCCN guidelines..."

