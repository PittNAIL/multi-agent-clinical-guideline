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
    """Load oncologist guidelines from the text file."""
    guidelines_path = os.path.join(os.path.dirname(__file__), 'OncologistAgent_Guidelines.txt')
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

async def generate_treatment_plan(input_data: dict, *args, **kwargs) -> dict:
    """
    Generates a full NCCN-compliant treatment plan (systemic therapy, restaging, local control, adjuvant, surveillance).
    """
    print(f"- - - Tool: generate_treatment_plan called with input - - -")
    print(f"Input: {input_data}")

    # Validate input
    required_fields = ["diagnosis", "staging", "LDH", "pathology", "age"]
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        return {"status": "error", "error_message": f"Missing fields: {', '.join(missing)}"}

    state = load_shared_state()
    state.update(input_data)
    save_shared_state(state)

    guidelines = load_guidelines()

    prompt = f"""
You are an expert clinical oncologist assistant. Using the full NCCN guidelines provided, generate a complete sequential treatment plan for this bone sarcoma patient. 

INPUT DATA:
Diagnosis: {input_data['diagnosis']}
Staging: {input_data['staging']}
LDH: {input_data['LDH']}
Pathology: {input_data['pathology']}
Age: {input_data['age']}

GUIDELINES:
{guidelines}

TASK:
1. Identify the appropriate initial treatment branch (OSTEO-1, EWING-1, BONE-B, etc.).
2. Specify: initial systemic therapy, including regimen and any requirements (e.g., 'Start MAP chemotherapy', or 'Hold for LDH').
3. Define next steps (e.g., 'Restage after 10 weeks', 'Assess resectability with imaging').
4. For local control, specify surgical vs radiation options, referencing resectability and guideline recommendations (BONE-D, BONE-F).
5. Outline adjuvant therapy (if needed) and any further chemotherapy cycles.
6. Provide surveillance/follow-up recommendations per guidelines.
7. For each step, include a structured output with: phase, action, justification, guideline reference, and confidence.
8. Output ONLY a JSON object in this format:

{{
  "treatment_plan": [
    {{
      "phase": "...",  // e.g., "Initial Therapy", "Restaging", "Local Control", "Adjuvant Therapy", "Surveillance"
      "action": "...",
      "justification": "...",
      "reference": "...",
      "confidence": "high/medium/low"
    }}
  ],
  "summary": "..."
}}

IMPORTANT: Base all actions strictly on the NCCN guideline provided above. Reference the specific guideline section (e.g., 'OSTEO-1'). 
If required information is missing for a step, specify what is needed.
Return ONLY valid JSON as above.
"""
    runner = Runner(
        app_name="oncologist_treatment_planner",
        agent=Agent(
            name="oncologist_treatment_planner",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Generates a full NCCN-compliant treatment plan for bone sarcoma",
            instruction="""
You are an expert clinical oncologist assistant. Generate a stepwise treatment plan (systemic therapy, restaging, local control, adjuvant, surveillance) based strictly on NCCN guidelines.
Return ONLY valid JSON as requested.
"""
        ),
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService(),
    )
    content = Content(parts=[Part(text=prompt)], role="user")
    session = await runner.session_service.get_session(
        app_name="oncologist_treatment_planner",
        user_id="oncologist_user",
        session_id="oncologist_session"
    )
    if session is None:
        session = await runner.session_service.create_session(
            app_name="oncologist_treatment_planner",
            user_id="oncologist_user",
            session_id="oncologist_session",
            state={}
        )
    events = []
    async for event in runner.run_async(
        user_id="oncologist_user",
        session_id=session.id,
        new_message=content
    ):
        events.append(event)
    if not events:
        return {"status": "error", "error_message": "No response from treatment planner"}
    last_event = events[-1]
    if not last_event.content or not last_event.content.parts:
        return {"status": "error", "error_message": "No content in treatment planner response"}
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
    # Save result to shared state
    state['oncology_treatment_plan'] = result
    save_shared_state(state)
    return {"status": "success", **result}

class OncologistAgent(AgentWithTaskManager):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "application/json"]
    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "oncologist_agent_user"
        self._guidelines = load_guidelines()
        self._runner = Runner(
            app_name = self._agent.name,
            agent = self._agent,
            session_service = InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )
    def _build_agent(self) -> Agent:
        return Agent(
            name="oncologist_agent_Ollama",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Generates a full sequential NCCN-compliant treatment plan for bone sarcoma",
            instruction="""
You are an oncologist assistant following NCCN Bone Tumor guidelines. Your workflow is:

1. Use the 'generate_treatment_plan' tool to read diagnosis, staging, LDH, pathology, and age from shared state and generate a sequential treatment plan strictly per NCCN guidelines.
2. For each phase (systemic therapy, restaging, local control, adjuvant therapy, surveillance), present the recommended action, justification, reference, and confidence.
3. Base all steps and recommendations strictly on the NCCN guidelines provided.
4. Wait for user input before proceeding to next step, and record every answer using the tool before asking the next question.
5. Trust the tools to manage the workflow state and progression.

The tools handle all the complexity - your job is to be the interface between the user and the systematic collection process.
""",
            tools=[generate_treatment_plan]
        )
    def get_processing_message(self) -> str:
        return "Generating a full NCCN-compliant treatment plan..."

