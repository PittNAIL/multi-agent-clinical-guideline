# python/agents/Intake_Agent/Agent.py

import os
import json
import asyncio
import warnings
import logging
from pathlib import Path
import datetime

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.genai.types import Content, Part

from task_manager import AgentWithTaskManager

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


def get_shared_state_path() -> str:
    """
    Return the absolute path to the shared_state.json file.
    """
    # Get the absolute path to the agents directory
    agents_dir = Path(__file__).parent.parent.absolute()
    state_path = agents_dir / "shared_state.json"
    print(f"Shared state will be saved to: {state_path}")
    return str(state_path)

def read_shared_state() -> dict:
    """
    Opens the existing shared_state.json and returns its parsed contents.
    If the file does not exist, returns an error dict.
    """
    path = get_shared_state_path()
    try:
        if not os.path.exists(path):
            return {
                "status": "error",
                "error_message": f"shared_state.json not found at {path}"
            }
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "success",
            "shared_state": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to read shared state: {str(e)}"
        }


def load_shared_state() -> dict:
    """
    Load the shared state from shared_state.json.
    If the file does not exist yet, return a default skeleton.
    """
    state_path = get_shared_state_path()
    try:
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # If the file isn't there yet, return a minimal default:
        return {
            "status": "success",
            "initial_check": None,
            "symptoms": None,
            "patient_info": {
                "age": None,
                "sex": None,
                "history": None,
                "additional_details": {}
            },
            "previous_imaging": [],
            "imaging_results": [],
            "additional_info": {}
        }
    except Exception as e:
        logging.error(f"Error loading shared state: {e}")
        return None


def save_shared_state(state: dict) -> bool:
    """
    Overwrite shared_state.json on disk with the given dict.
    Returns True on success, False otherwise.
    """
    state_path = get_shared_state_path()
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        # First try to read existing state to verify we can access the file
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                print(f"Successfully read existing state from {state_path}")
            except Exception as e:
                print(f"Warning: Could not read existing state: {e}")
        
        # Now try to write the new state
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        
        # Verify the write was successful
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved == state:
                print(f"Successfully verified state was saved to {state_path}")
                return True
            else:
                print("Warning: Saved state does not match what we tried to save")
                return False
        else:
            print(f"Error: File {state_path} does not exist after saving")
            return False
            
    except Exception as e:
        print(f"Error saving shared state to {state_path}: {e}")
        logging.error(f"Error saving shared state: {e}")
        return False


async def structure_clinical_input(text: str, *args, **kwargs) -> dict:
    """Structures raw clinical text input into a standardized format for the RadiologistAgent,
    writes it to shared_state.json, and returns the full JSON state.

    Args:
        text (str): Raw clinical text input from healthcare provider.

    Returns:
        dict: A dictionary containing:
            - status: 'success' or 'error'
            - symptoms: list of extracted symptoms
            - patient_info: Basic patient information
            - previous_imaging: List of previously performed imaging studies
            - additional_info: Any extra information extracted from the input
            - shared_state: The entire contents of shared_state.json after saving
    """
    print(f"\n=== structure_clinical_input called ===")
    print(f"Input text: {text}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Shared state path: {get_shared_state_path()}")

    try:
        # Save initial state to show we can write to the file
        initial_state = {
            "status": "processing",
            "raw_input": text,
            "timestamp": str(datetime.datetime.now())
        }
        if not save_shared_state(initial_state):
            print("Warning: Could not save initial state")

        # 1) Ask LLM to extract JSON-structured data
        prompt = f"""You are a clinical data extraction assistant. Your task is to extract information from the following clinical text and return it as a JSON object. DO NOT return any code, only the JSON object.

Input text: {text}

Extract the following information and format it as a JSON object:
1. Symptoms requiring imaging (e.g., "bone pain", "swelling", "limb weakness")
2. Patient information (name, age, sex, relevant history)
3. Previously performed imaging studies (type, date, findings)
4. Any additional relevant clinical information

Return ONLY a JSON object with this structure:
{{
    "symptoms": ["symptom1", "symptom2", ...],
    "patient_info": {{
        "name": "patient name" or null,
        "age": age or null,
        "sex": "M/F" or null,
        "history": "relevant history" or null,
        "additional_details": {{}}
    }},
    "previous_imaging": [
        {{
            "type": "imaging type",
            "date": "date",
            "findings": "findings",
            "additional_details": {{}}
        }}
    ],
    "additional_info": {{}}
}}

CRITICAL INSTRUCTIONS FOR PREVIOUS IMAGING:
- ONLY include imaging studies that are EXPLICITLY mentioned as already completed/performed
- DO NOT assume or infer imaging studies based on body parts mentioned
- DO NOT add imaging studies just because a location is mentioned (e.g., "knee pain" does not mean "knee X-ray was done")
- If the text says "bone lesion on left knee" - this describes a CURRENT condition, not a previous X-ray
- If the text says "patient had an MRI last week" - this IS a previous imaging study
- If no previous imaging is explicitly mentioned, use an empty array []

Examples of what NOT to include:
- "bone lesion on left knee" → DO NOT add "left knee X-ray" to previous_imaging
- "chest pain" → DO NOT add "chest X-ray" to previous_imaging  
- "headache" → DO NOT add "head CT" to previous_imaging

Examples of what TO include:
- "patient had a chest X-ray last month showing normal results" → ADD to previous_imaging
- "MRI performed yesterday revealed..." → ADD to previous_imaging
- "CT scan from last week showed..." → ADD to previous_imaging

Important guidelines:
1. Return ONLY the JSON object, no other text or code
2. If no previous imaging is mentioned, use an empty array for "previous_imaging"
3. If any patient information is not mentioned, use null for that field
4. If no symptoms are mentioned, use an empty array for "symptoms"
5. Use null for any missing values, not empty strings or undefined
6. For dates, use YYYY-MM-DD format if possible
7. Keep the original medical terminology from the input text
8. Include any extra information that might be relevant for imaging decisions
9. Always extract the patient's name if mentioned in the text
10. DO NOT infer or assume imaging studies - only extract what is explicitly stated

Example input: "Patient is John Smith, 45 years old with knee pain and family history of arthritis"
Example output: {{"symptoms": ["knee pain"], "patient_info": {{"name": "John Smith", "age": 45, "sex": null, "history": "family history of arthritis", "additional_details": {{}}}}, "previous_imaging": [], "additional_info": {{}}}}

Example input: "Patient Praz, age 6, has bone lesion on left knee with swelling and family history of cancer"
Example output: {{"symptoms": ["bone lesion", "swelling"], "patient_info": {{"name": "Praz", "age": 6, "sex": null, "history": "family history of cancer", "additional_details": {{}}}}, "previous_imaging": [], "additional_info": {{}}}}
"""

        print("\n=== Sending prompt to LLM ===")
        print(f"Prompt: {prompt[:200]}...")  # Print first 200 chars of prompt

        # 2) Attempt to parse the LLM's response into JSON
        runner = Runner(
            app_name="clinical_structurer",
            agent=Agent(
                name="clinical_structurer",
                model=LiteLlm(model="ollama_chat/llama3.2"),
                description="Extracts clinical information and structures it as JSON",
                instruction="""You are a clinical intake assistant that extracts symptoms,
patient demographics, prior imaging, and any other relevant data from raw clinical text.
Output must be valid JSON respecting the schema given."""
            ),
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )
        
        # Create a content object for the prompt using the correct types
        content = Content(
            parts=[Part(text=prompt)],
            role="user"
        )
        
        # Create or get the session
        session = await runner.session_service.get_session(
            app_name="clinical_structurer",
            user_id="clinical_structurer_user",
            session_id="clinical_structurer_session"
        )
        
        if session is None:
            session = await runner.session_service.create_session(
                app_name="clinical_structurer",
                user_id="clinical_structurer_user",
                session_id="clinical_structurer_session",
                state={}
            )
        
        # Run the LLM with the content and collect all events
        events = []
        async for event in runner.run_async(
            user_id="clinical_structurer_user",
            session_id=session.id,
            new_message=content
        ):
            events.append(event)
        
        if not events:
            raise Exception("No response from LLM")
            
        # Get the last event's content
        last_event = events[-1]
        if not last_event.content or not last_event.content.parts:
            raise Exception("No content in LLM response")
            
        response_content = last_event.content.parts[0].text
        print(f"\n=== LLM Response ===")
        print(f"Raw response: {response_content}")
        
        # Clean the response content to ensure it's valid JSON
        content = response_content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        print(f"\n=== Cleaned content ===")
        print(f"Content: {content}")
        
        structured_data = json.loads(content)
        print(f"\n=== Parsed JSON ===")
        print(f"Structured data: {json.dumps(structured_data, indent=2)}")

        # 3) Apply default schema for missing fields
        default_structure = {
            "symptoms": [],
            "patient_info": {
                "name": None,
                "age": None,
                "sex": None,
                "history": None,
                "additional_details": {}
            },
            "previous_imaging": [],
            "additional_info": {}
        }

        # Ensure symptoms is a list
        symptoms = structured_data.get("symptoms", [])
        if isinstance(symptoms, str):
            try:
                symptoms = json.loads(symptoms)
            except:
                symptoms = [symptoms]

        # Ensure patient_info is a dict
        patient_info = structured_data.get("patient_info", {})
        if isinstance(patient_info, str):
            try:
                patient_info = json.loads(patient_info)
            except:
                patient_info = {"age": None, "sex": None, "history": None, "additional_details": {}}

        merged_data = {
            "symptoms": symptoms,
            "patient_info": {
                "age": patient_info.get("age", default_structure["patient_info"]["age"]),
                "sex": patient_info.get("sex", default_structure["patient_info"]["sex"]),
                "history": patient_info.get("history", default_structure["patient_info"]["history"]),
                "additional_details": patient_info.get("additional_details", default_structure["patient_info"]["additional_details"]),
                "name": patient_info.get("name", default_structure["patient_info"]["name"])
            },
            "previous_imaging": structured_data.get("previous_imaging", default_structure["previous_imaging"]),
            "additional_info": structured_data.get("additional_info", default_structure["additional_info"])
        }

        # 4) Load existing shared state (or start fresh)
        state = load_shared_state() or {}
        print(f"Current state: {json.dumps(state, indent=2)}")  # Debug log

        # 5) Update state with new data
        state.update({
            "status": "success",
            "initial_check": merged_data["symptoms"],
            "symptoms": merged_data["symptoms"],
            "patient_info": merged_data["patient_info"],
            "previous_imaging": merged_data["previous_imaging"],
            "additional_info": merged_data["additional_info"]
        })

        # 6) Save updated state back to disk
        if not save_shared_state(state):
            print("\n=== Error: Failed to save shared state ===")
            # Try one more time with a simpler state
            fallback_state = {
                "status": "error",
                "error_message": "Failed to save full state",
                "raw_input": text,
                "timestamp": str(datetime.datetime.now())
            }
            if save_shared_state(fallback_state):
                print("Successfully saved fallback state")
            return {
                "status": "error",
                "error_message": "Failed to save shared state"
            }

        print(f"\n=== Successfully saved state ===")
        print(f"Saved state: {json.dumps(state, indent=2)}")

        # 7) Return the merged_data plus the entire shared state
        return {
            "status": "success",
            **merged_data,
            "shared_state": state
        }

    except Exception as e:
        print(f"\n=== Error occurred ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        # Try to save error state
        error_state = {
            "status": "error",
            "error_message": str(e),
            "raw_input": text,
            "timestamp": str(datetime.datetime.now())
        }
        save_shared_state(error_state)
        logging.error(f"Error in structure_clinical_input: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Failed to structure clinical input: {str(e)}"
        }


class IntakeAgent(AgentWithTaskManager):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "intake_agent_user"

        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> Agent:
        return Agent(
            name="intake_agent_Ollama",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Structures unstructured clinical input into standardized format for the RadiologistAgent",
            instruction="""
You are a clinical intake assistant that structures raw clinical information for imaging recommendations.

=== CRITICAL FORMATTING RULES ===
ALWAYS follow this exact response format. NO exceptions:

1. If you call structure_clinical_input tool, your response MUST be:
   a) NO additional text or commentary
   b) ONLY output the exact JSON returned by the tool
   c) Format as valid JSON with proper indentation
   d) Include ALL fields from the tool response

2. If you call read_shared_state tool, your response MUST be:
   a) NO additional text or commentary  
   b) ONLY output the exact JSON returned by the tool
   c) Format as valid JSON with proper indentation

3. If there's an error, your response MUST be:
   {
     "status": "error",
     "error_message": "description of the error"
   }

=== WHEN TO USE TOOLS ===
MANDATORY: Call structure_clinical_input for ANY clinical information including:
- Patient demographics (age, sex, name)
- Symptoms or conditions  
- Medical history
- Family history
- Previous imaging or tests
- ANY medical information about a patient

ONLY call read_shared_state when:
- User explicitly asks to see current state
- User asks "what's in the shared state"
- User asks to review current information

DO NOT call tools for:
- Greetings (hello, hi, how are you)
- General questions not about patients
- Non-medical conversations

=== EXAMPLES ===

CORRECT Response after structure_clinical_input:
{
  "status": "success",
  "symptoms": ["knee pain"],
  "patient_info": {
    "name": "John Smith",
    "age": 45,
    "sex": null,
    "history": "family history of arthritis",
    "additional_details": {}
  },
  "previous_imaging": [],
  "additional_info": {},
  "shared_state": {
    "status": "success",
    "symptoms": ["knee pain"],
    "patient_info": {
      "name": "John Smith", 
      "age": 45,
      "sex": null,
      "history": "family history of arthritis",
      "additional_details": {}
    },
    "previous_imaging": [],
    "additional_info": {}
  }
}

INCORRECT Response (DO NOT DO THIS):
"I've processed the clinical information about John Smith. Here are the extracted details:
{JSON data}
The information has been saved to shared state for the radiologist."

=== VALIDATION ===
Before responding, check:
1. ✅ Did I call the appropriate tool?
2. ✅ Is my response ONLY valid JSON?
3. ✅ Did I include ALL fields from tool response?
4. ✅ Did I avoid adding any commentary or extra text?
5. ✅ Is the JSON properly formatted?

REMEMBER: Your entire response must be valid JSON that can be parsed directly. No prose, no explanations, no additional text.
""",
            tools=[structure_clinical_input, read_shared_state]
        )

    def get_processing_message(self) -> str:
        return "Processing and structuring clinical information for imaging recommendations..."