import os
import json
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.genai.types import Content, Part
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from task_manager import AgentWithTaskManager

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level = logging.ERROR)

def load_guidelines():
    """Load clinical guidelines from the text file."""
    guidelines_path = os.path.join(os.path.dirname(__file__), 'RadiologistAgent_Guidelines.txt')
    try:
        with open(guidelines_path, 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading guidelines: {e}")
        return ""

def parse_guidelines_for_imaging():
    """Parse guidelines to extract specific imaging recommendations for different clinical scenarios."""
    guidelines_text = load_guidelines()
    
    # Extract imaging recommendations based on key phrases in guidelines
    imaging_mappings = {
        "bone lesion": [
            "X-rays of primary site",
            "Bone scan or FDG-PET/CT",
            "Chest x-ray",
            "MRI with and without contrast",
            "CT with contrast"
        ],
        "symptomatic bone lesion": [
            "X-rays of primary site", 
            "Bone scan or FDG-PET/CT",
            "Chest x-ray",
            "Serum protein electrophoresis (SPEP)/labs",
            "Chest/abdomen/pelvis (C/A/P) CT with contrast"
        ],
        "sarcoma": [
            "Contrast-enhanced MRI ± CT of primary site",
            "Chest CT",
            "FDG-PET/CT (head-to-toe)",
            "Bone scan",
            "X-rays of primary site"
        ],
        "ewing sarcoma": [
            "Contrast-enhanced MRI ± CT of the primary site",
            "Chest CT", 
            "FDG-PET/CT (preferred) (head-to-toe)",
            "Bone scan",
            "Screening MRI of spine & pelvis"
        ],
        "osteosarcoma": [
            "Adequate cross-sectional imaging of primary site (x-ray, MRI, CT)",
            "Screening MRI of spinal axis",
            "C/A/P CT with contrast",
            "FDG-PET/CT (skull base to mid-thigh)",
            "Bone scan if FDG-PET/CT is negative"
        ],
        "metastatic disease": [
            "Chest CT",
            "FDG-PET/CT (head-to-toe)",
            "Bone scan",
            "MRI (with and without contrast) or CT (with contrast) of skeletal metastatic sites"
        ]
    }
    
    return imaging_mappings

def get_shared_state_path():
    """
    Return the path to the single shared_state.json file.
    Since Radiologist_agent/Agent.py is in …/agents/Radiologist_agent,
    we go up one level (to …/agents) and join "shared_state.json".
    """
    # __file__ == …/python/agents/Radiologist_agent/Agent.py
    agents_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(agents_dir, "shared_state.json")

def load_shared_state():
    """Load the shared state from shared_state.json (at …/python/agents/shared_state.json)."""
    state_path = get_shared_state_path()
    try:
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                return json.load(f)
        # If the file doesn't exist yet, return a default-shaped dictionary.
        return {
            "status": "success",
            "clinical_concern": None,
            "patient_info": {
                "age": None,
                "sex": None,
                "history": None
            },
            "previous_imaging": [],
            "imaging_results": [],
            "additional_info": {}
        }
    except Exception as e:
        logging.error(f"Error loading shared state: {e}")
        return None

def save_shared_state(state):
    """Save the shared state to shared_state.json (…/python/agents/shared_state.json)."""
    state_path = get_shared_state_path()
    try:
        # Ensure the directory exists (it should, because we are in …/agents)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error saving shared state: {e}")
        return False

def add_imaging_result(imaging_type: str, result: dict, *args, **kwargs) -> dict:
    """Add a new imaging result to the shared state. This function is called by the user to add imaging results.

    Args:
        imaging_type (str): Type of imaging study
        result (dict): Result of the imaging study including:
            - date: Date of imaging
            - findings: Imaging findings
            - status: Status of the imaging study
            - additional_details: Any additional details

    Returns:
        dict: Status of the update operation
    """
    print(f"- - - Tool: add_imaging_result called for imaging type: {imaging_type} - - -")
    
    state = load_shared_state()
    if not state:
        return {
            "status": "error",
            "error_message": "Failed to load shared state"
        }
    
    # Initialize imaging_results if it doesn't exist
    if 'imaging_results' not in state:
        state['imaging_results'] = []
    
    # Add the new imaging result
    state['imaging_results'].append({
        "type": imaging_type,
        "date": result.get('date'),
        "findings": result.get('findings'),
        "status": result.get('status', 'completed'),
        "additional_details": result.get('additional_details', {})
    })
    
    # Initialize previous_imaging if it doesn't exist
    if 'previous_imaging' not in state:
        state['previous_imaging'] = []
    
    # Update previous_imaging to include this result
    state['previous_imaging'].append({
        "type": imaging_type,
        "date": result.get('date'),
        "findings": result.get('findings')
    })
    
    if save_shared_state(state):
        return {
            "status": "success",
            "message": f"Successfully added imaging result for {imaging_type}"
        }
    else:
        return {
            "status": "error",
            "error_message": "Failed to save shared state"
        }

async def analyze_imaging_request(structured_input: dict, *args, **kwargs) -> dict:
    """Analyzes structured clinical input and recommends appropriate imaging studies based on guidelines.

    Args:
        structured_input (dict): Structured clinical information from IntakeAgent.
            Should contain:
            - clinical_concern: The main clinical concern requiring imaging
            - previous_imaging: List of previously performed imaging studies

    Returns:
        dict: A dictionary containing the imaging analysis and recommendations.
            Includes:
            - status: 'success' or 'error'
            - staging_status: 'localized' or 'metastatic'
            - recommended_imaging: List of recommended imaging studies
            - guideline_reference: Reference to clinical guideline used
            - guideline_name: Name of the guideline used
    """
    print(f"- - - Tool: analyze_imaging_request called with structured input - - -")
    print(f"Input type: {type(structured_input)}")
    print(f"Input value: {structured_input}")
    
    # Handle input parsing and validation
    if structured_input is None:
        return {"status": "error", "error_message": "No input provided"}
        
    if isinstance(structured_input, str):
        try:
            structured_input = json.loads(structured_input)
        except json.JSONDecodeError as e:
            return {"status": "error", "error_message": f"Invalid JSON input: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error_message": f"Error parsing input: {str(e)}"}
    
    if not isinstance(structured_input, dict):
        return {"status": "error", "error_message": "Input must be a dictionary or valid JSON string"}
    
    # Parse previous_imaging field properly
    previous_imaging = structured_input.get('previous_imaging', [])
    if isinstance(previous_imaging, str):
        try:
            # Try to parse as JSON string
            previous_imaging = json.loads(previous_imaging)
            print(f"Parsed previous_imaging from JSON string: {previous_imaging}")
        except json.JSONDecodeError:
            # If it's not valid JSON, treat as a single string item
            previous_imaging = [previous_imaging] if previous_imaging.strip() else []
            print(f"Treated previous_imaging as single item: {previous_imaging}")
    elif not isinstance(previous_imaging, list):
        # Convert other types to list
        previous_imaging = [previous_imaging] if previous_imaging else []
        print(f"Converted previous_imaging to list: {previous_imaging}")
    
    # Update structured_input with the properly parsed previous_imaging
    structured_input['previous_imaging'] = previous_imaging
    
    # Update shared state with new structured input
    state = load_shared_state()
    if state:
        state.update(structured_input)
        save_shared_state(state)
    
    guidelines = load_guidelines()
    clinical_concern = structured_input.get('clinical_concern', '').lower()
    
    # Get symptoms from shared state for guideline matching
    symptoms = state.get('symptoms', []) if state else []
    symptoms_text = ' '.join(symptoms).lower() if isinstance(symptoms, list) else str(symptoms).lower()
    
    print(f"DEBUG: Using symptoms for guideline matching: {symptoms}")
    print(f"DEBUG: Symptoms text for matching: '{symptoms_text}'")
    
    # Get imaging mappings from parsed guidelines
    imaging_mappings = parse_guidelines_for_imaging()
    
    # Initialize analysis with default values
    analysis = {
        "needs_metastatic_workup": False,
        "explanation": "Recommendations based on clinical guidelines",
        "confidence_level": "medium"
    }
    
    # Try to match clinical concern with guidelines
    recommended_imaging = []
    matched_guideline = None
    
    print(f"DEBUG: Attempting to match symptoms: '{symptoms_text}'")
    print(f"DEBUG: Available imaging mappings keys: {list(imaging_mappings.keys())}")
    
    # Direct matching first - use symptoms instead of clinical_concern
    # Sort concern keys by length (longest first) to prioritize more specific matches
    sorted_concern_keys = sorted(imaging_mappings.keys(), key=len, reverse=True)
    
    for concern_key in sorted_concern_keys:
        imaging_list = imaging_mappings[concern_key]
        print(f"DEBUG: Checking if '{concern_key.lower()}' is in '{symptoms_text}'")
        if concern_key.lower() in symptoms_text:
            print(f"DEBUG: MATCH FOUND! '{concern_key}' matches symptoms '{symptoms_text}'")
            print(f"DEBUG: Imaging list for '{concern_key}': {imaging_list}")
            recommended_imaging.extend(imaging_list)
            matched_guideline = concern_key
            analysis.update({
                "explanation": f"Direct match with '{concern_key}' guidelines based on symptoms",
                "confidence_level": "high"
            })
            break
        else:
            print(f"DEBUG: No match for '{concern_key}'")
    
    print(f"DEBUG: After direct matching - recommended_imaging: {recommended_imaging}")
    print(f"DEBUG: matched_guideline: {matched_guideline}")
    
    # If no direct match, use LLM for more sophisticated analysis
    if not recommended_imaging:
        try:
            # Create a prompt for the LLM to analyze the clinical concern
            prompt = f"""
As a radiologist, analyze the following clinical case against the provided guidelines and recommend ONLY the imaging studies explicitly mentioned in the guidelines.

CLINICAL CASE:
Symptoms: {symptoms}
Clinical Concern: {clinical_concern}
Patient Info: {structured_input.get('patient_info', {})}
Previous Imaging: {previous_imaging}

CLINICAL GUIDELINES:
{guidelines}

AVAILABLE IMAGING MAPPINGS:
{json.dumps(imaging_mappings, indent=2)}

ANALYSIS TASK:
1. Match the symptoms to the most appropriate guideline category
2. List ALL imaging studies recommended for this specific scenario
3. Consider patient demographics, symptoms, and previous imaging
4. Determine if metastatic workup is needed

Output ONLY a JSON object with this exact structure:
{{
    "recommended_imaging": ["imaging study 1", "imaging study 2", ...],
    "needs_metastatic_workup": true/false,
    "explanation": "brief explanation of reasoning",
    "matched_guideline": "name of matched guideline category",
    "confidence_level": "high/medium/low"
}}

IMPORTANT: Only recommend imaging studies that are explicitly mentioned in the guidelines. Do not invent new studies.
"""

            # Use the corrected Runner pattern from Intake Agent
            runner = Runner(
                app_name="guideline_matcher",
                agent=Agent(
                    name="guideline_matcher",
                    model=LiteLlm(model="ollama_chat/llama3.2"),
                    description="Matches clinical concerns with appropriate imaging guidelines",
                    instruction="""You are a specialized radiologist assistant that recommends imaging studies based on clinical guidelines.
                    Return only valid JSON. Consider the full clinical context when making recommendations."""
                ),
                session_service=InMemorySessionService(),
                artifact_service=InMemoryArtifactService(),
                memory_service=InMemoryMemoryService(),
            )
            
            # Create content object with proper structure
            content = Content(
                parts=[Part(text=prompt)],
                role="user"
            )
            
            # Create or get session
            session = await runner.session_service.get_session(
                app_name="guideline_matcher",
                user_id="guideline_matcher_user",
                session_id="guideline_matcher_session"
            )
            
            if session is None:
                session = await runner.session_service.create_session(
                    app_name="guideline_matcher",
                    user_id="guideline_matcher_user",
                    session_id="guideline_matcher_session",
                    state={}
                )
            
            # Run the LLM with correct parameters
            events = []
            async for event in runner.run_async(
                user_id="guideline_matcher_user",
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
            
            # Clean the response content to ensure it's valid JSON
            content_text = response_content.strip()
            if content_text.startswith('```json'):
                content_text = content_text[7:]
            if content_text.endswith('```'):
                content_text = content_text[:-3]
            content_text = content_text.strip()
            
            # Parse the LLM's response and update analysis
            llm_analysis = json.loads(content_text)
            recommended_imaging = llm_analysis.get('recommended_imaging', [])
            matched_guideline = llm_analysis.get('matched_guideline', 'general guidelines')
            
            # Update analysis with LLM results
            analysis.update({
                "needs_metastatic_workup": llm_analysis.get('needs_metastatic_workup', False),
                "explanation": llm_analysis.get('explanation', 'LLM analysis of clinical guidelines'),
                "confidence_level": llm_analysis.get('confidence_level', 'medium')
            })
            
        except Exception as e:
            logging.error(f"Error in LLM analysis: {e}")
            # Fallback to basic recommendations for bone lesions
            if 'bone' in clinical_concern:
                recommended_imaging = imaging_mappings.get('bone lesion', [])
                matched_guideline = 'bone lesion (fallback)'
            else:
                recommended_imaging = ["X-rays of primary site", "Chest x-ray"]
                matched_guideline = 'general (fallback)'
            
            analysis.update({
                "explanation": f"Fallback recommendation due to analysis error: {str(e)}",
                "confidence_level": "low"
            })
    
    # Filter out imaging studies that have already been performed
    performed_types = set()
    if previous_imaging:
        for img in previous_imaging:
            if isinstance(img, dict) and 'type' in img:
                performed_types.add(img['type'].lower())
            elif isinstance(img, str):
                performed_types.add(img.lower())
        print(f"Performed imaging types identified: {performed_types}")
    
    # Filter recommendations
    filtered_imaging = []
    for img in recommended_imaging:
        img_lower = img.lower()
        already_done = any(performed.lower() in img_lower or img_lower in performed.lower() 
                          for performed in performed_types)
        if not already_done:
            filtered_imaging.append(img)
        else:
            print(f"Filtered out '{img}' - already performed")
    
    print(f"Final filtered imaging recommendations: {filtered_imaging}")
    
    # Create user message listing all recommended imaging studies
    if filtered_imaging:
        imaging_list_str = "\n".join(f"- {study}" for study in filtered_imaging)
        user_message = (
            f"Based on the symptoms '{symptoms}' and matching guideline '{matched_guideline}', "
            f"the following imaging studies are recommended:\n\n{imaging_list_str}\n\n"
            "I will now ask you for details about each study, one at a time."
        )
    else:
        user_message = (
            "All recommended imaging studies have already been performed according to the previous imaging history. "
            "No additional imaging is needed at this time."
        )
    
    # Save this message to the state for the agent to use in its next turn
    if state:
        state['last_imaging_recommendations_message'] = user_message
        state['clinical_concern'] = clinical_concern
        state['symptoms_used_for_matching'] = symptoms_text
        save_shared_state(state)

    return {
        "status": "success",
        "staging_status": "metastatic" if analysis.get('needs_metastatic_workup', False) else "localized",
        "recommended_imaging": filtered_imaging,
        "explanation": analysis.get('explanation', 'Recommendations based on clinical guidelines'),
        "matched_guideline": matched_guideline or 'general guidelines',
        "confidence_level": analysis.get('confidence_level', 'medium'),
        "user_message": user_message,
        "message": "Please provide the results of the recommended imaging studies using the add_imaging_result tool." if filtered_imaging else "No additional imaging needed."
    }

class RadiologistAgent(AgentWithTaskManager):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "application/json"]
    
    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "radiologist_agent_user"
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
            name="radiologist_agent_Ollama",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Analyzes structured clinical input and recommends appropriate imaging studies based on clinical guidelines",
            instruction="""
    You are a specialized radiologist assistant. Your workflow is:

    1. Receive structured clinical input from the IntakeAgent.
    2. Use the 'analyze_imaging_request' tool to decide which imaging studies to order.
    3. Once the tool returns, you will send the user a message that:
    a. Lists all the recommended imagaing based on the guidelines
    b. For *each* recommended study, you will collect the following fields in separate sub-turns:
        1. "What was the date of this imaging study?"
        2. "What were the findings?"
        3. "What is the status (e.g., completed, pending)?"
        4. "Any additional details?"
    c. Only after you have gathered *all four fields* for study 1 should you move on to study 2.
    d. Store each completed study as soon as you have all four fields, by calling the `add_imaging_result` tool.

    4. After you ask for one field, wait for the user's reply. Once they answer, you immediately prompt the next field. If the user says "I don't know," you can record an empty string (""), but continue to the next field.

    5. Once you have used `add_imaging_result` for every recommended imaging, send a final confirmation: 
    "All imaging results have been recorded in shared state. Let me know if you need anything else."

    Note: 
    - Do not ask for multiple fields in the same sentence. Ask exactly one question per turn.
    - Use the ADK tool `add_imaging_result(imaging_type, result_dict)` whenever you have all four fields for a study.
    """
    ,
            tools=[analyze_imaging_request, add_imaging_result]
        )

        
    def get_processing_message(self) -> str:
        return "Analyzing structured clinical input and determining appropriate imaging studies based on guidelines..."
