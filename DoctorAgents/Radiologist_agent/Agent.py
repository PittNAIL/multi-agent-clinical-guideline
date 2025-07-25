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
from datetime import datetime

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

async def collect_imaging_details(current_study: str | None = None, *args, **kwargs) -> dict:
    """Systematically collects details for each recommended imaging study.
    
    Args:
        current_study (str): The current imaging study to collect details for. 
                           If None, starts with the first study in the queue.
    
    Returns:
        dict: Contains the next question to ask or completion status
    """
    print(f"- - - Tool: collect_imaging_details called for study: {current_study} - - -")
    
    state = load_shared_state()
    if not state:
        return {"status": "error", "error_message": "Failed to load shared state"}
    
    # Get imaging recommendations from shared state
    imaging_recs = state.get('imaging_recommendations', {})
    imaging_studies = imaging_recs.get('imaging_studies', {})
    recommended_studies = imaging_studies.get('recommended', [])
    
    if not recommended_studies:
        return {
            "status": "complete",
            "message": "No imaging studies to collect details for."
        }
    
    # Initialize or get the collection workflow state
    if 'imaging_collection_workflow' not in state:
        state['imaging_collection_workflow'] = {
            "current_study_index": 0,
            "current_field": "date",  # date, findings, status, additional_details
            "collected_data": {},
            "completed_studies": []
        }
    
    workflow = state['imaging_collection_workflow']
    
    # If current_study is provided, we might be resuming or starting a specific study
    if current_study:
        # Find the study in the recommended list
        study_names = [study['study_name'] for study in recommended_studies]
        if current_study in study_names:
            workflow['current_study_index'] = study_names.index(current_study)
            workflow['current_field'] = "date"  # Reset to first field
        else:
            return {"status": "error", "error_message": f"Study '{current_study}' not found in recommendations"}
    
    # Check if we've completed all studies
    if workflow['current_study_index'] >= len(recommended_studies):
        return {
            "status": "complete",
            "message": "All imaging studies have been processed. Collection workflow complete."
        }
    
    # Get current study
    current_study_data = recommended_studies[workflow['current_study_index']]
    study_name = current_study_data['study_name']
    
    # Initialize data collection for this study if not exists
    if study_name not in workflow['collected_data']:
        workflow['collected_data'][study_name] = {}
    
    # Determine what question to ask based on current field
    field_questions = {
        "date": f"What was the date when the '{study_name}' was performed? (Please provide in MM/DD/YYYY format, or say 'unknown' if not available)",
        "findings": f"What were the findings from the '{study_name}'? (Please describe the imaging findings and any abnormalities noted)",
        "status": f"What is the current status of the '{study_name}'? (e.g., 'completed', 'pending', 'scheduled', etc.)",
        "additional_details": f"Are there any additional details about the '{study_name}' that should be recorded? (e.g., contrast used, technical notes, follow-up recommendations, or say 'none' if no additional details)"
    }
    
    current_field = workflow['current_field']
    
    # Save the updated workflow state
    save_shared_state(state)
    
    return {
        "status": "collecting",
        "current_study": study_name,
        "current_field": current_field,
        "question": field_questions[current_field],
        "progress": f"Study {workflow['current_study_index'] + 1} of {len(recommended_studies)} - Field: {current_field}",
        "message": field_questions[current_field]
    }

async def record_imaging_detail(study_name: str, field: str, value: str, *args, **kwargs) -> dict:
    """Records a specific detail for an imaging study and advances the collection workflow.
    
    Args:
        study_name (str): Name of the imaging study
        field (str): Field being recorded (date, findings, status, additional_details)  
        value (str): Value to record for the field
    
    Returns:
        dict: Next step in the workflow or completion status
    """
    print(f"- - - Tool: record_imaging_detail called for {study_name}, field: {field}, value: {value} - - -")
    
    state = load_shared_state()
    if not state:
        return {"status": "error", "error_message": "Failed to load shared state"}
    
    workflow = state.get('imaging_collection_workflow', {})
    
    if not workflow:
        return {"status": "error", "error_message": "No active collection workflow found"}
    
    # Record the value
    if study_name not in workflow['collected_data']:
        workflow['collected_data'][study_name] = {}
    
    workflow['collected_data'][study_name][field] = value if value.lower() not in ['unknown', 'none', ''] else ""
    
    # Determine next field in sequence
    field_sequence = ["date", "findings", "status", "additional_details"]
    current_field_index = field_sequence.index(field)
    
    if current_field_index < len(field_sequence) - 1:
        # Move to next field for same study
        workflow['current_field'] = field_sequence[current_field_index + 1]
        save_shared_state(state)
        
        # Ask the next question for this study
        return await collect_imaging_details()
    else:
        # All fields collected for this study - save it using add_imaging_result
        collected_data = workflow['collected_data'][study_name]
        
        result_data = {
            "date": collected_data.get('date', ''),
            "findings": collected_data.get('findings', ''),
            "status": collected_data.get('status', 'completed'),
            "additional_details": collected_data.get('additional_details', '')
        }
        
        # Use the existing add_imaging_result function to save to shared state
        save_result = add_imaging_result(study_name, result_data)
        
        if save_result.get('status') == 'success':
            # Mark study as completed and move to next study
            workflow['completed_studies'].append(study_name)
            workflow['current_study_index'] += 1
            workflow['current_field'] = "date"  # Reset to first field for next study
            
            # Clear collected data for this study to save memory
            if study_name in workflow['collected_data']:
                del workflow['collected_data'][study_name]
            
            save_shared_state(state)
            
            # Check if more studies remain
            imaging_recs = state.get('imaging_recommendations', {})
            imaging_studies = imaging_recs.get('imaging_studies', {})
            recommended_studies = imaging_studies.get('recommended', [])
            
            if workflow['current_study_index'] >= len(recommended_studies):
                # All studies completed
                completion_message = (
                    f"✓ Successfully recorded details for '{study_name}'.\n\n"
                    f"All {len(recommended_studies)} recommended imaging studies have been processed and saved to shared state:\n"
                )
                for i, completed_study in enumerate(workflow['completed_studies'], 1):
                    completion_message += f"{i}. {completed_study}\n"
                
                completion_message += "\nImaging workflow complete! The data is now available for the Pathologist agent."
                
                # Clean up workflow state
                if 'imaging_collection_workflow' in state:
                    del state['imaging_collection_workflow']
                save_shared_state(state)
                
                return {
                    "status": "workflow_complete",
                    "message": completion_message
                }
            else:
                # Move to next study
                next_study = recommended_studies[workflow['current_study_index']]['study_name']
                return {
                    "status": "study_completed",
                    "message": f"✓ Successfully recorded details for '{study_name}'. Now moving to the next study: '{next_study}'.",
                    "next_action": "continue_collection"
                }
        else:
            return {
                "status": "error", 
                "error_message": f"Failed to save imaging result: {save_result.get('error_message', 'Unknown error')}"
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
        imaging_list_str = "\n".join(f"{i+1}. {study}" for i, study in enumerate(filtered_imaging))
        user_message = (
            f"Based on the symptoms '{symptoms}' and matching guideline '{matched_guideline}', "
            f"the following {len(filtered_imaging)} imaging studies are recommended:\n\n{imaging_list_str}\n\n"
            f"I will now systematically collect the findings for each study. "
            f"Let me start with the first one."
        )
    else:
        user_message = (
            "All recommended imaging studies have already been performed according to the previous imaging history. "
            "No additional imaging is needed at this time."
        )
    
    # Save structured imaging recommendations to the state instead of just a text message
    if state:
        # Create structured imaging recommendations object
        imaging_recommendations = {
            "analysis_timestamp": structured_input.get('timestamp', datetime.now().isoformat()),
            "matched_guideline": {
                "name": matched_guideline or 'general guidelines',
                "confidence_level": analysis.get('confidence_level', 'medium'),
                "explanation": analysis.get('explanation', 'Recommendations based on clinical guidelines')
            },
            "clinical_context": {
                "symptoms": symptoms if isinstance(symptoms, list) else [symptoms] if symptoms else [],
                "symptoms_text": symptoms_text,
                "clinical_concern": clinical_concern,
                "needs_metastatic_workup": analysis.get('needs_metastatic_workup', False),
                "staging_status": "metastatic" if analysis.get('needs_metastatic_workup', False) else "localized"
            },
            "imaging_studies": {
                "recommended": [
                    {
                        "study_name": study,
                        "status": "recommended",
                        "priority": "standard",
                        "guideline_source": matched_guideline or 'general guidelines'
                    } for study in filtered_imaging
                ],
                "already_performed": [
                    {
                        "study_name": img.get('type') if isinstance(img, dict) else str(img),
                        "status": "completed",
                        "date": img.get('date') if isinstance(img, dict) else None,
                        "findings": img.get('findings') if isinstance(img, dict) else None
                    } for img in previous_imaging
                ],
                "total_recommended_count": len(filtered_imaging),
                "total_performed_count": len(previous_imaging)
            },
            "workflow_status": {
                "current_step": "imaging_recommendation",
                "next_action": "collect_imaging_details" if filtered_imaging else "proceed_to_next_specialist",
                "completion_status": "pending" if filtered_imaging else "complete"
            }
        }
        
        # Save the structured recommendations
        state['imaging_recommendations'] = imaging_recommendations
        state['clinical_concern'] = clinical_concern
        
        # Keep the text message for backward compatibility but mark it as deprecated
        state['last_imaging_recommendations_message'] = user_message
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
        "start_collection": len(filtered_imaging) > 0,
        "message": user_message
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

1. **Initial Analysis**: Use the 'analyze_imaging_request' tool to analyze clinical input and determine which imaging studies to recommend based on guidelines.

2. **Show All Recommendations**: After analysis, present ALL recommended imaging studies in a numbered list to give the user a complete overview.

3. **Systematic Collection**: Use the 'collect_imaging_details' tool to start systematic collection of findings for each study. This tool will:
   - Guide you through each study one by one
   - Ask for specific fields in order: date, findings, status, additional_details
   - Tell you exactly what question to ask next

4. **Record Each Response**: When the user provides an answer, use the 'record_imaging_detail' tool to record their response and get the next question.

5. **Follow the Workflow**: The tools will guide you through the entire process. Simply:
   - Ask the question provided by the tool
   - Wait for user response  
   - Call record_imaging_detail with their answer
   - Ask the next question provided by the tool
   - Repeat until all studies are complete

6. **Completion**: When all imaging studies have been processed, confirm that all data has been saved to shared state.

IMPORTANT RULES:
- Always use the tools to guide the workflow - don't improvise questions
- Ask exactly ONE question at a time as directed by the tools
- Wait for user response before proceeding to next field
- Use the exact questions provided by collect_imaging_details
- Record every answer using record_imaging_detail before asking the next question
- Trust the tools to manage the workflow state and progression

The tools handle all the complexity - your job is to be the interface between the user and the systematic collection process.
"""
            ,
            tools=[analyze_imaging_request, add_imaging_result, collect_imaging_details, record_imaging_detail]
        )

        
    def get_processing_message(self) -> str:
        return "Analyzing structured clinical input and determining appropriate imaging studies based on guidelines..."
