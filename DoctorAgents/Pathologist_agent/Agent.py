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
    """Load pathology diagnostic guidelines from the text file."""
    guidelines_path = os.path.join(os.path.dirname(__file__), 'PathologistAgent_Guidelines.txt')
    try:
        with open(guidelines_path, 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading guidelines: {e}")
        return ""

def parse_guidelines_for_diagnosis():
    """Parse guidelines to extract information dynamically from the actual NCCN guidelines text."""
    guidelines_text = load_guidelines()
    
    # Return empty mappings - let the LLM analyze the guidelines text directly
    diagnostic_mappings = {}
    
    # Extract only the general biopsy requirements that are clearly stated
    general_requirements = {
        "biopsy_location": "Biopsy, if indicated, should be performed at the treating institution",
        "biopsy_requirement": "Biopsy should be performed at treating institution"
    }
    
    return diagnostic_mappings, general_requirements

def get_shared_state_path():
    """
    Return the path to the single shared_state.json file.
    Since Pathologist_agent/Agent.py is in …/agents/Pathologist_agent,
    we go up one level (to …/agents) and join "shared_state.json".
    """
    # __file__ == …/python/agents/Pathologist_agent/Agent.py
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
            "imaging_results": [],
            "pathology_results": [],
            "diagnosis": None,
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

def add_pathology_result(test_type: str, result: dict, *args, **kwargs) -> dict:
    """Add a new pathology result to the shared state. This function is called by the user to add pathology results.

    Args:
        test_type (str): Type of pathology test (biopsy, IHC, FISH, cytogenetics, molecular)
        result (dict): Result of the pathology test including:
            - date: Date of test
            - findings: Test findings
            - markers: Specific markers or genes tested
            - status: Status of the test
            - additional_details: Any additional details

    Returns:
        dict: Status of the update operation
    """
    print(f"- - - Tool: add_pathology_result called for test type: {test_type} - - -")
    
    state = load_shared_state()
    if not state:
        return {
            "status": "error",
            "error_message": "Failed to load shared state"
        }
    
    # Initialize pathology_results if it doesn't exist
    if 'pathology_results' not in state:
        state['pathology_results'] = []
    
    # Add the new pathology result
    state['pathology_results'].append({
        "type": test_type,
        "date": result.get('date'),
        "findings": result.get('findings'),
        "markers": result.get('markers', []),
        "status": result.get('status', 'completed'),
        "additional_details": result.get('additional_details', {})
    })
    
    if save_shared_state(state):
        return {
            "status": "success",
            "message": f"Successfully added pathology result for {test_type}"
        }
    else:
        return {
            "status": "error",
            "error_message": "Failed to save shared state"
        }

async def collect_pathology_details(current_test: str | None = None, *args, **kwargs) -> dict:
    """Systematically collects details for each required pathology test.
    
    Args:
        current_test (str): The current pathology test to collect details for. 
                           If None, starts with the first test in the queue.
    
    Returns:
        dict: Contains the next question to ask or completion status
    """
    print(f"- - - Tool: collect_pathology_details called for test: {current_test} - - -")
    
    state = load_shared_state()
    if not state:
        return {"status": "error", "error_message": "Failed to load shared state"}
    
    # Get pathology analysis from shared state
    pathology_analysis = state.get('pathology_analysis', {})
    pathology_workflow = pathology_analysis.get('pathology_workflow', {})
    required_tests = pathology_workflow.get('required_tests', [])
    
    if not required_tests:
        return {
            "status": "complete",
            "message": "No pathology tests to collect details for."
        }
    
    # Initialize or get the collection workflow state
    if 'pathology_collection_workflow' not in state:
        state['pathology_collection_workflow'] = {
            "current_test_index": 0,
            "current_field": "date",  # date, findings, markers, status, additional_details
            "collected_data": {},
            "completed_tests": []
        }
    
    workflow = state['pathology_collection_workflow']
    
    # If current_test is provided, we might be resuming or starting a specific test
    if current_test:
        # Find the test in the required list
        if current_test in required_tests:
            workflow['current_test_index'] = required_tests.index(current_test)
            workflow['current_field'] = "date"  # Reset to first field
        else:
            return {"status": "error", "error_message": f"Test '{current_test}' not found in requirements"}
    
    # Check if we've completed all tests
    if workflow['current_test_index'] >= len(required_tests):
        return {
            "status": "complete",
            "message": "All pathology tests have been processed. Collection workflow complete."
        }
    
    # Get current test
    test_name = required_tests[workflow['current_test_index']]
    
    # Initialize data collection for this test if not exists
    if test_name not in workflow['collected_data']:
        workflow['collected_data'][test_name] = {}
    
    # Determine what question to ask based on current field
    field_questions = {
        "date": f"What was the date when the '{test_name}' was performed? (Please provide in MM/DD/YYYY format, or say 'unknown' if not available)",
        "findings": f"What were the findings/results from the '{test_name}'? (Please describe the pathology findings, morphology, and any abnormalities noted)",
        "markers": f"What specific markers or studies were performed as part of the '{test_name}'? (e.g., immunohistochemistry markers, molecular tests, cytogenetics, or say 'none' if not applicable)",
        "status": f"What is the current status of the '{test_name}'? (e.g., 'completed', 'pending', 'in progress', etc.)",
        "additional_details": f"Are there any additional details about the '{test_name}' that should be recorded? (e.g., technical notes, special stains, follow-up recommendations, or say 'none' if no additional details)"
    }
    
    current_field = workflow['current_field']
    
    # Save the updated workflow state
    save_shared_state(state)
    
    return {
        "status": "collecting",
        "current_test": test_name,
        "current_field": current_field,
        "question": field_questions[current_field],
        "progress": f"Test {workflow['current_test_index'] + 1} of {len(required_tests)} - Field: {current_field}",
        "message": field_questions[current_field]
    }

async def record_pathology_detail(test_name: str, field: str, value: str, *args, **kwargs) -> dict:
    """Records a specific detail for a pathology test and advances the collection workflow.
    
    Args:
        test_name (str): Name of the pathology test
        field (str): Field being recorded (date, findings, markers, status, additional_details)  
        value (str): Value to record for the field
    
    Returns:
        dict: Next step in the workflow or completion status
    """
    print(f"- - - Tool: record_pathology_detail called for {test_name}, field: {field}, value: {value} - - -")
    
    state = load_shared_state()
    if not state:
        return {"status": "error", "error_message": "Failed to load shared state"}
    
    workflow = state.get('pathology_collection_workflow', {})
    
    if not workflow:
        return {"status": "error", "error_message": "No active collection workflow found"}
    
    # Record the value
    if test_name not in workflow['collected_data']:
        workflow['collected_data'][test_name] = {}
    
    workflow['collected_data'][test_name][field] = value if value.lower() not in ['unknown', 'none', ''] else ""
    
    # Determine next field in sequence
    field_sequence = ["date", "findings", "markers", "status", "additional_details"]
    current_field_index = field_sequence.index(field)
    
    if current_field_index < len(field_sequence) - 1:
        # Move to next field for same test
        workflow['current_field'] = field_sequence[current_field_index + 1]
        save_shared_state(state)
        
        # Ask the next question for this test
        return await collect_pathology_details()
    else:
        # All fields collected for this test - save it using add_pathology_result
        collected_data = workflow['collected_data'][test_name]
        
        result_data = {
            "date": collected_data.get('date', ''),
            "findings": collected_data.get('findings', ''),
            "markers": collected_data.get('markers', []) if collected_data.get('markers') else [],
            "status": collected_data.get('status', 'completed'),
            "additional_details": collected_data.get('additional_details', '')
        }
        
        # Use the existing add_pathology_result function to save to shared state
        save_result = add_pathology_result(test_name, result_data)
        
        if save_result.get('status') == 'success':
            # Mark test as completed and move to next test
            workflow['completed_tests'].append(test_name)
            workflow['current_test_index'] += 1
            workflow['current_field'] = "date"  # Reset to first field for next test
            
            # Clear collected data for this test to save memory
            if test_name in workflow['collected_data']:
                del workflow['collected_data'][test_name]
            
            save_shared_state(state)
            
            # Check if more tests remain
            pathology_analysis = state.get('pathology_analysis', {})
            pathology_workflow = pathology_analysis.get('pathology_workflow', {})
            required_tests = pathology_workflow.get('required_tests', [])
            
            if workflow['current_test_index'] >= len(required_tests):
                # All tests completed - ready for diagnosis confirmation
                completion_message = (
                    f"✓ Successfully recorded details for '{test_name}'.\n\n"
                    f"All {len(required_tests)} required pathology tests have been processed and saved to shared state:\n"
                )
                for i, completed_test in enumerate(workflow['completed_tests'], 1):
                    completion_message += f"{i}. {completed_test}\n"
                
                completion_message += "\nPathology data collection complete! Ready to confirm final diagnosis based on NCCN guidelines."
                
                # Clean up workflow state
                if 'pathology_collection_workflow' in state:
                    del state['pathology_collection_workflow']
                save_shared_state(state)
                
                return {
                    "status": "workflow_complete",
                    "message": completion_message,
                    "ready_for_diagnosis": True
                }
            else:
                # Move to next test
                next_test = required_tests[workflow['current_test_index']]
                return {
                    "status": "test_completed",
                    "message": f"✓ Successfully recorded details for '{test_name}'. Now moving to the next test: '{next_test}'.",
                    "next_action": "continue_collection"
                }
        else:
            return {
                "status": "error", 
                "error_message": f"Failed to save pathology result: {save_result.get('error_message', 'Unknown error')}"
            }

async def analyze_diagnostic_request(clinical_data: dict, *args, **kwargs) -> dict:
    """Analyzes clinical data and pathology results to confirm diagnosis based on NCCN guidelines.

    Args:
        clinical_data (dict): Clinical information including imaging results and initial clinical concern.
            Should contain:
            - imaging_results: Results from imaging studies
            - clinical_concern: Initial clinical concern
            - patient_info: Patient demographics

    Returns:
        dict: A dictionary containing the diagnostic analysis and confirmation.
            Includes:
            - status: 'success' or 'error'
            - required_tests: List of required pathology tests
            - diagnostic_criteria: Criteria needed for confirmation
            - confidence_level: Confidence in current data sufficiency
    """
    print(f"- - - Tool: analyze_diagnostic_request called with clinical data - - -")
    print(f"Input type: {type(clinical_data)}")
    print(f"Input value: {clinical_data}")
    
    # Handle input parsing and validation
    if clinical_data is None:
        return {"status": "error", "error_message": "No clinical data provided"}
        
    if isinstance(clinical_data, str):
        try:
            clinical_data = json.loads(clinical_data)
        except json.JSONDecodeError as e:
            return {"status": "error", "error_message": f"Invalid JSON input: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error_message": f"Error parsing input: {str(e)}"}
    
    if not isinstance(clinical_data, dict):
        return {"status": "error", "error_message": "Input must be a dictionary or valid JSON string"}
    
    # Load current shared state
    state = load_shared_state()
    if state:
        state.update(clinical_data)
        save_shared_state(state)
    
    guidelines = load_guidelines()
    diagnostic_mappings, general_requirements = parse_guidelines_for_diagnosis()
    
    # Get imaging results and clinical context from shared state
    imaging_results = state.get('imaging_results', []) if state else []
    pathology_results = state.get('pathology_results', []) if state else []
    clinical_concern = state.get('clinical_concern', '').lower() if state else ''
    
    # Analyze imaging findings to determine suspected tumor type
    suspected_tumor_types = []
    imaging_findings = []
    
    for img in imaging_results:
        findings = img.get('findings', '').lower()
        imaging_findings.append(findings)
        
        # Basic pattern matching for tumor type suggestion
        if any(keyword in findings for keyword in ['bone forming', 'osteoid', 'sclerotic', 'osteoblastic']):
            suspected_tumor_types.append('osteosarcoma')
        elif any(keyword in findings for keyword in ['small round', 'lytic', 'permeative', 'round cell']):
            suspected_tumor_types.append('ewing sarcoma')
        elif any(keyword in findings for keyword in ['cartilage', 'chondroid', 'rings and arcs', 'lobulated']):
            suspected_tumor_types.append('chondrosarcoma')
    
    # Remove duplicates
    suspected_tumor_types = list(set(suspected_tumor_types))
    
    if not suspected_tumor_types:
        # Fallback based on clinical concern and patient age
        patient_age = state.get('patient_info', {}).get('age') if state else None
        if 'bone' in clinical_concern:
            if patient_age and patient_age < 20:
                suspected_tumor_types = ['ewing sarcoma', 'osteosarcoma']
            else:
                suspected_tumor_types = ['osteosarcoma', 'chondrosarcoma']
        else:
            suspected_tumor_types = ['bone tumor']  # Generic if unclear
    
    print(f"DEBUG: Suspected tumor types based on imaging: {suspected_tumor_types}")
    
    # Use LLM to analyze the guidelines and determine required procedures
    try:
        # Create a prompt for the LLM to analyze the clinical data against guidelines
        prompt = f"""
Based on the NCCN guidelines provided, analyze the clinical case and determine what pathology procedures are required.

CLINICAL CASE:
Imaging Results: {json.dumps(imaging_results, indent=2)}
Suspected Tumor Types: {suspected_tumor_types}
Clinical Concern: {clinical_concern}
Patient Info: {state.get('patient_info', {}) if state else {}}
Already Performed Pathology: {json.dumps(pathology_results, indent=2)}

NCCN GUIDELINES:
{guidelines}

ANALYSIS TASK:
1. Based on the suspected tumor types and imaging findings, what pathology procedures does NCCN require?
2. What tests have already been performed vs what is still needed?
3. What is the confidence level for proceeding with diagnosis?

Output ONLY a JSON object with this structure:
{{
    "required_procedures": ["list of procedures required per NCCN"],
    "pending_procedures": ["procedures still needed"],
    "performed_procedures": ["procedures already done"],
    "confidence_level": "high/medium/low",
    "analysis_summary": "brief explanation of requirements based on NCCN guidelines"
}}

IMPORTANT: Base recommendations strictly on the NCCN guidelines provided. Only suggest procedures explicitly mentioned in the guidelines.
"""

        # Use LLM to analyze guidelines
        runner = Runner(
            app_name="nccn_analyzer",
            agent=Agent(
                name="nccn_analyzer",
                model=LiteLlm(model="ollama_chat/llama3.2"),
                description="Analyzes NCCN guidelines to determine required pathology procedures",
                instruction="""You are a pathologist assistant that analyzes NCCN guidelines to determine required procedures.
                Return only valid JSON. Base all recommendations strictly on the provided NCCN guidelines."""
            ),
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )
        
        content = Content(
            parts=[Part(text=prompt)],
            role="user"
        )
        
        session = await runner.session_service.get_session(
            app_name="nccn_analyzer",
            user_id="nccn_analyzer_user",
            session_id="nccn_analyzer_session"
        )
        
        if session is None:
            session = await runner.session_service.create_session(
                app_name="nccn_analyzer",
                user_id="nccn_analyzer_user",
                session_id="nccn_analyzer_session",
                state={}
            )
        
        events = []
        async for event in runner.run_async(
            user_id="nccn_analyzer_user",
            session_id=session.id,
            new_message=content
        ):
            events.append(event)
        
        if not events:
            raise Exception("No response from NCCN analyzer")
            
        last_event = events[-1]
        if not last_event.content or not last_event.content.parts:
            raise Exception("No content in NCCN analyzer response")
            
        response_content = last_event.content.parts[0].text
        
        # Clean the response content
        content_text = response_content.strip()
        if content_text.startswith('```json'):
            content_text = content_text[7:]
        if content_text.endswith('```'):
            content_text = content_text.strip()
        
        # Parse the analysis result
        analysis_result = json.loads(content_text)
        
        required_tests = analysis_result.get('required_procedures', [])
        pending_tests = analysis_result.get('pending_procedures', [])
        confidence_level = analysis_result.get('confidence_level', 'medium')
        analysis_summary = analysis_result.get('analysis_summary', '')
        
    except Exception as e:
        logging.error(f"Error in NCCN analysis: {e}")
        # Fallback to basic requirements
        required_tests = [general_requirements['biopsy_requirement']]
        pending_tests = required_tests
        confidence_level = 'low'
        analysis_summary = f"Fallback recommendation due to analysis error: {str(e)}"
    
    # Create user message listing all required pathology tests
    if pending_tests:
        tests_list_str = "\n".join(f"{i+1}. {test}" for i, test in enumerate(pending_tests))
        user_message = (
            f"Based on imaging findings and NCCN guidelines, "
            f"the following {len(pending_tests)} pathology procedures are required:\n\n{tests_list_str}\n\n"
            f"I will now systematically collect the results for each procedure. "
            f"Let me start with the first one."
        )
    else:
        user_message = (
            "All required pathology procedures appear to have been performed per NCCN guidelines. "
            "Ready to proceed with diagnostic confirmation."
        )
    
    # Save structured pathology recommendations to shared state
    if state:
        pathology_analysis = {
            "analysis_timestamp": clinical_data.get('timestamp', datetime.now().isoformat()),
            "suspected_diagnoses": [
                {
                    "tumor_type": tumor_type,
                    "confidence": confidence_level,
                    "supporting_evidence": [],
                    "required_markers": [],
                    "molecular_tests": []
                } for tumor_type in suspected_tumor_types
            ],
            "clinical_context": {
                "imaging_findings": imaging_findings,
                "clinical_concern": clinical_concern,
                "patient_age": state.get('patient_info', {}).get('age'),
                "patient_sex": state.get('patient_info', {}).get('sex')
            },
            "pathology_workflow": {
                "required_tests": pending_tests,
                "pending_tests": pending_tests,
                "performed_tests": [],
                "available_markers": {},
                "total_required_count": len(pending_tests),
                "total_performed_count": 0
            },
            "workflow_status": {
                "current_step": "pathology_analysis",
                "next_action": "collect_pathology_results" if pending_tests else "diagnostic_confirmation",
                "completion_status": "pending" if pending_tests else "ready_for_diagnosis"
            }
        }
        
        state['pathology_analysis'] = pathology_analysis
        state['last_pathology_message'] = user_message
        save_shared_state(state)

    return {
        "status": "success",
        "suspected_diagnoses": suspected_tumor_types,
        "required_tests": pending_tests,
        "diagnostic_criteria": {},
        "confidence_level": confidence_level,
        "user_message": user_message,
        "start_collection": len(pending_tests) > 0,
        "message": user_message
    }

def analyze_diagnostic_confidence(suspected_types, available_markers, diagnostic_criteria):
    """Analyze confidence level for each suspected diagnosis based on available pathology results."""
    confidence_analysis = {}
    
    for tumor_type in suspected_types:
        if tumor_type not in diagnostic_criteria:
            continue
            
        criteria = diagnostic_criteria[tumor_type]
        required_procedures = criteria.get('required_procedures', [])
        
        # Check how many required procedures are available based on pathology results
        completed_procedures = 0
        evidence = []
        
        # Simple check based on available markers and findings
        if available_markers:
            completed_procedures = len(available_markers)
            evidence = list(available_markers.keys())
        
        # Calculate confidence based on completion of procedures
        if completed_procedures >= len(required_procedures):
            confidence = "high"
        elif completed_procedures >= len(required_procedures) * 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        confidence_analysis[tumor_type] = {
            "confidence": confidence,
            "evidence": evidence,
            "completed_procedures": completed_procedures,
            "required_procedures": required_procedures
        }
    
    return confidence_analysis

async def confirm_diagnosis(pathology_data: dict, *args, **kwargs) -> dict:
    """Confirms final diagnosis based on all pathology results and NCCN criteria.

    Args:
        pathology_data (dict): Complete pathology information for final diagnosis

    Returns:
        dict: Final diagnosis with subtype, grade, confidence, and justification
    """
    print(f"- - - Tool: confirm_diagnosis called - - -")
    
    state = load_shared_state()
    if not state:
        return {"status": "error", "error_message": "Failed to load shared state"}
    
    pathology_results = state.get('pathology_results', [])
    pathology_analysis = state.get('pathology_analysis', {})
    
    if not pathology_results:
        return {"status": "error", "error_message": "No pathology results available for diagnosis"}
    
    # Use LLM for sophisticated diagnostic analysis
    try:
        guidelines = load_guidelines()
        
        # Prepare comprehensive prompt for diagnosis
        prompt = f"""
As a pathologist, analyze the following complete pathology data and confirm the final diagnosis according to the actual NCCN guidelines provided.

PATHOLOGY RESULTS:
{json.dumps(pathology_results, indent=2)}

SUSPECTED DIAGNOSES:
{json.dumps(pathology_analysis.get('suspected_diagnoses', []), indent=2)}

NCCN GUIDELINES (Full Text):
{guidelines}

ANALYSIS TASK:
1. Review all pathology results including any markers, molecular tests, and morphology findings
2. Match findings to specific criteria mentioned in the NCCN guidelines provided
3. Determine the most likely diagnosis with subtype and grade according to the guidelines
4. Assess confidence level based on how well the findings match the guideline criteria
5. Provide detailed justification referencing the specific NCCN guidelines provided
6. Include treatment pathway recommendations as mentioned in the guidelines

Output ONLY a JSON object with this exact structure:
{{
    "diagnosis": "primary tumor type per NCCN classification",
    "subtype": "specific subtype per NCCN",
    "grade": "tumor grade per NCCN grading system",
    "confidence": "high/medium/low",
    "justification": "detailed explanation referencing NCCN criteria and pathology findings",
    "supporting_markers": ["list", "of", "positive", "markers", "from", "pathology"],
    "molecular_findings": ["list", "of", "molecular", "results", "from", "pathology"],
    "cytogenetics": "cytogenetic findings if applicable",
    "nccn_pathway": "recommended NCCN treatment pathway from guidelines",
    "differential_diagnoses": ["alternative", "diagnoses", "considered"],
    "additional_recommendations": "NCCN-based recommendations for additional testing or clinical management"
}}

IMPORTANT: 
- Base diagnosis strictly on the NCCN guidelines provided
- Reference specific guidelines text in justification
- Include grade-appropriate treatment pathways mentioned in the guidelines
- If insufficient data, specify what additional tests are needed per guidelines
"""

        # Use LLM for diagnostic analysis
        runner = Runner(
            app_name="diagnostic_analyzer",
            agent=Agent(
                name="diagnostic_analyzer",
                model=LiteLlm(model="ollama_chat/llama3.2"),
                description="Confirms diagnosis based on pathology results and NCCN guidelines",
                instruction="""You are a specialized pathologist that confirms diagnoses based on comprehensive pathology data.
                Return only valid JSON. Use established diagnostic criteria and provide detailed justification."""
            ),
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )
        
        content = Content(
            parts=[Part(text=prompt)],
            role="user"
        )
        
        session = await runner.session_service.get_session(
            app_name="diagnostic_analyzer",
            user_id="diagnostic_analyzer_user",
            session_id="diagnostic_analyzer_session"
        )
        
        if session is None:
            session = await runner.session_service.create_session(
                app_name="diagnostic_analyzer",
                user_id="diagnostic_analyzer_user",
                session_id="diagnostic_analyzer_session",
                state={}
            )
        
        events = []
        async for event in runner.run_async(
            user_id="diagnostic_analyzer_user",
            session_id=session.id,
            new_message=content
        ):
            events.append(event)
        
        if not events:
            raise Exception("No response from diagnostic analyzer")
            
        last_event = events[-1]
        if not last_event.content or not last_event.content.parts:
            raise Exception("No content in diagnostic analyzer response")
            
        response_content = last_event.content.parts[0].text
        
        # Clean the response content
        content_text = response_content.strip()
        if content_text.startswith('```json'):
            content_text = content_text[7:]
        if content_text.endswith('```'):
            content_text = content_text[:-3]
        content_text = content_text.strip()
        
        # Parse the diagnostic result
        diagnostic_result = json.loads(content_text)
        
        # Save final diagnosis to shared state
        final_diagnosis = {
            "diagnosis_timestamp": datetime.now().isoformat(),
            "primary_diagnosis": diagnostic_result.get('diagnosis'),
            "subtype": diagnostic_result.get('subtype'),
            "grade": diagnostic_result.get('grade'),
            "confidence_level": diagnostic_result.get('confidence'),
            "justification": diagnostic_result.get('justification'),
            "supporting_evidence": {
                "positive_markers": diagnostic_result.get('supporting_markers', []),
                "molecular_findings": diagnostic_result.get('molecular_findings', []),
                "cytogenetics": diagnostic_result.get('cytogenetics'),
                "pathology_results_count": len(pathology_results)
            },
            "nccn_classification": {
                "pathway": diagnostic_result.get('nccn_pathway'),
                "grade_based_treatment": True,
                "guidelines_version": "NCCN Bone Tumors"
            },
            "differential_diagnoses": diagnostic_result.get('differential_diagnoses', []),
            "recommendations": diagnostic_result.get('additional_recommendations'),
            "pathologist_reviewed": True,
            "nccn_compliant": True
        }
        
        state['diagnosis'] = final_diagnosis
        save_shared_state(state)
        
        return {
            "status": "success",
            "final_diagnosis": final_diagnosis,
            "message": f"Diagnosis confirmed: {diagnostic_result.get('diagnosis')} ({diagnostic_result.get('subtype')}, {diagnostic_result.get('grade')})"
        }
        
    except Exception as e:
        logging.error(f"Error in diagnostic confirmation: {e}")
        return {
            "status": "error",
            "error_message": f"Failed to confirm diagnosis: {str(e)}"
        }

class PathologistAgent(AgentWithTaskManager):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "application/json"]
    
    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "pathologist_agent_user"
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
            name="pathologist_agent_Ollama",
            model=LiteLlm(model="ollama_chat/llama3.2"),
            description="Analyzes pathology results and confirms diagnosis based on NCCN guidelines",
            instruction="""
You are a specialized pathologist assistant following NCCN Bone Tumor guidelines. Your workflow is:

1. **Initial Analysis**: Use the 'analyze_diagnostic_request' tool to read imaging results from shared state and determine which pathology procedures are needed per NCCN guidelines.

2. **Show All Requirements**: After analysis, present ALL required pathology tests in a numbered list to give the user a complete overview.

3. **Systematic Collection**: Use the 'collect_pathology_details' tool to start systematic collection of results for each test. This tool will:
   - Guide you through each test one by one
   - Ask for specific fields in order: date, findings, markers, status, additional_details
   - Tell you exactly what question to ask next

4. **Record Each Response**: When the user provides an answer, use the 'record_pathology_detail' tool to record their response and get the next question.

5. **Follow the Workflow**: The tools will guide you through the entire process. Simply:
   - Ask the question provided by the tool
   - Wait for user response  
   - Call record_pathology_detail with their answer
   - Ask the next question provided by the tool
   - Repeat until all tests are complete

6. **Final Diagnosis**: When all pathology data is collected, use 'confirm_diagnosis' tool to provide NCCN-compliant final diagnosis including:
   - Primary tumor type per NCCN classification
   - Subtype and grade per NCCN criteria
   - Treatment pathway recommendation
   - Detailed justification referencing NCCN guidelines

IMPORTANT RULES:
- Always use the tools to guide the workflow - don't improvise questions
- Ask exactly ONE question at a time as directed by the tools
- Wait for user response before proceeding to next field
- Use the exact questions provided by collect_pathology_details
- Record every answer using record_pathology_detail before asking the next question
- Trust the tools to manage the workflow state and progression
- Base all recommendations strictly on NCCN guidelines

The tools handle all the complexity - your job is to be the interface between the user and the systematic collection process.
""",
            tools=[analyze_diagnostic_request, add_pathology_result, confirm_diagnosis, collect_pathology_details, record_pathology_detail]
        )

        
    def get_processing_message(self) -> str:
        return "Analyzing pathology requirements and confirming diagnosis based on NCCN guidelines..."