from typing import Dict
from .agent_card import AgentCard, AgentType, Capability, ModelInfo

def get_agent_cards() -> Dict[str, AgentCard]:
    """Create and return agent cards for all specialist agents"""
    return {
        "intake": AgentCard(
            agent_id="intake-001",
            name="Intake Agent",
            description="Initial patient assessment and workflow trigger",
            version="1.0.0",
            agent_type=AgentType.CLIENT,
            capabilities=[
                Capability.STATE_MANAGEMENT,
                Capability.DECISION_MAKING
            ],
            model=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                type="large_language_model",
                parameters={"temperature": 0.1}
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "sex": {"type": "string"},
                    "symptoms": {"type": "array"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "next_steps": {"type": "array"},
                    "imaging_required": {"type": "boolean"}
                }
            },
            safety_protocols=["Input validation", "Age range check"]
        ),
        
        "orchestrator": AgentCard(
            agent_id="orch-001",
            name="Orchestrator Agent",
            description="Manages workflow and routes cases to specialists",
            version="1.0.0",
            agent_type=AgentType.SERVER,
            capabilities=[
                Capability.WORKFLOW_CONTROL,
                Capability.STATE_MANAGEMENT
            ],
            model=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                type="large_language_model",
                parameters={"temperature": 0.1}
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "patient_state": {"type": "object"},
                    "current_phase": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "next_agent": {"type": "string"},
                    "routing_reason": {"type": "string"}
                }
            },
            safety_protocols=["Workflow validation", "Loop detection"]
        ),
        
        "radiologist": AgentCard(
            agent_id="rad-001",
            name="Radiologist Agent",
            description="Imaging analysis and recommendations",
            version="1.0.0",
            agent_type=AgentType.HOST,
            capabilities=[
                Capability.GUIDELINE_PROCESSING,
                Capability.DECISION_MAKING
            ],
            model=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                type="large_language_model",
                parameters={"temperature": 0.1}
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "clinical_concern": {"type": "string"},
                    "patient_history": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "imaging_recommendation": {"type": "string"},
                    "metastasis_check": {"type": "boolean"}
                }
            },
            safety_protocols=["Radiation safety check", "Protocol adherence"]
        ),
        
        "pathologist": AgentCard(
            agent_id="path-001",
            name="Pathologist Agent",
            description="Tissue analysis and diagnosis",
            version="1.0.0",
            agent_type=AgentType.HOST,
            capabilities=[
                Capability.GUIDELINE_PROCESSING,
                Capability.DECISION_MAKING
            ],
            model=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                type="large_language_model",
                parameters={"temperature": 0.1}
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "biopsy_results": {"type": "object"},
                    "molecular_tests": {"type": "array"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string"},
                    "grade": {"type": "string"},
                    "markers": {"type": "array"}
                }
            },
            safety_protocols=["Diagnosis verification", "Marker validation"]
        ),
        
        "oncologist": AgentCard(
            agent_id="onc-001",
            name="Oncologist Agent",
            description="Treatment planning and management",
            version="1.0.0",
            agent_type=AgentType.HOST,
            capabilities=[
                Capability.GUIDELINE_PROCESSING,
                Capability.DECISION_MAKING
            ],
            model=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                type="large_language_model",
                parameters={"temperature": 0.1}
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string"},
                    "ldh_level": {"type": "number"},
                    "staging": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "chemo_plan": {"type": "object"},
                    "timeline": {"type": "array"}
                }
            },
            safety_protocols=["Dosage check", "Drug interaction check"]
        ),
        
        "surgeon": AgentCard(
            agent_id="surg-001",
            name="Surgical Agent",
            description="Surgical planning and recommendations",
            version="1.0.0",
            agent_type=AgentType.HOST,
            capabilities=[
                Capability.GUIDELINE_PROCESSING,
                Capability.DECISION_MAKING
            ],
            model=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                type="large_language_model",
                parameters={"temperature": 0.1}
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "post_chemo_imaging": {"type": "object"},
                    "patient_status": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "surgery_recommendation": {"type": "string"},
                    "radiation_needed": {"type": "boolean"}
                }
            },
            safety_protocols=["Pre-op checklist", "Risk assessment"]
        ),
        
        "surveillance": AgentCard(
            agent_id="surv-001",
            name="Surveillance Agent",
            description="Long-term monitoring and follow-up",
            version="1.0.0",
            agent_type=AgentType.HOST,
            capabilities=[
                Capability.GUIDELINE_PROCESSING,
                Capability.STATE_MANAGEMENT
            ],
            model=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                type="large_language_model",
                parameters={"temperature": 0.1}
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "post_treatment_status": {"type": "object"},
                    "tumor_type": {"type": "string"},
                    "time_since_treatment": {"type": "string"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "follow_up_schedule": {"type": "array"},
                    "imaging_protocol": {"type": "object"}
                }
            },
            safety_protocols=["Schedule validation", "Risk stratification"]
        )
    } 