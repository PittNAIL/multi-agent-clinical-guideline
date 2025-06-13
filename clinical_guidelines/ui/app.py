import streamlit as st
import asyncio
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List
import json

from clinical_guidelines.schemas.base_schemas import PatientState, ClinicalRole
from clinical_guidelines.agents.orchestrator_agent import OrchestratorAgent
from clinical_guidelines.agents.oncologist_agent import OncologistAgent
from clinical_guidelines.schemas.agent_card import AgentCard, AgentType, AgentCapability, ModelInfo, AgentConstraint, AgentProtocol
from clinical_guidelines.example import SAMPLE_GUIDELINES

async def streamlit_user_input_handler(prompt: str, field_name: str, agent_role: str) -> str:
    """Handle user input requests through Streamlit UI"""
    # Create a unique key for the input field
    key = f"{agent_role}_{field_name}_{prompt}"
    
    # Show the request in the UI
    st.write(f"**{agent_role.title()} Agent** needs information:")
    value = st.text_input(prompt, key=key)
    
    # Add a submit button
    if st.button("Submit", key=f"submit_{key}"):
        return value
    return None

def create_agent_cards() -> Dict[str, AgentCard]:
    """Create agent cards for each specialist agent"""
    empty_constraints: List[AgentConstraint] = []
    empty_protocols: List[AgentProtocol] = []
    
    return {
        "intake": AgentCard(
            agent_id="intake-001",
            name="Intake Agent",
            role="intake",
            description="Handles initial patient intake and information gathering",
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=[AgentCapability.DIAGNOSIS],
            constraints=empty_constraints,
            protocols=empty_protocols,
            performance_metrics={"accuracy": 0.9},
            trust_score=0.9,
            specialization="intake",
            interaction_patterns=["guideline-based-decision"],
            model_info=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                version="1.0.0",
                capabilities=["text-generation", "medical-reasoning"],
                limitations=["No real-time data access"],
                performance_metrics={"accuracy": 0.9}
            ),
            required_inputs=["age", "current_symptoms", "medical_history"],
            provided_outputs=["initial_assessment", "risk_factors"],
            metadata={}
        ),
        "radiologist": AgentCard(
            agent_id="rad-001",
            name="Radiology Specialist Agent",
            role="radiologist",
            description="Processes and interprets imaging studies",
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=[AgentCapability.IMAGING],
            constraints=empty_constraints,
            protocols=empty_protocols,
            performance_metrics={"accuracy": 0.85},
            trust_score=0.85,
            specialization="radiology",
            interaction_patterns=["guideline-based-decision"],
            model_info=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                version="1.0.0",
                capabilities=["text-generation", "medical-reasoning"],
                limitations=["No real-time data access"],
                performance_metrics={"accuracy": 0.85}
            ),
            required_inputs=["imaging_required", "clinical_concern"],
            provided_outputs=["imaging_findings", "recommendations"],
            metadata={}
        ),
        "pathologist": AgentCard(
            agent_id="path-001",
            name="Pathology Specialist Agent",
            role="pathologist",
            description="Analyzes tissue samples and laboratory results",
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=[AgentCapability.PATHOLOGY],
            constraints=empty_constraints,
            protocols=empty_protocols,
            performance_metrics={"accuracy": 0.9},
            trust_score=0.9,
            specialization="pathology",
            interaction_patterns=["guideline-based-decision"],
            model_info=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                version="1.0.0",
                capabilities=["text-generation", "medical-reasoning"],
                limitations=["No real-time data access"],
                performance_metrics={"accuracy": 0.9}
            ),
            required_inputs=["biopsy_results", "molecular_tests"],
            provided_outputs=["diagnosis", "molecular_markers"],
            metadata={}
        ),
        "oncologist": AgentCard(
            agent_id="onc-001",
            name="Oncology Specialist Agent",
            role="oncologist",
            description="Processes oncology cases following NCCN guidelines",
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=[AgentCapability.TREATMENT],
            constraints=empty_constraints,
            protocols=empty_protocols,
            performance_metrics={"accuracy": 0.85},
            trust_score=0.8,
            specialization="oncology",
            interaction_patterns=["guideline-based-decision"],
            model_info=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                version="1.0.0",
                capabilities=["text-generation", "medical-reasoning"],
                limitations=["No real-time data access"],
                performance_metrics={"accuracy": 0.85}
            ),
            required_inputs=["diagnosis", "treatment_phase"],
            provided_outputs=["treatment_plan", "next_steps"],
            metadata={}
        ),
        "surgeon": AgentCard(
            agent_id="surg-001",
            name="Surgical Specialist Agent",
            role="surgeon",
            description="Evaluates surgical options and provides recommendations",
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=[AgentCapability.SURGERY],
            constraints=empty_constraints,
            protocols=empty_protocols,
            performance_metrics={"accuracy": 0.85},
            trust_score=0.85,
            specialization="surgery",
            interaction_patterns=["guideline-based-decision"],
            model_info=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                version="1.0.0",
                capabilities=["text-generation", "medical-reasoning"],
                limitations=["No real-time data access"],
                performance_metrics={"accuracy": 0.85}
            ),
            required_inputs=["diagnosis", "imaging_findings", "surgical_candidate"],
            provided_outputs=["surgical_plan", "post_op_care"],
            metadata={}
        ),
        "surveillance": AgentCard(
            agent_id="surv-001",
            name="Surveillance Specialist Agent",
            role="surveillance",
            description="Monitors patient progress and follow-up care",
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=[AgentCapability.SURVEILLANCE],
            constraints=empty_constraints,
            protocols=empty_protocols,
            performance_metrics={"accuracy": 0.85},
            trust_score=0.85,
            specialization="surveillance",
            interaction_patterns=["guideline-based-decision"],
            model_info=ModelInfo(
                name="mistral-7b-instruct",
                provider="HuggingFace",
                version="1.0.0",
                capabilities=["text-generation", "medical-reasoning"],
                limitations=["No real-time data access"],
                performance_metrics={"accuracy": 0.85}
            ),
            required_inputs=["diagnosis", "treatment_phase"],
            provided_outputs=["follow_up_schedule", "monitoring_plan"],
            metadata={}
        )
    }

def initialize_session_state():
    """Initialize session state variables."""
    try:
        if "initialization_error" not in st.session_state:
            st.session_state.initialization_error = None
            
        if "patient_state" not in st.session_state:
            st.session_state.patient_state = PatientState(
                patient_id="PENDING",
                treatment_phase="initial",
                current_agent=ClinicalRole.INTAKE,
                last_updated=datetime.now()
            )
        
        if "orchestrator" not in st.session_state:
            try:
                orchestrator = OrchestratorAgent(guidelines=SAMPLE_GUIDELINES)
                # Configure user input handler for orchestrator
                orchestrator.user_input_handler = streamlit_user_input_handler
                
                # Configure user input handler for all specialist agents
                for specialist in orchestrator.specialists.values():
                    specialist.user_input_handler = streamlit_user_input_handler
                    
                st.session_state.orchestrator = orchestrator
            except Exception as e:
                st.session_state.initialization_error = str(e)
                raise
        
        # Initialize agent cards
        if "agent_cards" not in st.session_state:
            st.session_state.agent_cards = create_agent_cards()
        
        if "patient_history" not in st.session_state:
            st.session_state.patient_history = {}
            
    except Exception as e:
        st.session_state.initialization_error = str(e)
        raise

def render_agent_cards():
    """Render agent cards in the UI"""
    st.sidebar.title("Available Agents")
    for agent_id, card in st.session_state.agent_cards.items():
        with st.sidebar.expander(f"📋 {card.name}"):
            st.write(f"**ID:** {card.agent_id}")
            st.write(f"**Type:** {card.agent_type}")
            st.write(f"**Model:** {card.model_info.name}")
            st.write("**Capabilities:**")
            for cap in card.model_info.capabilities:
                st.write(f"- {cap}")
            st.write("**Safety Protocols:**")
            for protocol in card.protocols:
                st.write(f"- {protocol}")

def display_workflow_status():
    """Display the current workflow status with guideline references"""
    state = st.session_state.patient_state
    st.subheader("Current Status")
    st.write(f"**Patient ID:** {state.patient_id}")
    st.write(f"**Current Agent:** {state.current_agent}")
    if state.requires_human_review:
        st.warning("⚠️ This case requires human review!")
    
    if state.agent_decisions:
        st.subheader("Decision History")
        for decision in state.agent_decisions:
            with st.expander(f"{decision.role} - {decision.timestamp}"):
                st.write(f"**Decision:** {decision.decision}")
                st.write(f"**Confidence Score:** {decision.confidence_score:.2f}")
                
                if decision.rationale:
                    st.write(f"**Rationale:** {decision.rationale}")
                
                st.write("**Next Steps:**")
                for step in decision.next_steps:
                    st.write(f"- {step}")
                
                if decision.guideline_references:
                    st.write("**NCCN Guidelines Referenced:**")
                    for ref in decision.guideline_references:
                        st.write(f"- {ref.guideline_id} ({ref.section})")
                        st.write(f"  Version: {ref.version}")
                        st.write(f"  Confidence: {ref.confidence_score:.2f}")
                
                if decision.alternative_options:
                    st.write("**Alternative Options Considered:**")
                    for option in decision.alternative_options:
                        st.write(f"- {option}")
                
                if decision.contraindications:
                    st.write("**Contraindications:**")
                    for contra in decision.contraindications:
                        st.write(f"- {contra}")

def display_metrics():
    """Display performance metrics"""
    st.sidebar.subheader("System Metrics")
    
    # Calculate average confidence scores
    if st.session_state.patient_history:
        confidence_scores = []
        for case in st.session_state.patient_history.values():
            for decision in case['state'].agent_decisions:
                confidence_scores.append(decision.confidence_score)
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            st.sidebar.metric("Average Confidence Score", f"{avg_confidence:.2f}")
    
    # Display guideline coverage
    guideline_refs = set()
    for case in st.session_state.patient_history.values():
        for decision in case['state'].agent_decisions:
            for ref in decision.guideline_references:
                guideline_refs.add(ref.guideline_id)
    
    st.sidebar.metric("NCCN Guidelines Referenced", len(guideline_refs))

async def process_patient_case():
    """Process a patient case through the orchestrator"""
    if st.session_state.initialization_error:
        st.error(f"System not properly initialized: {st.session_state.initialization_error}")
        return
        
    try:
        # Create a container for agent requests
        request_container = st.empty()
        
        with request_container:
            response = await st.session_state.orchestrator.run(st.session_state.patient_state)
            st.session_state.patient_state = response.updated_state
            
        # Clear the request container after processing
        request_container.empty()
        
    except Exception as e:
        error_msg = f"Error processing case: {str(e)}"
        st.error(error_msg)
        st.session_state.patient_state.error = error_msg
        # Don't raise the error, just log it
        print(f"Error in process_patient_case: {str(e)}")

def main():
    st.title("Clinical Guideline Support System")
    
    try:
        initialize_session_state()
    except Exception as e:
        st.error("Failed to initialize the system. Please try refreshing the page.")
        st.error(f"Error details: {str(e)}")
        return
        
    if st.session_state.initialization_error:
        st.error("System initialization failed. Please try refreshing the page.")
        st.error(f"Error details: {st.session_state.initialization_error}")
        return
    
    # Render agent cards in sidebar
    render_agent_cards()
    
    # Main interface
    tab1, tab2 = st.tabs(["New Patient", "Patient History"])
    
    with tab1:
        st.header("Patient Information")
        
        # Patient input form
        with st.form("patient_form"):
            patient_id = st.text_input("Patient ID")
            age = st.number_input("Age", min_value=0, max_value=120)
            symptoms = st.text_area("Current Symptoms (one per line)")
            diagnosis = st.text_input("Initial Diagnosis")
            
            col1, col2 = st.columns(2)
            with col1:
                mri_result = st.text_area("MRI Results")
            with col2:
                ct_result = st.text_area("CT Results")
            
            submitted = st.form_submit_button("Process Patient Case")
            
            if submitted and patient_id and age:
                # Create patient state
                st.session_state.patient_state = PatientState(
                    patient_id=patient_id,
                    age=age,
                    current_symptoms=symptoms.split("\n") if symptoms else [],
                    diagnosis=diagnosis if diagnosis else None,
                    treatment_phase="initial",
                    current_agent=ClinicalRole.INTAKE,
                    last_updated=datetime.now(),
                    imaging_results={
                        "mri": mri_result if mri_result else "",
                        "ct_chest": ct_result if ct_result else ""
                    }
                )
                
                # Process patient case
                with st.spinner("Processing patient case..."):
                    try:
                        asyncio.run(process_patient_case())
                        
                        # Only store in history if no errors
                        if not hasattr(st.session_state.patient_state, 'error'):
                            st.session_state.patient_history[patient_id] = {
                                "state": st.session_state.patient_state,
                                "timestamp": datetime.now()
                            }
                            st.success("Case processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Failed to process case: {str(e)}")
                
                # Display results
                display_workflow_status()
    
    with tab2:
        st.header("Patient History")
        if st.session_state.patient_history:
            for patient_id, data in st.session_state.patient_history.items():
                with st.expander(f"Patient {patient_id} - {data['timestamp']}"):
                    state = data['state']
                    st.write(f"**Age:** {state.age}")
                    st.write(f"**Diagnosis:** {state.diagnosis}")
                    st.write(f"**Treatment Phase:** {state.treatment_phase}")
                    st.write(f"**Last Updated:** {state.last_updated}")
                    if state.agent_decisions:
                        st.write("**Decisions:**")
                        for decision in state.agent_decisions:
                            st.write(f"- {decision.role}: {decision.decision}")
        else:
            st.info("No patient history available")

if __name__ == "__main__":
    main() 