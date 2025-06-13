import asyncio
from datetime import datetime
from typing import List

from clinical_guidelines.agents.oncologist_agent import OncologistAgent
from clinical_guidelines.schemas.base_schemas import PatientState, ClinicalRole
from clinical_guidelines.agents.orchestrator_agent import OrchestratorAgent

# Example NCCN guidelines (simplified for demonstration)
SAMPLE_GUIDELINES = [
    """For high-grade osteosarcoma:
    1. Initial workup should include:
       - Complete imaging (MRI of primary site, CT chest)
       - Biopsy for confirmation
    2. Standard treatment involves:
       - Neoadjuvant chemotherapy
       - Surgical resection
       - Adjuvant chemotherapy based on necrosis
    3. Follow-up:
       - Every 3 months for 2 years
       - Then every 6 months for years 3-5""",
    
    """Treatment response evaluation:
    1. After neoadjuvant chemotherapy:
       - Repeat imaging of primary site
       - Assessment of tumor necrosis
    2. Poor response indicators:
       - <90% necrosis
       - Progressive disease
    3. Consider clinical trial for poor response"""
]

async def main():
    # Create a new patient state
    patient = PatientState(
        patient_id="12345",
        age=45,
        current_symptoms=["persistent cough", "chest pain"],
        medical_history=["hypertension"],
        current_medications=["lisinopril"]
    )

    # Initialize orchestrator
    orchestrator = OrchestratorAgent(guidelines=[])

    try:
        # Process the patient case
        response = await orchestrator.run(patient)
        updated_state = response.updated_state

        # Print workflow summary
        print("\nWorkflow Summary:")
        print(f"Patient ID: {updated_state.patient_id}")
        print(f"Current Agent: {updated_state.current_agent}")
        print(f"Requires Human Review: {updated_state.requires_human_review}")
        print("\nDecisions:")
        for decision in updated_state.agent_decisions:
            print(f"\nRole: {decision.role}")
            print(f"Decision: {decision.decision}")
            print(f"Next Steps: {decision.next_steps}")
            print(f"Timestamp: {decision.timestamp}")

    except Exception as e:
        print(f"Error processing patient case: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 