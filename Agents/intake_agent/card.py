# intake_agent/card.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Common.types import AgentCard, AgentCapabilities, AgentSkill
from Agents.intake_agent.Agent import IntakeAgent


def get_agent_card(host: str = "localhost", port: int = 10006) -> AgentCard:
    # This check is to ensure that IntakeAgent has the required class attribute.
    # We can make this more robust later if needed.
    supported_types = getattr(IntakeAgent, 'SUPPORTED_CONTENT_TYPES', ['text/plain'])
    
    return AgentCard(
        name="IntakeAgent",
        description="Structures unstructured clinical input into structured fields like diagnosis, tumor type, and treatment steps.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=supported_types,
        defaultOutputModes=["application/json"],
        skills=[
            AgentSkill(
                id="structure_clinical_data",
                name="Clinical Intake Structuring",
                description="Extracts structured fields from a clinical paragraph.",
                tags=["Intake", "NCCN", "Bone Cancer"],
                examples=[
                    "45M with distal femur lesion, osteosarcoma, undergoing chemo",
                    "72F, pelvic mass suspected chondrosarcoma, treated with surgery"
                ],
            )
        ],
        custom={"priority": 1, "always_first": True},
    )
