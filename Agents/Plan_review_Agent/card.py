# Agents/Plan_review_Agent/card.py

from Common.types import AgentCard, AgentCapabilities, AgentSkill
from Agents.Plan_review_Agent.Agent import PlanReviewAgent


def get_agent_card(host: str = "localhost", port: int = 5001) -> AgentCard:
    return AgentCard(
        name="PlanReviewAgent",
        description="Reviews oncologist treatment plans for NCCN guideline concordance.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["application/json"],
        defaultOutputModes=["application/json"],
        skills=[
            AgentSkill(
                id="review_treatment_plan",
                name="Review Treatment Plan",
                description="Checks a cancer treatment plan against NCCN guidelines for concordance.",
                tags=["Review", "NCCN", "Bone Cancer"],
                examples=[
                    '{"diagnosis": "osteosarcoma", "steps": ["biopsy", "chemo"]}'
                ]
            )
        ],
        custom={"priority": 2},
    ) 