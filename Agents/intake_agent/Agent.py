# intake_agent/agent.py
import json
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig
from google.adk.models.lite_llm import LiteLlm
from schema import IntakeOutput

class IntakeAgent:
    """
    An LLM-based ADK agent that structures raw clinical text into a JSON format.
    """
    def __init__(self):
        self.agent = self._create_agent()

    def _create_agent(self) -> LlmAgent:
        model_config = LiteLlm(
            model="ollama_chat/llama3.2",
            generate_content_config=GenerateContentConfig(temperature=0.0)
        )
        
        return LlmAgent(
            name="IntakeAgent",
            model=model_config,
            instruction=self._get_instruction(),
            output_schema=IntakeOutput,
            output_key="intake_output"
        )

    def _get_instruction(self) -> str:
        """Builds the detailed prompt for the LLM."""
        return """
You are a clinical data structuring assistant. Your sole responsibility is to
extract structured data from the provided clinical text and return it in a
valid JSON format. Do not add any commentary or introductory text.

The user will provide the raw text. Extract the following fields:
- Patient age, sex, and any relevant clinical notes.
- The specific cancer diagnosis (e.g., osteosarcoma, chondrosarcoma).
- The cancer subtype, if mentioned.
- The primary tumor location (e.g., pelvis, skull base).
- Whether the cancer is metastatic.
- A list of distinct treatment steps (e.g., "chemotherapy", "surgery").

Format your entire response as a single, valid `IntakeOutput` JSON object.
"""

    async def structure_plan(self, raw_text: str) -> dict:
        """
        Uses the ADK agent to structure the raw text.
        """
        response = await self.agent.run(message=raw_text)
        
        intake = response.get("intake_output", {})
        if isinstance(intake, IntakeOutput):
            return intake.dict()
            
        print(f"IntakeAgent: Received unexpected response format: {intake}")
        return intake