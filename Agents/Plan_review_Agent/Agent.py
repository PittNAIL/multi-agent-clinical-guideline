# Agents/Plan_review_Agent/Agent.py
import json
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.genai.types import GenerateContentConfig
from google.adk.models.lite_llm import LiteLlm
from .tools import fetch_guideline_for_diagnosis

class PlanReviewAgent:
    """
    An LLM-based ADK agent that reviews treatment plans for NCCN guideline concordance.
    """
    def __init__(self):
        self.agent = self._create_agent()

    def _create_agent(self) -> LlmAgent:
        guideline_tool = FunctionTool(fetch_guideline_for_diagnosis)
        
        model_config = LiteLlm(
            model="ollama_chat/llama3.2",
            generate_content_config=GenerateContentConfig(temperature=0.1)
        )
        
        return LlmAgent(
            name="PlanReviewAgent",
            model=model_config,
            instruction=self._get_instruction(),
            tools=[guideline_tool]
        )

    def _get_instruction(self) -> str:
        """Builds the detailed prompt for the LLM to perform the review."""
        return """
You are an expert clinical reviewer specializing in NCCN guideline compliance for bone cancer.
Your task is to compare a patient's proposed treatment plan against the official NCCN guideline and determine if it is concordant.

The user will provide a diagnosis and a list of treatment steps.
1. Your first step is to call the `fetch_guideline_for_diagnosis` tool using the provided diagnosis. This will return the raw text of the guideline from the document.
2. Once you have the guideline text, carefully read and understand the recommended treatment pathway.
3. Compare the user's treatment steps against the guideline you retrieved.
4. Formulate a detailed explanation for your conclusion, identifying any missing or extra steps.

Finally, format your entire response as a single, valid JSON object with the following keys: "status", "missing_steps", "extra_steps", "explanation". Do not include any text outside of the JSON block.
"""

    async def review_treatment_plan(self, diagnosis: str, treatment_steps: list) -> dict:
        """
        Uses the ADK agent to review a treatment plan against NCCN guidelines.
        """
        request_message = (
            f"Please review the following treatment plan.\n"
            f"Diagnosis: {diagnosis}\n"
            f"Treatment Steps: {json.dumps(treatment_steps)}"
        )
        
        response = await self.agent.run(message=request_message)
        
        try:
            model_output_str = response['message']
            # The model might wrap the JSON in markdown, so we extract it.
            json_str = model_output_str[model_output_str.find('{'):model_output_str.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            print(f"PlanReviewAgent: Error parsing LLM response: {e}")
            return {"status": "error", "explanation": "Failed to parse the review from the agent."}
