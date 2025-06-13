import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from task_manager import AgentWithTaskManager

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level = logging.ERROR)

def get_weather(city: str, *args, **kwargs) -> dict:
        """Retrieves the current weather report for a specified city.

        Args:
            city (str): The name of the city (e.g., "New York", "London", "Tokyo").

        Returns:
            dict: A dictionary containing the weather information.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'report' key with weather details.
                If 'error', includes an 'error_message' key.
        """
        print(f"- - - Tool: get_weather called for city: {city} - - -")
        
        city_raw = city.lower().replace(" ","")
        
        # Mock weather data
        mock_weather_db = {
            "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
            "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
            "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
        }
        
        if city_raw in mock_weather_db:
            return mock_weather_db[city_raw]
        
        else:
            return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

class TellweatherAgent(AgentWithTaskManager):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    
    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "weather_agent_user"
        
        self._runner = Runner(
            app_name = self._agent.name,
            agent = self._agent,
            session_service = InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )
        
        
    def _build_agent(self) -> Agent:
        return Agent(
            name = "weather_agent_Ollama",
            model = LiteLlm(model="ollama_chat/llama3.2") ,
            description = "Provides weather information for specific cities",
            instruction="You are a helpful weather assistant. "
                        "When the user asks for the weather in a specific city, "
                        "use the 'get_weather' tool to find the information. "
                        "Use the 'get_weather' tool for city weather requests. "
                        "Clearly present successful reports or polite error messages based on the tool's output status.",
            tools=[get_weather]
        )
        
    def get_processing_message(self) -> str:
        return "Looking up the weather for the requested city..."

