# task_manager.py

from typing import Any
from types import TaskRequest, TaskResponse

class AgentTaskManager:
    def __init__(self, agent):
        self.agent = agent

    async def handle_task(self, request: TaskRequest) -> TaskResponse:
        """
        Dispatches the task to the agent's run method.
        """
        return await self.agent.run(request)
