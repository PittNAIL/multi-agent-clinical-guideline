# intake_agent/client.py
import asyncio
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Common.server import A2AServer
from Agents.intake_agent.Agent import IntakeAgent
from Agents.intake_agent.card import get_agent_card
from Common.types import TaskRequest
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intake_agent")

class IntakeTaskManager:
    def __init__(self):
        self.agent = IntakeAgent()

    async def handle_task(self, task_request: TaskRequest):
        raw_text = task_request.message.get("raw_text")
        # The ADK agent's run method expects the raw text directly in the message
        return await self.agent.structure_plan(raw_text)

@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=5002)
def run_agent(host: str, port: int):
    try:
        agent_card = get_agent_card(host, port)
        task_manager = IntakeTaskManager()

        server = A2AServer(
            agent_card=agent_card,
            task_manager=task_manager,
            host=host,
            port=port,
        )
        logger.info(f"Starting IntakeAgent at http://{host}:{port}/")
        server.start()

    except Exception as e:
        logger.error(f"Failed to start IntakeAgent: {e}")
        exit(1)

if __name__ == '__main__':
    run_agent()