# Agents/Plan_review_Agent/client.py

import asyncio
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Common.server import A2AServer
from Agents.Plan_review_Agent.Agent import PlanReviewAgent
from Agents.Plan_review_Agent.card import get_agent_card
from Common.types import TaskRequest
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plan_review_agent")

class PlanReviewTaskManager:
    def __init__(self):
        self.agent = PlanReviewAgent()

    async def handle_task(self, task_request: TaskRequest):
        diagnosis = task_request.message.get("diagnosis")
        steps = task_request.message.get("steps")
        return await self.agent.review_treatment_plan(diagnosis, steps)

@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=5001)
def run_agent(host: str, port: int):
    try:
        agent_card = get_agent_card(host, port)
        task_manager = PlanReviewTaskManager()

        server = A2AServer(
            agent_card=agent_card,
            task_manager=task_manager,
            host=host,
            port=port,
        )
        logger.info(f"Starting PlanReviewAgent at http://{host}:{port}/")
        server.start()

    except Exception as e:
        logger.error(f"Failed to start PlanReviewAgent: {e}")
        exit(1)

if __name__ == '__main__':
    run_agent() 