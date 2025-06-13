#To run the agent

import logging
import os
import sys

sys.path.append("/Users/prasannanagarajan/Desktop/Main_Desktop/PITTNAIL_research/A2A/samples/python/common")

import click

from Agent import TellweatherAgent
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from task_manager import AgentTaskManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10005)
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='tell weather',
            name='tell weather Tool',
            description='Helps with the knowledge of the weather',
            tags=['Weather'],
            examples=[
                'can you tell me the weather of tokyo?'
            ],
        )
        agent_card = AgentCard(
            name='Weather Agent',
            description='Helps with the knowledge of the weather.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=TellweatherAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=TellweatherAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=TellweatherAgent()),
            host=host,
            port=port,
        )
        server.start()
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
