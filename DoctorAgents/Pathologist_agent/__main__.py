import logging
import os
import sys

# Use a relative path to find the common module
common_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(common_path)

import click

from Agent import PathologistAgent
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
@click.option('--port', default=10008)  # Using a different port than RadiologistAgent
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='pathologist_assistant',
            name='PathologistAgent',
            description='Helps with pathology analysis and NCCN-based diagnostic confirmation',
            tags=['Pathology', 'Diagnostics', 'Medical', 'NCCN'],
            examples=[
                'What pathology tests are required for suspected bone tumors?',
                'Has the biopsy been performed for this patient?',
                'What are the NCCN requirements for diagnostic confirmation?'
            ],
        )
        agent_card = AgentCard(
            name='PathologistAgent',
            description='Specialized assistant for pathology analysis and NCCN-based diagnostic confirmation.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=PathologistAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=PathologistAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=PathologistAgent()),
            host=host,
            port=port,
        )
        server.start()
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main() 