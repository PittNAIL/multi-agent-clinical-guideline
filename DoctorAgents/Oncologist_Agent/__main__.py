import logging
import os
import sys

# Use a relative path to find the common module
common_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(common_path)

import click

from Agent import OncologistAgent
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
@click.option('--port', default=10009)  # Using a different port
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='oncologist_assistant',
            name='OncologistAgent',
            description='Assists oncologists with diagnosis analysis and NCCN-based treatment plan confirmation.',
            tags=['Oncologist', 'Diagnostics', 'Medical', 'NCCN', 'Treatment Plan'],
            examples=[
            'What are the recommended NCCN treatment options for breast cancer?',
            'Has the patient received all required diagnostic tests for oncology evaluation?',
            'What is the standard chemotherapy protocol for this diagnosis according to NCCN guidelines?'
            ],
        )
        agent_card = AgentCard(
            name='OncologistAgent',
            description='Specialized assistant for oncologists to create NCCN-based treatment plans.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=OncologistAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=OncologistAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=OncologistAgent()),
            host=host,
            port=port,
        )
        server.start()
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main() 