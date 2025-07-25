import logging
import os
import sys

# Use a relative path to find the common module
common_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(common_path)

import click

from Agent import SurgicalAgent
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
@click.option('--port', default=10010)  # Use a unique port for SurgicalAgent
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='surgical_assistant',
            name='SurgicalAgent',
            description='Plans local control (surgery or radiation) per NCCN guideline.',
            tags=['Surgery', 'Radiation', 'Bone Cancer', 'NCCN'],
            examples=[
                'Is the tumor resectable after chemotherapy?',
                'Should we proceed to resection or use radiation?',
                'What does the guideline recommend for positive surgical margins?'
            ],
        )
        agent_card = AgentCard(
            name='SurgicalAgent',
            description='Specialized assistant for surgical planning and local control per NCCN guidelines.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=SurgicalAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SurgicalAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=SurgicalAgent()),
            host=host,
            port=port,
        )
        server.start()
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)

if __name__ == '__main__':
    main()