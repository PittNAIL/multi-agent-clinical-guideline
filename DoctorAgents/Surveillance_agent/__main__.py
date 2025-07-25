import logging
import os
import sys

# Use a relative path to find the common module
common_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(common_path)

import click

from Agent import SurveillanceAgent  # <-- import your SurveillanceAgent
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
@click.option('--port', default=10011)  # Unique port for SurveillanceAgent
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='surveillance_assistant',
            name='SurveillanceAgent',
            description='Manages NCCN-compliant post-treatment surveillance for bone sarcoma.',
            tags=['Surveillance', 'Follow-up', 'Bone Cancer', 'NCCN'],
            examples=[
                'What follow-up imaging is required after surgery?',
                'Is the patient overdue for MRI or CT scan?',
                'What is the NCCN surveillance schedule for osteosarcoma?'
            ],
        )
        agent_card = AgentCard(
            name='SurveillanceAgent',
            description='Specialized assistant for post-treatment surveillance per NCCN guidelines.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=SurveillanceAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SurveillanceAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=SurveillanceAgent()),
            host=host,
            port=port,
        )
        server.start()
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)

if __name__ == '__main__':
    main()
