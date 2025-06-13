import logging
import os
import sys

sys.path.append("/Users/prasannanagarajan/Desktop/Main_Desktop/PITTNAIL_research/A2A/samples/python/common")

import click

from Agent import RadiologistAgent
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
@click.option('--port', default=10007)  # Using a different port than WeatherAgent
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='radiologist_assistant',
            name='Radiologist Assistant Tool',
            description='Helps with imaging analysis and recommendations based on clinical concerns',
            tags=['Radiology', 'Imaging', 'Medical'],
            examples=[
                'What imaging studies should be ordered for a patient with swelling?',
                'Has the MRI been performed for this patient?',
                'Is PET-CT needed for suspected metastasis?'
            ],
        )
        agent_card = AgentCard(
            name='Radiologist Agent',
            description='Specialized assistant for imaging analysis and recommendations based on clinical guidelines.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=RadiologistAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=RadiologistAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=RadiologistAgent()),
            host=host,
            port=port,
        )
        server.start()
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main() 