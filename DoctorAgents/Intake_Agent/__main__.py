import logging
import os
import sys

# Use a relative path to find the common module
common_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(common_path)

import click

from Agent import IntakeAgent
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
@click.option('--port', default=10006)
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id='structure_clinical_data',
            name='IntakeAgent',
            description='Structures unstructured clinical text into standardized format',
            tags=['Clinical', 'Intake', 'Data Structuring'],
            examples=[
                'Patient is a 45yo male with right knee pain for 2 weeks, worse with activity. PE shows swelling.',
                '72F presenting with fatigue, weight loss, and night sweats. Physical exam unremarkable.'
            ],
        )
        agent_card = AgentCard(
            name='IntakeAgent',
            description='Structures raw clinical text input into standardized format with age, sex, symptoms, physical findings, and other clinical details.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=IntakeAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=IntakeAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
            custom={"priority": 1, "always_first": True},
        )
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=IntakeAgent()),
            host=host,
            port=port,
        )
        server.start()
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main() 