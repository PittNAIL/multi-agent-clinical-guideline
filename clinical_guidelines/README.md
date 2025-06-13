# Clinical Guidelines Support System

A system for managing clinical workflows and guidelines using AI agents.

## Overview

This system provides AI-powered support for clinical decision making by:
- Processing patient cases through specialized medical agents
- Coordinating workflow between different specialists
- Applying clinical guidelines to patient cases
- Managing state transitions and decision tracking
- Providing a user interface for case management

## Architecture

The system consists of several key components:

1. **Specialist Agents**: Individual AI agents (Oncologist, Radiologist, etc.) that process patient cases
2. **Orchestrator Agent**: Coordinates workflow between specialists and manages state transitions
3. **UI Layer**: Streamlit interface for case management and visualization
4. **Tools**: Reusable components for guideline processing and analysis
5. **Schemas**: Data models for patient state and agent interactions

## Quick Start

```python
from clinical_guidelines.schemas.base_schemas import PatientState
from clinical_guidelines.agents.orchestrator_agent import OrchestratorAgent

# Create a patient case
patient = PatientState(
    patient_id="12345",
    age=45,
    current_symptoms=["persistent cough"]
)

# Initialize orchestrator
orchestrator = OrchestratorAgent(guidelines=[])

# Process the case
response = await orchestrator.run(patient)
updated_state = response.updated_state
```

## Running the UI

```bash
python run.py
```

This will start the Streamlit interface on http://localhost:8501

## Components

### Agents
- `BaseAgent`: Abstract base class for all agents
- `OrchestratorAgent`: Manages workflow and coordinates between specialists
- `OncologistAgent`: Handles oncology-specific decisions
- `RadiologistAgent`: Processes imaging and radiology cases
- (Other specialist agents)

### Tools
- `GuidelineProcessor`: Processes and analyzes clinical guidelines
- `StateManager`: Manages patient state transitions
- `DecisionTracker`: Tracks and logs agent decisions

### Schemas
- `PatientState`: Core data model for patient information
- `AgentResponse`: Standard format for agent decisions
- `ClinicalRole`: Enumeration of specialist roles

## Development

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`

### Adding New Agents
1. Inherit from `BaseAgent`
2. Implement required methods
3. Register with `OrchestratorAgent`

### Guidelines
- Store in `guideline_store/`
- Format as structured text
- Include metadata for processing

## Testing

```bash
pytest tests/
```

## License

MIT License 