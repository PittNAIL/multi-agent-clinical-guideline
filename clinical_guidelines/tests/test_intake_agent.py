import asyncio
from ..agents.intake_agent import IntakeAgent

# Sample guidelines
SAMPLE_GUIDELINES = {
    "chest_pain": """
    For patients presenting with chest pain:
    1. Assess vital signs immediately
    2. Consider cardiac imaging if:
       - Age > 40
       - Family history
       - Multiple risk factors
    3. Urgent care needed if:
       - Severe pain
       - Shortness of breath
       - Radiation to arm/jaw
    """,
    "general_intake": """
    General intake guidelines:
    1. Document chief complaint
    2. Record vital signs
    3. Note duration of symptoms
    4. Document medical history
    5. List current medications
    """
}

async def main():
    # Create the agent
    agent = IntakeAgent(guidelines=SAMPLE_GUIDELINES)
    
    # Test case 1: Chest pain in older patient
    result = await agent.process_intake(
        symptoms="Chest pain radiating to left arm, shortness of breath",
        age=55,
        sex="M"
    )
    print("\nTest Case 1 - Chest Pain:")
    print(result["output"])
    
    # Test case 2: Minor symptoms in young patient
    result = await agent.process_intake(
        symptoms="Mild cough for 2 days, no fever",
        age=25,
        sex="F"
    )
    print("\nTest Case 2 - Minor Symptoms:")
    print(result["output"])

if __name__ == "__main__":
    asyncio.run(main()) 