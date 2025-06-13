from pathlib import Path
from typing import Dict, List

class GuidelineLoader:
    def __init__(self, guideline_dir: str = "guideline_store"):
        self.guideline_dir = Path(__file__).parent.parent / guideline_dir
    
    def load_guidelines_for_agent(self, agent_name: str) -> Dict[str, str]:
        """
        Load guidelines for a specific agent from their guideline file.
        
        Args:
            agent_name: Name of the agent (e.g., "IntakeAgent", "RadiologistAgent")
            
        Returns:
            Dictionary mapping guideline IDs to guideline content
        """
        guideline_file = self.guideline_dir / f"{agent_name}_Guidelines.txt"
        
        if not guideline_file.exists():
            raise FileNotFoundError(f"No guideline file found for {agent_name}")
        
        guidelines: Dict[str, str] = {}
        current_guideline = []
        current_id = None
        
        with open(guideline_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("GUIDELINE_ID:"):
                    # Save previous guideline if exists
                    if current_id and current_guideline:
                        guidelines[current_id] = "\n".join(current_guideline)
                        current_guideline = []
                    current_id = line.split(":", 1)[1].strip()
                elif line and current_id:
                    current_guideline.append(line)
            
            # Save last guideline
            if current_id and current_guideline:
                guidelines[current_id] = "\n".join(current_guideline)
        
        return guidelines
    
    def load_all_guidelines(self) -> Dict[str, Dict[str, str]]:
        """
        Load guidelines for all agents.
        
        Returns:
            Dictionary mapping agent names to their guidelines
        """
        all_guidelines = {}
        
        for guideline_file in self.guideline_dir.glob("*_Guidelines.txt"):
            agent_name = guideline_file.stem.replace("_Guidelines", "")
            all_guidelines[agent_name] = self.load_guidelines_for_agent(agent_name)
        
        return all_guidelines 