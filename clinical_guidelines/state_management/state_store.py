from typing import Dict, List, Optional
from datetime import datetime
import json
import logging
from pathlib import Path
from ..schemas.base_schemas import PatientState, AgentDecision

class StateStore:
    def __init__(self, storage_dir: str = "data/states"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.storage_dir / "state_changes.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger("state_store")
    
    async def save_state(self, state: PatientState) -> None:
        """Save patient state to persistent storage"""
        state_file = self.storage_dir / f"{state.patient_id}.json"
        
        # Log the state change
        self.logger.info(
            f"Saving state for patient {state.patient_id} - "
            f"Phase: {state.treatment_phase}, "
            f"Agent: {state.current_agent}"
        )
        
        # Save state to file
        with open(state_file, "w") as f:
            json.dump(state.dict(), f, default=str)
    
    async def load_state(self, patient_id: str) -> Optional[PatientState]:
        """Load patient state from storage"""
        state_file = self.storage_dir / f"{patient_id}.json"
        
        if not state_file.exists():
            return None
        
        with open(state_file, "r") as f:
            state_dict = json.load(f)
            return PatientState.parse_obj(state_dict)
    
    async def list_active_cases(self) -> List[str]:
        """List all active patient cases"""
        return [f.stem for f in self.storage_dir.glob("*.json")]
    
    async def archive_state(self, patient_id: str) -> None:
        """Archive a completed case"""
        state_file = self.storage_dir / f"{patient_id}.json"
        archive_dir = self.storage_dir / "archived"
        archive_dir.mkdir(exist_ok=True)
        
        if state_file.exists():
            archive_file = archive_dir / f"{patient_id}_{datetime.now().strftime('%Y%m%d')}.json"
            state_file.rename(archive_file)
            self.logger.info(f"Archived state for patient {patient_id}")
    
    async def get_state_history(self, patient_id: str) -> List[Dict]:
        """Get history of state changes for a patient"""
        history = []
        with open(self.storage_dir / "state_changes.log", "r") as f:
            for line in f:
                if patient_id in line:
                    history.append(line.strip())
        return history 