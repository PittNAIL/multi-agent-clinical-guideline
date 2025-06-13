from typing import Dict, List, Optional
from ..schemas.agent_card import AgentCard, AgentType
from ..schemas.agent_registry import get_agent_cards

class CardResolver:
    def __init__(self):
        self.agent_cards: Dict[str, AgentCard] = get_agent_cards()
        self._validate_cards()
    
    def _validate_cards(self):
        """Validate that we have necessary agent types"""
        types = {card.agent_type for card in self.agent_cards.values()}
        required_types = {AgentType.CLIENT, AgentType.SERVER, AgentType.HOST}
        missing = required_types - types
        if missing:
            raise ValueError(f"Missing required agent types: {missing}")
    
    def get_card(self, agent_id: str) -> Optional[AgentCard]:
        """Get an agent card by ID"""
        return self.agent_cards.get(agent_id)
    
    def get_cards_by_type(self, agent_type: AgentType) -> List[AgentCard]:
        """Get all agent cards of a specific type"""
        return [
            card for card in self.agent_cards.values()
            if card.agent_type == agent_type
        ]
    
    def validate_interaction(self, source_id: str, target_id: str) -> bool:
        """Validate if two agents can interact based on their types"""
        source = self.get_card(source_id)
        target = self.get_card(target_id)
        
        if not source or not target:
            return False
        
        # Define valid interactions
        valid_interactions = {
            AgentType.CLIENT: [AgentType.SERVER],
            AgentType.SERVER: [AgentType.HOST],
            AgentType.HOST: [AgentType.SERVER]
        }
        
        return target.agent_type in valid_interactions.get(source.agent_type, []) 