from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from ..schemas.base_schemas import PatientState, AgentDecision

class HumanReview(BaseModel):
    reviewer_id: str
    decision: str
    notes: str
    approved: bool
    timestamp: datetime = datetime.now()

class ReviewInterface:
    def __init__(self):
        self.pending_reviews: Dict[str, PatientState] = {}
        self.completed_reviews: Dict[str, List[HumanReview]] = {}
    
    async def request_review(self, state: PatientState, reason: str) -> str:
        """Queue a case for human review"""
        review_id = f"REV_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.pending_reviews[review_id] = state
        return review_id
    
    async def get_pending_reviews(self) -> List[Dict]:
        """Get all cases pending review"""
        return [
            {
                "review_id": rid,
                "patient_id": state.patient_id,
                "current_phase": state.treatment_phase,
                "last_decision": state.agent_decisions[-1] if state.agent_decisions else None
            }
            for rid, state in self.pending_reviews.items()
        ]
    
    async def submit_review(self, review_id: str, review: HumanReview) -> PatientState:
        """Submit a human review decision"""
        if review_id not in self.pending_reviews:
            raise ValueError(f"No pending review found for ID: {review_id}")
        
        state = self.pending_reviews.pop(review_id)
        
        # Record the review
        if review_id not in self.completed_reviews:
            self.completed_reviews[review_id] = []
        self.completed_reviews[review_id].append(review)
        
        # Update state with human decision
        state.add_clinical_note(
            f"Human Review by {review.reviewer_id}: {review.decision}",
            "HUMAN_REVIEWER"
        )
        state.requires_human_review = False
        
        return state
    
    async def get_review_history(self, review_id: str) -> List[HumanReview]:
        """Get history of reviews for a case"""
        return self.completed_reviews.get(review_id, []) 