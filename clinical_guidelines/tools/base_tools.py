from langchain_core.tools import Tool
from typing import Dict, Any, List, Optional
from ..schemas.base_schemas import PatientState
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re
from datetime import datetime
import os

def search_guidelines(query: str, guidelines: Dict[str, str]) -> str:
    """Search through clinical guidelines based on query."""
    # Simple keyword matching - can be enhanced with semantic search
    relevant = []
    for guideline_id, content in guidelines.items():
        if query.lower() in content.lower():
            relevant.append(f"{guideline_id}: {content}")
    return "\n".join(relevant) if relevant else "No relevant guidelines found."

def update_patient_state(current_state: PatientState, updates: Dict[str, Any]) -> PatientState:
    """Update patient state with new information."""
    for key, value in updates.items():
        if hasattr(current_state, key):
            setattr(current_state, key, value)
    return current_state

class ClinicalTools:
    def __init__(self, guideline_store_path: str = "guideline_store"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Check if vector store exists
        if not os.path.exists(guideline_store_path):
            raise RuntimeError(
                "Vector store not found. Please run setup_models.py first to initialize the models and database."
            )
        
        self.guideline_store = Chroma(
            persist_directory=guideline_store_path,
            embedding_function=self.embeddings
        )
        
        # Medical terminology regex patterns
        self.icd_pattern = re.compile(r'^[A-Z]\d{2}(\.\d+)?$')
        self.snomed_pattern = re.compile(r'^\d{6,}$')
    
    def search_guidelines(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search through clinical guidelines"""
        results = self.guideline_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def validate_clinical_code(self, code: str, code_type: str) -> bool:
        """Validate clinical coding"""
        if code_type.lower() == "icd":
            return bool(self.icd_pattern.match(code))
        elif code_type.lower() == "snomed":
            return bool(self.snomed_pattern.match(code))
        return False
    
    def validate_date_sequence(self, dates: List[datetime], sequence_type: str) -> bool:
        """Validate clinical date sequences"""
        if not dates:
            return True
            
        sorted_dates = sorted(dates)
        if sequence_type == "treatment":
            # Treatment dates should be within reasonable timeframe
            time_span = (sorted_dates[-1] - sorted_dates[0]).days
            return time_span <= 365  # One year treatment window
        
        elif sequence_type == "follow_up":
            # Follow-up dates should have reasonable intervals
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            return all(14 <= interval <= 180 for interval in intervals)  # 2 weeks to 6 months
        
        return True
    
    def get_tools(self) -> List[Tool]:
        """Get all available clinical tools"""
        return [
            Tool(
                name="search_guidelines",
                func=self.search_guidelines,
                description="Search through clinical guidelines using semantic search"
            ),
            Tool(
                name="validate_clinical_code",
                func=self.validate_clinical_code,
                description="Validate ICD or SNOMED clinical codes"
            ),
            Tool(
                name="validate_date_sequence",
                func=self.validate_date_sequence,
                description="Validate sequences of clinical dates"
            )
        ] 