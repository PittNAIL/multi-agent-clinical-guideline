from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator, Callable
from datetime import datetime
import json
import asyncio
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import AsyncCallbackHandler
from ..tools.base_tools import ClinicalTools
from ..schemas.base_schemas import (
    PatientState,
    AgentResponse,
    AgentDecision,
    ClinicalRole,
    GuidelineMatch
)
from langchain.agents import AgentExecutor, create_react_agent
from langchain_huggingface import HuggingFacePipeline
from langchain.tools import Tool
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import torch
from langchain_core.language_models import BaseLLM
from ..schemas.agent_card import AgentCard, AgentCapability, AgentConstraint, AgentProtocol

class StreamingCallback(AsyncCallbackHandler):
    """Callback handler for streaming responses"""
    def __init__(self, stream_handler: Callable[[str], None]):
        self.stream_handler = stream_handler

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stream tokens as they're generated"""
        await self.stream_handler(token)

class BaseAgent(ABC):
    """Base class for all clinical agents following A2A protocol"""
    
    _shared_model = None
    _shared_tokenizer = None
    
    def __init__(
        self,
        role: ClinicalRole,
        guideline_chunks: List[str],
        llm: Optional[BaseLLM] = None,
        stream_handler: Optional[Callable[[str], None]] = None
    ):
        self.role = role
        self.guidelines = guideline_chunks
        self.stream_handler = stream_handler
        self.agent_card = self._create_agent_card()
        self.callbacks = [StreamingCallback(stream_handler)] if stream_handler else []
        
        # Initialize LLM if not provided
        if llm is None:
            self.llm = self._init_llm()
        else:
            self.llm = llm

    def _create_agent_card(self) -> AgentCard:
        """Create A2A protocol agent card"""
        return AgentCard(
            agent_id=f"{self.role.value}_agent",
            name=f"{self.role.value.capitalize()} Specialist",
            role=self.role.value,
            description=f"Clinical specialist agent for {self.role.value} decisions",
            version="1.0.0",
            capabilities=self._get_capabilities(),
            constraints=[],
            protocols=[
                AgentProtocol(
                    input_format={"patient_state": "PatientState"},
                    output_format={"decision": "AgentDecision"},
                    communication_pattern="request-response"
                )
            ],
            required_inputs=self._get_required_inputs(),
            provided_outputs=self._get_provided_outputs(),
            performance_metrics={
                "accuracy": 0.85,
                "response_time": 2.0
            },
            trust_score=0.8,
            specialization=self.role.value,
            interaction_patterns=[
                "guideline-based-decision",
                "human-in-the-loop",
                "multi-agent-collaboration"
            ]
        )

    @abstractmethod
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get agent-specific capabilities"""
        pass

    @abstractmethod
    def _get_required_inputs(self) -> List[str]:
        """Get required input fields"""
        pass

    @abstractmethod
    def _get_provided_outputs(self) -> List[str]:
        """Get provided output fields"""
        pass

    @abstractmethod
    async def validate_input(self, state: PatientState) -> bool:
        """Validate input state"""
        pass

    @abstractmethod
    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Process guidelines for current state"""
        pass

    @abstractmethod
    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Make clinical decision"""
        pass

    def _validate_state(self, state: PatientState) -> bool:
        """Basic state validation"""
        return (
            hasattr(state, "age") and
            state.age is not None and
            hasattr(state, "current_symptoms") and
            isinstance(state.current_symptoms, list)
        )

    async def request_user_input(self, field: str, description: str) -> Any:
        """Request input from user for missing information"""
        if not hasattr(self, 'user_input_handler'):
            raise ValueError("No user input handler configured")
        
        return await self.user_input_handler(
            prompt=f"Please provide {description} for {field}:",
            field_name=field,
            agent_role=self.role.value
        )

    async def validate_and_request_missing(self, state: PatientState) -> PatientState:
        """Validate state and request missing required inputs"""
        required_inputs = self._get_required_inputs()
        missing_fields = []
        
        for field in required_inputs:
            if not hasattr(state, field) or getattr(state, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            updated_state = state.copy()
            for field in missing_fields:
                value = await self.request_user_input(
                    field,
                    self._get_field_description(field)
                )
                setattr(updated_state, field, value)
            return updated_state
        
        return state

    @abstractmethod
    def _get_field_description(self, field: str) -> str:
        """Get description for a required field"""
        pass

    async def stream_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AsyncGenerator[str, None]:
        """Stream decision making process"""
        decision = await self.make_decision(state, relevant_guidelines)
        
        # Stream each part of the decision
        yield f"Decision from {self.role.value} agent:\n"
        yield f"Main decision: {decision.decision}\n\n"
        
        if decision.next_steps:
            yield "Next steps:\n"
            for step in decision.next_steps:
                yield f"- {step}\n"
            yield "\n"

    async def run(self, state: PatientState) -> Dict[str, Any]:
        """Run agent following A2A protocol with streaming and user interaction"""
        # First request any missing required inputs
        state = await self.validate_and_request_missing(state)
        
        if not await self.validate_input(state):
            raise ValueError(f"Invalid input state for {self.role.value} agent")

        # Process guidelines
        relevant_guidelines = await self.process_guidelines(state)

        # Stream decision making process
        if self.stream_handler:
            async for token in self.stream_decision(state, relevant_guidelines):
                await self.stream_handler(token)

        # Make final decision
        decision = await self.make_decision(state, relevant_guidelines)

        return {
            "status": "success",
            "agent_card": self.agent_card,
            "decision": decision,
            "guidelines_used": [g.guideline_id for g in relevant_guidelines],
            "timestamp": datetime.now()
        }

    def _init_llm(self) -> BaseLLM:
        """Initialize the language model"""
        import os
        import torch
        
        # Set PyTorch settings
        torch.backends.cudnn.benchmark = True
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        # Create offload directory if it doesn't exist
        offload_folder = os.path.join(os.getcwd(), "model_offload")
        os.makedirs(offload_folder, exist_ok=True)

        # Use shared model if available
        if BaseAgent._shared_model is None:
            # Load model and tokenizer
            model_name = "facebook/opt-350m"  # Using a smaller model
            try:
                BaseAgent._shared_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map={"": device},
                    offload_folder=offload_folder,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                BaseAgent._shared_tokenizer = AutoTokenizer.from_pretrained(model_name)
                BaseAgent._shared_model.eval()  # Set to evaluation mode
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

        # Create text generation pipeline using shared model
        pipe = pipeline(
            "text-generation",
            model=BaseAgent._shared_model,
            tokenizer=BaseAgent._shared_tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            device_map={"": device}
        )
        
        # Return HuggingFacePipeline from langchain_huggingface
        return HuggingFacePipeline(pipeline=pipe)

    def _init_model(self):
        """Initialize the model."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )
        
        return model, tokenizer

class BaseClinicalAgent:
    """Base class for all clinical agents using LangChain's agent framework."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        guidelines: Dict[str, str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        self.model_name = model_name
        self.guidelines = guidelines or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize model with 8-bit quantization
        self.model, self.tokenizer = self._init_model()
        self.llm = self._init_llm()
        self.tools = self._get_tools()
        self.agent_executor = self._create_agent()
    
    def _init_model(self):
        """Initialize the model."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto"
        )
        
        return model, tokenizer
    
    def _init_llm(self):
        """Initialize LangChain LLM wrapper."""
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            repetition_penalty=1.1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def _get_tools(self) -> List[Tool]:
        """Get the tools available to this agent."""
        return [
            Tool(
                name="search_guidelines",
                func=self._search_guidelines,
                description="Search through clinical guidelines based on query"
            )
        ]
    
    def _search_guidelines(self, query: str) -> str:
        """Search through guidelines based on query."""
        relevant = []
        for guideline_id, content in self.guidelines.items():
            if query.lower() in content.lower():
                relevant.append(f"{guideline_id}: {content}")
        return "\\n".join(relevant) if relevant else "No relevant guidelines found."
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical specialist agent that follows clinical guidelines. "
                      "Use the available tools to search guidelines when needed."),
            ("user", "{input}")
        ])
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    async def process(self, query: str) -> Dict[str, Any]:
        """Process a query through the agent."""
        return await self.agent_executor.ainvoke({"input": query})

    async def _validate_state(self, state: PatientState) -> bool:
        """Base validation all agents should perform"""
        return (
            hasattr(state, "patient_id") and
            state.patient_id is not None and
            hasattr(state, "treatment_phase") and
            state.treatment_phase is not None
        )
    
    @abstractmethod
    async def validate_input(self, state: PatientState) -> bool:
        """Validate that the state has required information"""
        pass
    
    @abstractmethod
    async def process_guidelines(self, state: PatientState) -> List[GuidelineMatch]:
        """Process guidelines to find relevant matches"""
        pass
    
    @abstractmethod
    async def make_decision(
        self,
        state: PatientState,
        relevant_guidelines: List[GuidelineMatch]
    ) -> AgentDecision:
        """Make a clinical decision based on guidelines"""
        pass
    
    async def process_case(self, state: PatientState) -> PatientState:
        """Process a patient case"""
        # Validate input
        if not await self.validate_input(state):
            return AgentDecision(
                role=self.role,
                timestamp=datetime.now(),
                decision="Invalid or insufficient input data",
                next_steps=["Gather required information"],
                requires_escalation=True,
                escalation_reason="Missing required data",
                supporting_data={}
            )
        
        # Find relevant guidelines
        relevant_guidelines = await self.process_guidelines(state)
        
        # Make decision
        decision = await self.make_decision(state, relevant_guidelines)
        
        # Update state
        state.add_decision(decision)
        if decision.requires_escalation:
            state.requires_human_review = True
        
        return state
    
    def get_agent_card(self) -> AgentCard:
        """Get agent's capability card"""
        return AgentCard(
            name=f"{self.role.value.title()} Agent",
            role=self.role.value,
            capabilities=self._get_capabilities(),
            required_inputs=self._get_required_inputs(),
            provided_outputs=self._get_provided_outputs(),
            description=f"Clinical guideline agent for {self.role.value} decisions"
        )
    
    @abstractmethod
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get agent's capabilities"""
        pass
    
    @abstractmethod
    def _get_required_inputs(self) -> List[str]:
        """Get required input fields"""
        pass
    
    @abstractmethod
    def _get_provided_outputs(self) -> List[str]:
        """Get provided output fields"""
        pass 