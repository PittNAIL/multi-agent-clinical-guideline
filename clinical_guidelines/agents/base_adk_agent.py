from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from abc import ABC, abstractmethod
import uuid
import json
from datetime import datetime

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.memory import Memory, MemoryConfig
from google.adk.tools.streaming import StreamingConfig
from google.adk.tools.interaction import UserInteraction, InteractionConfig

from ..schemas.base_schemas import ClinicalRole
from ..schemas.agent_card import AgentCard, AgentType
from common.types import Task, Message

class BaseADKAgent(ABC):
    """Base class for all specialist agents using Google's ADK framework.
    
    This class provides:
    1. Common initialization and setup
    2. Agent card management
    3. Session state handling
    4. Memory management
    5. Streaming support
    6. User interaction handling
    7. Abstract methods for specialist-specific functionality
    """
    
    def __init__(
        self,
        role: ClinicalRole,
        guideline_chunks: List[str],
        name: str,
        description: str,
        stream_handler: Optional[Callable[[str], None]] = None,
        memory_config: Optional[MemoryConfig] = None,
        interaction_config: Optional[InteractionConfig] = None
    ):
        self.role = role
        self.guidelines = guideline_chunks
        self.name = name
        self.description = description
        self.stream_handler = stream_handler
        self.agent_card = self._create_agent_card()
        self.active_tasks: Dict[str, Task] = {}
        
        # Initialize memory
        self.memory = Memory(
            config=memory_config or MemoryConfig(
                max_items=1000,
                ttl_seconds=3600,
                embeddings_model="text-embedding-3-small"
            )
        )
        
        # Initialize streaming
        self.streaming_config = StreamingConfig(
            chunk_size=100,
            handler=self._handle_stream if stream_handler else None
        )
        
        # Initialize user interaction
        self.interaction = UserInteraction(
            config=interaction_config or InteractionConfig(
                timeout_seconds=300,
                max_retries=3
            )
        )
        
    def _create_agent_card(self) -> AgentCard:
        """Create the agent's capability card"""
        return AgentCard(
            agent_id=f"{self.role.value}-{uuid.uuid4()}",
            name=self.name,
            role=self.role.value,
            description=self.description,
            version="1.0.0",
            agent_type=AgentType.SPECIALIST,
            capabilities=self._get_capabilities(),
            constraints=self._get_constraints(),
            protocols=self._get_protocols(),
            required_inputs=self._get_required_inputs(),
            provided_outputs=self._get_provided_outputs(),
            performance_metrics=self._get_performance_metrics(),
            trust_score=0.85,
            specialization=self.role.value
        )
        
    def create_agent(self) -> Agent:
        """Create the Google ADK agent"""
        return Agent(
            model='gemini-pro',
            name=f'{self.role.value}_agent',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            after_model_callback=self.after_model_callback,
            description=self.description,
            tools=self._get_tools(),
            memory=self.memory,
            streaming_config=self.streaming_config,
            interaction=self.interaction
        )
        
    def root_instruction(self, context: ReadonlyContext) -> str:
        """Root instruction for the agent"""
        tools_desc = "\n".join([f"- {tool.__name__}: {tool.__doc__}" for tool in self._get_tools()])
        memory_context = self._get_relevant_memories(context)
        
        return f"""You are an expert {self.role.value} specializing in {self.description}.

Your capabilities include:
{self._get_capabilities_description()}

Tools available:
{tools_desc}

Guidelines:
{self.guidelines}

Previous Context:
{memory_context}

Please rely on tools to address requests, and don't make up responses. If you are unsure, ask for more details.
Focus on the most recent parts of the conversation primarily.

You can:
1. Stream responses in real-time
2. Ask users for clarification when needed
3. Access previous conversation context
4. Save important information for later use"""
        
    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        """Handle state before model call"""
        state = callback_context.state
        if 'session_id' not in state:
            state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True
            
        # Add memory context
        if 'memory' not in state:
            state['memory'] = []
            
    def after_model_callback(self, callback_context: CallbackContext, llm_response):
        """Handle state after model call"""
        # Save response to memory
        self._save_to_memory(
            callback_context.state['session_id'],
            llm_response.text
        )
        
    async def _handle_stream(self, chunk: str):
        """Handle streaming chunks"""
        if self.stream_handler:
            await self.stream_handler(chunk)
            
    def _save_to_memory(self, session_id: str, content: str):
        """Save content to memory"""
        self.memory.add(
            key=f"{session_id}_{datetime.now().isoformat()}",
            content=content
        )
        
    def _get_relevant_memories(self, context: ReadonlyContext) -> str:
        """Get relevant memories for current context"""
        query = context.current_message.text
        memories = self.memory.search(
            query=query,
            limit=5
        )
        return "\n".join([m.content for m in memories])
        
    async def ask_user(
        self,
        question: str,
        tool_context: ToolContext,
        options: Optional[List[str]] = None
    ) -> str:
        """Ask user for input with optional choices"""
        response = await self.interaction.ask(
            question=question,
            options=options,
            tool_context=tool_context
        )
        return response.text
        
    async def stream_response(
        self,
        content: str,
        tool_context: ToolContext
    ) -> AsyncGenerator[str, None]:
        """Stream response in chunks"""
        async for chunk in self.streaming_config.stream(
            content=content,
            tool_context=tool_context
        ):
            yield chunk
            
    async def check_guidelines(
        self,
        clinical_scenario: str,
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """Check relevant guidelines for the clinical scenario.
        
        Args:
            clinical_scenario: Description of the clinical scenario
            tool_context: Tool context
            
        Returns:
            Dictionary containing relevant guidelines
        """
        # Get relevant memories
        memories = self._get_relevant_memories(tool_context.context)
        
        prompt = f"""Find relevant guidelines for this clinical scenario:
        
        Scenario:
        {clinical_scenario}
        
        Previous Context:
        {memories}
        
        Guidelines:
        {self.guidelines}
        
        Return specific guideline recommendations and references.
        """
        
        # Stream the response
        response_chunks = []
        async for chunk in self.stream_response(prompt, tool_context):
            response_chunks.append(chunk)
            
        # Save to memory
        response_text = "".join(response_chunks)
        self._save_to_memory(
            tool_context.context.state['session_id'],
            response_text
        )
        
        return {
            "guideline_matches": response_text,
            "confidence": 0.9
        }
        
    @abstractmethod
    def _get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        pass
        
    @abstractmethod
    def _get_capabilities_description(self) -> str:
        """Get detailed description of capabilities"""
        pass
        
    @abstractmethod
    def _get_constraints(self) -> List[Dict[str, Any]]:
        """Get agent constraints"""
        pass
        
    @abstractmethod
    def _get_protocols(self) -> List[Dict[str, Any]]:
        """Get communication protocols"""
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
    def _get_tools(self) -> List[Any]:
        """Get list of available tools"""
        pass
        
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get default performance metrics"""
        return {
            "accuracy": 0.9,
            "response_time": 2.0,
            "reliability": 0.95
        } 