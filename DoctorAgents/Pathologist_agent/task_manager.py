import json
import logging
import asyncio

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from typing import Any

from common.server import utils
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    Artifact,
    InternalError,
    JSONRPCResponse,
    Message,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from google.genai import types


logger = logging.getLogger(__name__)


class AgentWithTaskManager(ABC):
    @abstractmethod
    def get_processing_message(self) -> str:
        pass

    async def invoke(self, query: str, session_id: str) -> str:
        """
        Asynchronous method that:
          1. Awaits session_service.get_session(...)
          2. If no session, then awaits session_service.create_session(...)
          3. Awaits Runner.run_async(...)
        """
        # 1. Await get_session
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )

        # 2. If no session exists, create one
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},          # initial empty state
                session_id=session_id,
            )

        # 3. Build a Content object for the user query
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )

        # 4. Run the LLM and gather all events
        events = []
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=session.id,
            new_message=content,
        ):
            events.append(event)

        # 5. Extract the "final" content from the last event
        if not events or not events[-1].content or not events[-1].content.parts:
            return ""

        # Join any text parts into a single string:
        return "\n".join(
            [p.text for p in events[-1].content.parts if p.text]
        )

    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        """
        Asynchronous streaming version, similarly "awaiting" session_service calls.
        """
        # 1. Await get_session(...) 
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )

        # 2. If session is None, create it
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )

        # 3. Build a Content
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)]
        )

        # 4. Stream events and yield them
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=session.id,
            new_message=content
        ):
            if event.is_final_response():
                # If the final response is a JSON‐structured function output:
                if (
                    event.content
                    and event.content.parts
                    and any(p.function_response for p in event.content.parts)
                ):
                    # pick the first part with function_response
                    model_response = next(
                        p.function_response.model_dump()
                        for p in event.content.parts
                        if p.function_response
                    )
                    yield {
                        "is_task_complete": True,
                        "content": model_response
                    }

                # Otherwise, gather plain‐text parts
                else:
                    yield {
                        "is_task_complete": True,
                        "content": "\n".join(
                            [p.text for p in event.content.parts if p.text]
                        )
                    }

            else:
                # Intermediate "processing" update
                yield {
                    "is_task_complete": False,
                    "updates": self.get_processing_message()
                }



class AgentTaskManager(InMemoryTaskManager):
    def __init__(self, agent: AgentWithTaskManager):
        super().__init__()
        self.agent = agent

    async def _stream_generator(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        try:
            async for item in self.agent.stream(
                query, task_send_params.sessionId
            ):
                is_task_complete = item['is_task_complete']
                artifacts = None
                if not is_task_complete:
                    task_state = TaskState.WORKING
                    parts = [{'type': 'text', 'text': item['updates']}]
                else:
                    if isinstance(item['content'], dict):
                        if (
                            'response' in item['content']
                            and 'result' in item['content']['response']
                        ):
                            data = json.loads(
                                item['content']['response']['result']
                            )
                            task_state = TaskState.INPUT_REQUIRED
                        else:
                            data = item['content']
                            task_state = TaskState.COMPLETED
                        parts = [{'type': 'data', 'data': data}]
                    else:
                        task_state = TaskState.COMPLETED
                        parts = [{'type': 'text', 'text': item['content']}]
                    artifacts = [Artifact(parts=parts, index=0, append=False)]
            message = Message(role='agent', parts=parts)
            task_status = TaskStatus(state=task_state, message=message)
            await self._update_store(
                task_send_params.id, task_status, artifacts
            )
            task_update_event = TaskStatusUpdateEvent(
                id=task_send_params.id,
                status=task_status,
                final=False,
            )
            yield SendTaskStreamingResponse(
                id=request.id, result=task_update_event
            )
            # Now yield Artifacts too
            if artifacts:
                for artifact in artifacts:
                    yield SendTaskStreamingResponse(
                        id=request.id,
                        result=TaskArtifactUpdateEvent(
                            id=task_send_params.id,
                            artifact=artifact,
                        ),
                    )
            if is_task_complete:
                yield SendTaskStreamingResponse(
                    id=request.id,
                    result=TaskStatusUpdateEvent(
                        id=task_send_params.id,
                        status=TaskStatus(
                            state=task_status.state,
                        ),
                        final=True,
                    ),
                )
        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            yield JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message='An error occurred while streaming the response'
                ),
            )

    def _validate_request(
        self, request: SendTaskRequest | SendTaskStreamingRequest
    ) -> None:
        task_send_params: TaskSendParams = request.params
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes,
            self.agent.SUPPORTED_CONTENT_TYPES,
        ):
            logger.warning(
                'Unsupported output mode. Received %s, Support %s',
                task_send_params.acceptedOutputModes,
                self.agent.SUPPORTED_CONTENT_TYPES,
            )
            return utils.new_incompatible_types_error(request.id)

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        error = self._validate_request(request)
        if error:
            return error
        await self.upsert_task(request.params)
        return await self._invoke(request)

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        error = self._validate_request(request)
        if error:
            return error
        await self.upsert_task(request.params)
        return self._stream_generator(request)

    async def _update_store(
        self, task_id: str, status: TaskStatus, artifacts: list[Artifact]
    ) -> Task:
        async with self.lock:
            try:
                task = self.tasks[task_id]
            except KeyError:
                logger.error(f'Task {task_id} not found for updating the task')
                raise ValueError(f'Task {task_id} not found')
            task.status = status
            if artifacts is not None:
                if task.artifacts is None:
                    task.artifacts = []
                task.artifacts.extend(artifacts)
            return task

    async def _invoke(self, request: SendTaskRequest) -> SendTaskResponse:
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        try:
            response = await self.agent.invoke(query, task_send_params.sessionId)
            task_state = TaskState.COMPLETED
            parts = [{'type': 'text', 'text': response}]
            message = Message(role='agent', parts=parts)
            task_status = TaskStatus(state=task_state, message=message)
            task = await self._update_store(task_send_params.id, task_status, None)
            return SendTaskResponse(
                id=request.id,
                result=task
            )
        except Exception as e:
            logger.error(f'An error occurred while invoking the agent: {e}')
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message='An error occurred while invoking the agent'
                ),
            )

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        if not task_send_params.message or not task_send_params.message.parts:
            return ''
        return task_send_params.message.parts[0].text 