# common/server.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import logging

from Common.types import TaskRequest

logger = logging.getLogger("A2AServer")

class A2AServer:
    def __init__(self, agent_card, task_manager, host="localhost", port=10006):
        self.agent_card = agent_card
        self.task_manager = task_manager
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/agent")
        async def get_card():
            return self.agent_card

        @self.app.post("/task")
        async def handle_task(request: Request):
            try:
                payload = await request.json()
                task_request = TaskRequest(**payload)
                response = await self.task_manager.handle_task(task_request)
                return response
            except Exception as e:
                logger.error(f"Task handling failed: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

    def start(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
