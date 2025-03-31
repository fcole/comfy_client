import json
import asyncio
import aiohttp
import websockets
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from comfy_utils import find_comfy_port, update_workflow, UnusedColumnsError

@dataclass
class QueuedPrompt:
    workflow: Dict[str, Any]
    future: asyncio.Future
    prompt_id: Optional[str] = None
    queued_at: datetime = None


class ComfyClient:
    def __init__(self, host: str, port: int):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.ws = None
        self.ws_logger = logging.getLogger('comfy_client.websocket')
        self.ws_logger.setLevel(logging.WARNING)
        self.prompt_queue = asyncio.Queue()
        self.current_prompt: Optional[QueuedPrompt] = None
        self.prompt_futures: Dict[str, asyncio.Future] = {}  # Map prompt_id to Future
        self.should_stop = False

    @classmethod
    async def create(cls, host: str = "127.0.0.1", port: Optional[int] = None):
        """Initialize the client by finding the port if not specified."""
        port = port or await find_comfy_port()
        if port is None:
            raise RuntimeError("Could not find running ComfyUI server")
        
        client = cls(host, port)
        client.ws_task = asyncio.create_task(client._websocket_handler())
        client.queue_task = asyncio.create_task(client._process_queue())

        # Test if the ComfyUI server is running and accessible
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{client.base_url}/system_stats") as response:
                if response.status != 200:
                    raise RuntimeError(f"Could not connect to ComfyUI server: {response.status}")

        return client

    async def _websocket_handler(self):
        """Handle WebSocket connection and messages."""
        async with websockets.connect(self.ws_url, logger=self.ws_logger) as websocket:
            self.ws = websocket
            while not self.should_stop:
                try:
                    message_str = await websocket.recv()
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("WebSocket connection closed")
                    break

                message = json.loads(message_str)
                message_type, message_data = message["type"], message["data"]

                if message_type == "crystaltools.monitor":
                    # Skip crystools monitoring messages completely
                    continue

                if message_type == "progress":
                    step = message_data["value"]
                    max_step = message_data["max"]
                    self.logger.info(f"Step: {step} / {max_step}")
                elif message_type == "status":
                    # Log status messages for debugging
                    queue_remaining = message_data["status"]["exec_info"]["queue_remaining"]
                    if queue_remaining == 0 and self.current_prompt:
                        # Resolve the future for the current prompt
                        future = self.prompt_futures.get(self.current_prompt.prompt_id)
                        future.set_result(True)
                        del self.prompt_futures[self.current_prompt.prompt_id]
                        self.current_prompt = None
                else:
                    self.logger.debug(f"Unknown message type: {message_type}")

    async def _process_queue(self):
        """Process the queue of prompts."""
        while not self.should_stop:
            # If we have a current prompt, wait for it to complete
            if self.current_prompt:
                await asyncio.sleep(0.1)
                continue

            # Get next prompt from queue
            try:
                queued_prompt = await asyncio.wait_for(self.prompt_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # Submit the prompt to the server
            self.logger.info("Sending workflow to ComfyUI...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/prompt",
                    json={"prompt": queued_prompt.workflow}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Server returned status code: {response.status}")
                        self.logger.error(f"Server response: {error_text}")
                        queued_prompt.future.set_exception(Exception(f"Failed to queue prompt: {error_text}"))
                        return

                    result = await response.json()
                    prompt_id = result["prompt_id"]
                    self.logger.info(f"Successfully queued prompt with ID: {prompt_id}")
                    queued_prompt.prompt_id = prompt_id
                    queued_prompt.queued_at = datetime.now()
                    self.current_prompt = queued_prompt
                    self.prompt_futures[prompt_id] = queued_prompt.future


    def queue_prompt(self, workflow: Dict[str, Any]) -> asyncio.Future:
        """Queue a prompt and return a Future that will be resolved when the prompt completes."""
        future = asyncio.Future()
        queued_prompt = QueuedPrompt(workflow=workflow, future=future)
        # Use create_task to schedule the queue put operation
        asyncio.create_task(self.prompt_queue.put(queued_prompt))
        return future


    def queue_sheet(self, workflow: Dict[str, Any], rows: list[Dict[str, str]]) -> Dict[str, asyncio.Future]:
        """Queue all rows for processing.
        
        Args:
            workflow: The base workflow to use
            rows: List of dictionaries, where each dictionary represents a row of data
            
        Returns:
            Dict mapping filename prefixes to their futures
        """
        futures = {}
        for row in rows:
            prefix = row['output.filename_prefix']
            updated_workflow = update_workflow(workflow, row)
            futures[prefix] = self.queue_prompt(updated_workflow)
        return futures


    async def close(self):
        """Close the WebSocket connection and stop the tasks."""
        self.should_stop = True
        if hasattr(self, 'ws_task'):
            self.ws_task.cancel()
        if hasattr(self, 'queue_task'):
            self.queue_task.cancel()
        try:
            await asyncio.gather(self.ws_task, self.queue_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass 