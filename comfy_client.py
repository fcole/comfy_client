import json
import asyncio
import aiohttp
import websockets
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from comfy_utils import find_comfy_port, update_workflow, get_workflow_hash

@dataclass
class QueuedPrompt:
    workflow: Dict[str, Any]
    future: asyncio.Future
    workflow_hash: str
    prompt_id: Optional[str] = None
    queued_at: datetime = None


class ComfyClient:
    def __init__(self, host: str, port: int, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.log_file = log_file
        self.completed_hashes = set()
        if self.log_file and self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.completed_hashes = set(line.strip() for line in f)
            self.logger.info(f"Loaded {len(self.completed_hashes)} completed hashes from {self.log_file}")

        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.ws = None
        self.ws_logger = logging.getLogger('comfy_client.websocket')
        self.ws_logger.setLevel(logging.WARNING)
        self.prompt_queue = asyncio.Queue()
        self.current_prompt: Optional[QueuedPrompt] = None
        self.should_stop = False

    @classmethod
    async def create(cls, host: str = "127.0.0.1", port: Optional[int] = None, log_file: Optional[Path] = None):
        """Initialize the client by finding the port if not specified."""
        port = port or await find_comfy_port()
        if port is None:
            raise RuntimeError("Could not find running ComfyUI server")
        
        client = cls(host, port, log_file)
        client.ws_task = asyncio.create_task(client._websocket_handler())
        client.queue_task = asyncio.create_task(client._process_queue())

        # Test if the ComfyUI server is running and accessible
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{client.base_url}/system_stats") as response:
                if response.status != 200:
                    raise RuntimeError(f"Could not connect to ComfyUI server: {response.status}")

        return client

    async def _check_prompt_history(self, prompt_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check the history of a prompt to see if it was successful.
        Returns a tuple of (success, message). Message is an error message on failure.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                if response.status != 200:
                    error_msg = f"Failed to get history for prompt {prompt_id}: {response.status} {await response.text()}"
                    self.logger.error(error_msg)
                    return False, error_msg

                history = await response.json()
                prompt_history = history.get(prompt_id)

                if not prompt_history:
                    warning_msg = f"Prompt {prompt_id} not found in history."
                    self.logger.warning(warning_msg)
                    return False, warning_msg

                if 'status' in prompt_history:
                    status = prompt_history['status']
                    if status.get('status_str') == 'error' or not status.get('completed'):
                        messages = status.get('messages', [])
                        error_message = 'Unknown error'
                        for msg in messages:
                            if msg[0] == 'execution_error':
                                error_message = msg[1].get('exception_message', 'Unknown error')
                                break
                        self.logger.error(f"Prompt {prompt_id} failed with error: {error_message}")
                        return False, error_message

                if 'outputs' in prompt_history and prompt_history['outputs']:
                    self.logger.info(f"Prompt {prompt_id} completed successfully.")
                    return True, None

                # Fallback if no explicit error but also no output
                warning_msg = f"Prompt {prompt_id} completed with no outputs and no explicit error status."
                self.logger.warning(warning_msg)
                return False, warning_msg

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
                        prompt_id = self.current_prompt.prompt_id
                        if prompt_id is None:
                            self.logger.error("Current prompt has no ID, cannot check history.")
                            self.current_prompt.future.set_exception(Exception("Prompt has no ID, cannot check history."))
                            self.current_prompt = None
                            continue

                        success, result_message = await self._check_prompt_history(prompt_id)
                        if success:
                            self.current_prompt.future.set_result(True)

                            # Log successful completion
                            if self.log_file:
                                workflow_hash = self.current_prompt.workflow_hash
                                with open(self.log_file, 'a') as f:
                                    f.write(f"{workflow_hash}\n")
                                self.completed_hashes.add(workflow_hash)
                                self.logger.info(f"Logged completed workflow: {workflow_hash[:8]}...")
                        else:
                            self.current_prompt.future.set_exception(Exception(result_message))

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


    def queue_prompt(self, workflow: Dict[str, Any]) -> asyncio.Future:
        """Queue a prompt and return a Future that will be resolved when the prompt completes."""
        workflow_hash = get_workflow_hash(workflow)

        if workflow_hash in self.completed_hashes:
            future = asyncio.Future()
            future.set_result("skipped")
            return future

        future = asyncio.Future()
        queued_prompt = QueuedPrompt(workflow=workflow, future=future, workflow_hash=workflow_hash)
        # Use create_task to schedule the queue put operation
        asyncio.create_task(self.prompt_queue.put(queued_prompt))
        return future


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