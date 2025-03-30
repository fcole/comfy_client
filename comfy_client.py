import json
import asyncio
import aiohttp
import websockets
import subprocess
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import difflib
import re


@dataclass
class QueuedPrompt:
    workflow: Dict[str, Any]
    future: asyncio.Future
    prompt_id: Optional[str] = None
    queued_at: datetime = None

class UnusedColumnsError(Exception):
    """Exception raised when CSV contains unused columns."""
    def __init__(self, unused_columns: set[str], suggestions: Dict[str, list[str]]):
        self.unused_columns = unused_columns
        self.suggestions = suggestions
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"The following CSV columns were not used: {', '.join(self.unused_columns)}\n"
        if self.suggestions:
            msg += "\nDid you mean:\n"
            for col, matches in self.suggestions.items():
                msg += f"  {col} -> {', '.join(matches)}\n"
        return msg

class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.base_url = None
        self.ws_url = None
        self.ws = None
        self.ws_logger = logging.getLogger('comfy_client.websocket')
        self.ws_logger.setLevel(logging.WARNING)
        self.prompt_queue = asyncio.Queue()
        self.current_prompt: Optional[QueuedPrompt] = None
        self.prompt_futures: Dict[str, asyncio.Future] = {}  # Map prompt_id to Future
        self.should_stop = False

    async def initialize(self):
        """Initialize the client by finding the port if not specified."""
        if self.port is None:
            self.port = await self._find_comfy_port()
            if self.port is None:
                raise RuntimeError("Could not find running ComfyUI server")
        
        self.base_url = f"http://{self.host}:{self.port}"
        self.ws_url = f"ws://{self.host}:{self.port}/ws"

    async def _check_port(self, port: int) -> Optional[int]:
        """Check if a port is running ComfyUI."""
        try:
            self.logger.info(f"Checking if port {port} is running ComfyUI...")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/system_stats", timeout=5) as response:
                    if response.status == 200:
                        self.logger.info(f"Found ComfyUI server on port {port}")
                        return port
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.logger.info(f"Port {port} is not running ComfyUI: {e}")
        return None

    async def _parse_port_from_line(self, line: str) -> Optional[int]:
        """Extract and validate port number from lsof output line.
        
        Example line: "Python    1234 user    4u   IPv4  0x1234567890abcdef      0t0    TCP *:8188 (LISTEN)"
        """
        # Match port number after the last colon in the line
        match = re.search(r':(\d+)\s*\(LISTEN\)$', line)
        if not match:
            self.logger.info(f"No port number found in line: {line}")
            return None
            
        try:
            return int(match.group(1))
        except ValueError as e:
            self.logger.info(f"Failed to parse port from match {match.group(1)}: {e}")
            return None

    async def _find_comfy_port(self) -> Optional[int]:
        """Find the port where ComfyUI is running using lsof on macOS."""
        try:
            self.logger.info("Starting port search...")
            cmd = ["lsof", "-i", "-P", "-n"]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                self.logger.error(f"lsof command failed with return code {result.returncode}")
                self.logger.error(f"stderr: {stderr.decode()}")
                return None
                
            # Parse the output to find listening ports
            for line in stdout.decode().splitlines():
                self.logger.debug(f"Processing line: {line}")
                # Look for Python processes that are listening
                if ("Python" in line or "python" in line) and "LISTEN" in line:
                    port = await self._parse_port_from_line(line)
                    if port:
                        if await self._check_port(port):
                            return port
            
            self.logger.error("No ComfyUI server found in listening ports")
            return None
            
        except subprocess.SubprocessError as e:
            self.logger.error(f"Failed to execute lsof command: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error while finding port: {e}")
            return None

    async def connect(self) -> bool:
        """Test if the ComfyUI server is running and accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system_stats") as response:
                    return response.status == 200
        except aiohttp.ClientError:
            return False

    async def _websocket_handler(self):
        """Handle WebSocket connection and messages."""
        try:
            async with websockets.connect(self.ws_url, logger=self.ws_logger) as websocket:
                self.ws = websocket
                while not self.should_stop:
                    try:
                        message = await websocket.recv()
                        await self._handle_message(message)
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.info("WebSocket connection closed")
                        break
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            if "type" in data:
                # Skip monitoring messages completely
                if data["type"] == "crystools.monitor":
                    return
                    
                if data["type"] == "progress":
                    step = data["data"]["value"]
                    max_step = data["data"]["max"]
                    self.logger.info(f"Step: {step} / {max_step}")
                elif data["type"] == "executed" and data["data"]["node"] is None:
                    if self.current_prompt:
                        self.logger.info("Image generation complete!")
                        # Resolve the future for the current prompt
                        future = self.prompt_futures.get(self.current_prompt.prompt_id)
                        if future:
                            future.set_result(True)
                            del self.prompt_futures[self.current_prompt.prompt_id]
                        self.current_prompt = None
                elif data["type"] == "status":
                    # Log status messages for debugging
                    if "data" in data and "status" in data["data"]:
                        status = data["data"]["status"]
                        self.logger.debug(f"Status message: {data}")  # Log full message
                        if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                            self.logger.debug("Queue empty status received")
                            if self.current_prompt:
                                # Resolve the future for the current prompt
                                future = self.prompt_futures.get(self.current_prompt.prompt_id)
                                if future:
                                    future.set_result(True)
                                    del self.prompt_futures[self.current_prompt.prompt_id]
                                self.current_prompt = None
                else:
                    self.logger.debug(f"Unknown message type: {data['type']}")
                    if "data" in data:
                        self.logger.debug(f"Message data: {data['data']}")
        except json.JSONDecodeError:
            pass

    async def _process_queue(self):
        """Process the queue of prompts."""
        while not self.should_stop:
            try:
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
                try:
                    self.logger.info("Sending workflow to ComfyUI...")
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/prompt",
                            json={"prompt": queued_prompt.workflow}
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                prompt_id = result["prompt_id"]
                                self.logger.info(f"Successfully queued prompt with ID: {prompt_id}")
                                queued_prompt.prompt_id = prompt_id
                                queued_prompt.queued_at = datetime.now()
                                self.current_prompt = queued_prompt
                                self.prompt_futures[prompt_id] = queued_prompt.future
                            else:
                                error_text = await response.text()
                                self.logger.error(f"Server returned status code: {response.status}")
                                self.logger.error(f"Server response: {error_text}")
                                queued_prompt.future.set_exception(Exception(f"Failed to queue prompt: {error_text}"))
                except Exception as e:
                    self.logger.error(f"Failed to queue prompt: {e}")
                    queued_prompt.future.set_exception(e)

            except Exception as e:
                self.logger.error(f"Error in queue processor: {e}")

    def queue_prompt(self, workflow: Dict[str, Any]) -> asyncio.Future:
        """Queue a prompt and return a Future that will be resolved when the prompt completes."""
        future = asyncio.Future()
        queued_prompt = QueuedPrompt(workflow=workflow, future=future)
        # Use create_task to schedule the queue put operation
        asyncio.create_task(self.prompt_queue.put(queued_prompt))
        return future

    async def start(self):
        """Start the client's tasks."""
        self.should_stop = False
        # Start WebSocket handler and queue processor tasks
        self.ws_task = asyncio.create_task(self._websocket_handler())
        self.queue_task = asyncio.create_task(self._process_queue())

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
            updated_workflow = self._update_workflow(workflow, row)
            futures[prefix] = self.queue_prompt(updated_workflow)
        return futures

    def _update_workflow(self, workflow: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
        """Update the workflow with values from the CSV row."""
        # Create a deep copy of the workflow to avoid modifying the original
        updated_workflow = json.loads(json.dumps(workflow))
        
        # Track which columns were used
        used_columns = set()
        
        # Map CSV column names to node inputs
        for node_id, node in updated_workflow.items():
            if "_meta" in node and "title" in node["_meta"]:
                node_title = node["_meta"]["title"].replace(" ", "_")
                # Look for a matching CSV column
                for column_name, value in row.items():
                    # Split column name into node title and field name
                    parts = column_name.split(".")
                    if len(parts) == 2 and parts[0] == node_title:
                        field_name = parts[1]
                        if "inputs" in node and field_name in node["inputs"]:
                            node["inputs"][field_name] = value
                            used_columns.add(column_name)
                            self.logger.debug(f"Updated node {node_id} ({node_title}) field {field_name} with value from column {column_name}")
        
        # Check for unused columns
        unused_columns = set(row.keys()) - used_columns
        if unused_columns:
            # Find suggestions for each unused column
            valid_node_titles = {
                node["_meta"]["title"].replace(" ", "_")
                for node in updated_workflow.values()
                if "_meta" in node and "title" in node["_meta"]
            }
        
            suggestions = {}
            for col in unused_columns:
                parts = col.split(".")
                if len(parts) == 2:
                    node_part = parts[0]
                    # Find close matches for the node part
                    matches = difflib.get_close_matches(node_part, valid_node_titles, n=3, cutoff=0.6)
                    if matches:
                        suggestions[col] = [f"{match}.{parts[1]}" for match in matches]
            
            raise UnusedColumnsError(unused_columns, suggestions)
        
        return updated_workflow

    async def wait_for_all(self, futures: Dict[str, asyncio.Future]) -> Dict[str, Any]:
        """Wait for all futures to complete and return results.
        
        Args:
            futures: Dict mapping prefixes to their futures
            
        Returns:
            Dict mapping prefixes to their results (True for success, Exception for failure)
        """
        results = {}
        for prefix, future in futures.items():
            try:
                await future
                results[prefix] = True
            except Exception as e:
                results[prefix] = e
        return results

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