#!/usr/bin/env python3

import json
import csv
import time
import asyncio
import aiohttp
import websockets
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out websockets debug messages
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('websockets.client').setLevel(logging.WARNING)

@dataclass
class QueuedPrompt:
    workflow: Dict[str, Any]
    future: asyncio.Future
    prompt_id: Optional[str] = None
    queued_at: datetime = None

class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.ws = None
        self.prompt_queue = asyncio.Queue()
        self.current_prompt: Optional[QueuedPrompt] = None
        self.prompt_futures: Dict[str, asyncio.Future] = {}  # Map prompt_id to Future
        self.should_stop = False

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
            async with websockets.connect(self.ws_url) as websocket:
                self.ws = websocket
                while not self.should_stop:
                    try:
                        message = await websocket.recv()
                        await self._handle_message(message)
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed")
                        break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            if "type" in data:
                # Skip monitoring messages completely
                if data["type"] == "crystools.monitor":
                    return
                    
                if data["type"] == "progress":
                    progress = data["data"]["value"]
                    logger.info(f"Progress: {progress}")
                elif data["type"] == "executed" and data["data"]["node"] is None:
                    if self.current_prompt:
                        logger.info("Image generation complete!")
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
                        logger.debug(f"Status message: {data}")  # Log full message
                        if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                            logger.debug("Queue empty status received")
                            if self.current_prompt:
                                # Resolve the future for the current prompt
                                future = self.prompt_futures.get(self.current_prompt.prompt_id)
                                if future:
                                    future.set_result(True)
                                    del self.prompt_futures[self.current_prompt.prompt_id]
                                self.current_prompt = None
                else:
                    logger.debug(f"Unknown message type: {data['type']}")
                    if "data" in data:
                        logger.debug(f"Message data: {data['data']}")
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
                    logger.info("Sending workflow to ComfyUI...")
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/prompt",
                            json={"prompt": queued_prompt.workflow}
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                prompt_id = result["prompt_id"]
                                logger.info(f"Successfully queued prompt with ID: {prompt_id}")
                                queued_prompt.prompt_id = prompt_id
                                queued_prompt.queued_at = datetime.now()
                                self.current_prompt = queued_prompt
                                self.prompt_futures[prompt_id] = queued_prompt.future
                            else:
                                error_text = await response.text()
                                logger.error(f"Server returned status code: {response.status}")
                                logger.error(f"Server response: {error_text}")
                                queued_prompt.future.set_exception(Exception(f"Failed to queue prompt: {error_text}"))
                except Exception as e:
                    logger.error(f"Failed to queue prompt: {e}")
                    queued_prompt.future.set_exception(e)

            except Exception as e:
                logger.error(f"Error in queue processor: {e}")

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

async def find_comfy_port() -> Optional[int]:
    """Find the port where ComfyUI is running using lsof on macOS."""
    try:
        logger.info("Starting port search...")
        # Run lsof to find Python processes listening on ports
        cmd = ["lsof", "-i", "-P", "-n"]
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            logger.error(f"lsof command failed with return code {result.returncode}")
            logger.error(f"stderr: {stderr.decode()}")
            return None
            
        logger.info(f"lsof output: {stdout.decode()}")
        
        # Parse the output to find listening ports
        for line in stdout.decode().splitlines():
            logger.info(f"Processing line: {line}")
            # Look for Python processes that are listening
            if ("Python" in line or "python" in line) and "LISTEN" in line:
                # Extract port number from the line
                parts = line.split()
                if len(parts) >= 9:
                    port_str = parts[8].split(':')[-1]
                    try:
                        port = int(port_str)
                        logger.info(f"Found potential port: {port}")
                        # Check if this port is running ComfyUI
                        try:
                            logger.info(f"Checking if port {port} is running ComfyUI...")
                            async with aiohttp.ClientSession() as session:
                                async with session.get(f"http://127.0.0.1:{port}/system_stats", timeout=5) as response:
                                    logger.info(f"Response status code: {response.status}")
                                    if response.status == 200:
                                        logger.info(f"Found ComfyUI server on port {port}")
                                        return port
                        except aiohttp.ClientError as e:
                            logger.info(f"Port {port} is not running ComfyUI: {e}")
                            continue
                        except asyncio.TimeoutError as e:
                            logger.info(f"Timeout checking port {port}: {e}")
                            continue
                    except ValueError as e:
                        logger.info(f"Failed to parse port from {port_str}: {e}")
                        continue
                else:
                    logger.info(f"Line does not have enough parts: {line}")
        
        logger.error("No ComfyUI server found in listening ports")
        return None
        
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to execute lsof command: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while finding port: {e}")
        return None

def update_workflow(workflow: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
    """Update the workflow with values from the CSV row."""
    # Create a deep copy of the workflow to avoid modifying the original
    updated_workflow = json.loads(json.dumps(workflow))
    
    # Map CSV column names to node inputs
    for node_id, node in updated_workflow.items():
        if "_meta" in node and "title" in node["_meta"]:
            node_title = node["_meta"]["title"].lower().replace(" ", "_")
            # Look for a matching CSV column
            for column_name, value in row.items():
                # Split column name into node title and field name
                parts = column_name.split(".")
                if len(parts) == 2 and parts[0] == node_title:
                    field_name = parts[1]
                    if "inputs" in node and field_name in node["inputs"]:
                        node["inputs"][field_name] = value
                        logger.debug(f"Updated node {node_id} ({node_title}) field {field_name} with value from column {column_name}")
    
    return updated_workflow

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run ComfyUI workflows in batch mode')
    parser.add_argument(
        '--workflow',
        type=Path,
        default=Path('workflow/text2image.json'),
        help='Path to the workflow JSON file'
    )
    parser.add_argument(
        '--prompts',
        type=Path,
        default=Path('testdata/text2image.csv'),
        help='Path to the CSV file containing prompts'
    )
    args = parser.parse_args()

    if not args.workflow.exists() or not args.prompts.exists():
        print("Error: One or both input files do not exist")
        print(f"Workflow file: {args.workflow}")
        print(f"Prompts file: {args.prompts}")
        sys.exit(1)

    # Find ComfyUI port
    port = await find_comfy_port()
    if not port:
        print("Error: Could not find running ComfyUI server")
        sys.exit(1)

    # Initialize client
    client = ComfyClient(port=port)
    if not await client.connect():
        print("Error: Could not connect to ComfyUI server")
        sys.exit(1)

    # Load workflow
    try:
        with open(args.workflow) as f:
            workflow = json.load(f)
        logger.info("Successfully loaded workflow file")
    except json.JSONDecodeError:
        print("Error: Invalid workflow JSON file")
        sys.exit(1)

    try:
        # Start client tasks
        await client.start()

        # Process CSV
        with open(args.prompts) as f:
            reader = csv.DictReader(f)
            row_count = 0
            futures = []
            
            for row in reader:
                row_count += 1
                logger.info(f"Processing row {row_count}: {row}")
                
                # Update workflow with values from the row
                updated_workflow = update_workflow(workflow, row)

                # Queue the prompt and store the future
                future = client.queue_prompt(updated_workflow)
                futures.append((row['output.filename_prefix'], future))

            # Wait for all futures to complete
            try:
                # Gather all futures and their prefixes
                results = await asyncio.gather(
                    *[future for _, future in futures],
                    return_exceptions=True
                )
                # Process results and log success/failure
                for (prefix, _), result in zip(futures, results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to generate image with prefix {prefix}: {result}")
                    else:
                        logger.info(f"Successfully generated image with prefix: {prefix}")
            except Exception as e:
                logger.error(f"Error waiting for futures to complete: {e}")

            logger.info(f"Completed processing {row_count} rows")
            logger.info("Script finished successfully")

    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 