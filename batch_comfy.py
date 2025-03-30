#!/usr/bin/env python3

import json
import csv
import time
import websocket
import requests
import sys
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.ws = None
        self.current_prompt_id = None
        self.processing_complete = False
        self.ws_thread = None
        self.has_queued_prompt = False  # Track if we've queued a prompt

    def connect(self) -> bool:
        """Test if the ComfyUI server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/system_stats")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def connect_websocket(self):
        """Establish WebSocket connection for progress updates."""
        websocket.enableTrace(False)  # Disable trace logging
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True  # Thread will exit when main program exits
        self.ws_thread.start()

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            if "type" in data:
                # Skip monitoring messages
                if data["type"] == "crystools.monitor":
                    return
                if data["type"] == "progress":
                    progress = data["data"]["value"]
                    logger.info(f"Progress: {progress}")
                elif data["type"] == "executed" and data["data"]["node"] is None:
                    self.processing_complete = True
                    logger.info("Image generation complete!")
                elif data["type"] == "status":
                    # Log status messages for debugging
                    if "data" in data and "status" in data["data"]:
                        status = data["data"]["status"]
                        logger.debug(f"Status message: {data}")  # Log full message
                        if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                            logger.debug(f"Queue empty status received. Has queued prompt: {self.has_queued_prompt}")
                            if self.has_queued_prompt:  # Only consider queue empty as completion if we've queued a prompt
                                self.processing_complete = True
                                logger.info("Queue empty - generation complete!")
                else:
                    logger.debug(f"Unknown message type: {data['type']}")
                    if "data" in data:
                        logger.debug(f"Message data: {data['data']}")
        except json.JSONDecodeError:
            pass

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")

    def queue_prompt(self, workflow: Dict[str, Any]) -> Optional[str]:
        """Submit a workflow to the ComfyUI server."""
        try:
            logger.info("Sending workflow to ComfyUI...")
            response = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow}
            )
            if response.status_code == 200:
                prompt_id = response.json()["prompt_id"]
                logger.info(f"Successfully queued prompt with ID: {prompt_id}")
                self.has_queued_prompt = True  # Mark that we've queued a prompt
                return prompt_id
            else:
                logger.error(f"Server returned status code: {response.status_code}")
                logger.error(f"Server response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to queue prompt: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Server response: {e.response.text}")
            return None

    def wait_for_completion(self, timeout: int = 300) -> bool:
        """Wait for the current prompt to complete."""
        start_time = time.time()
        logger.info("Waiting for image generation to complete...")
        while not self.processing_complete:
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for image generation")
                return False
            time.sleep(0.1)
        logger.info("Image generation completed successfully")
        return True

    def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            self.ws.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=1.0)

def find_comfy_port() -> Optional[int]:
    """Find the port where ComfyUI is running using lsof on macOS."""
    try:
        logger.info("Starting port search...")
        # Run lsof to find Python processes listening on ports
        cmd = ["lsof", "-i", "-P", "-n"]
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"lsof command failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return None
            
        logger.info(f"lsof output: {result.stdout}")
        
        # Parse the output to find listening ports
        for line in result.stdout.splitlines():
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
                            response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=5)
                            logger.info(f"Response status code: {response.status_code}")
                            if response.status_code == 200:
                                logger.info(f"Found ComfyUI server on port {port}")
                                return port
                        except requests.exceptions.ConnectionError as e:
                            logger.info(f"Port {port} is not running ComfyUI: {e}")
                            continue
                        except requests.exceptions.Timeout as e:
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

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_comfy.py <workflow.json> <prompts.csv>")
        sys.exit(1)

    workflow_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    if not workflow_path.exists() or not csv_path.exists():
        print("Error: One or both input files do not exist")
        sys.exit(1)

    # Find ComfyUI port
    port = find_comfy_port()
    if not port:
        print("Error: Could not find running ComfyUI server")
        sys.exit(1)

    # Initialize client
    client = ComfyClient(port=port)
    if not client.connect():
        print("Error: Could not connect to ComfyUI server")
        sys.exit(1)

    # Load workflow
    try:
        with open(workflow_path) as f:
            workflow = json.load(f)
        logger.info("Successfully loaded workflow file")
    except json.JSONDecodeError:
        print("Error: Invalid workflow JSON file")
        sys.exit(1)

    try:
        # Connect WebSocket
        logger.info("Connecting to WebSocket...")
        client.connect_websocket()

        # Process CSV
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                logger.info(f"Processing row {row_count}: {row}")
                
                # Update workflow with values from the row
                updated_workflow = update_workflow(workflow, row)

                # Reset completion flag
                client.processing_complete = False

                # Submit workflow
                prompt_id = client.queue_prompt(updated_workflow)
                if not prompt_id:
                    logger.error("Failed to submit workflow")
                    continue

                # Wait for completion
                if not client.wait_for_completion():
                    logger.error("Workflow execution failed or timed out")
                    continue

                logger.info(f"Successfully generated image with prefix: {row['output.filename_prefix']}")

            logger.info(f"Completed processing {row_count} rows")
            logger.info("Script finished successfully")

    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up WebSocket connection
        client.close()

if __name__ == "__main__":
    main() 