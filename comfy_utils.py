import aiohttp
import asyncio
import re
import difflib
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def is_ui_format_workflow(workflow: Dict) -> bool:
    """Check if a workflow JSON is in the UI format rather than API format.
    
    Args:
        workflow: The workflow JSON as a dictionary
        
    Returns:
        True if the workflow is in UI format, False if it's in API format
    """
    # UI format has these specific fields at the root level
    ui_specific_fields = {'links', 'version', 'nodes', 'groups'}
    return any(field in workflow for field in ui_specific_fields)

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

class NoSuchInputError(Exception):
    """Exception raised when CSV contains a column that does not exist in the workflow."""
    def __init__(self, node_title: str, column_name: str, suggestions: Dict[str, list[str]]):
        self.node_title = node_title
        self.column_name = column_name
        self.suggestions = suggestions
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"The field '{self.column_name}' does not exist in the node '{self.node_title}'\n"
        if self.suggestions:
            msg += "\nDid you mean:\n"
            for sugg in self.suggestions:
                msg += f"  {self.node_title}.{sugg}\n"
        return msg

async def check_port(port: int) -> Optional[int]:
    """Check if a port is running ComfyUI."""
    try:
        logger.info(f"Checking if port {port} is running ComfyUI...")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/system_stats", timeout=5) as response:
                if response.status == 200:
                    logger.info(f"Found ComfyUI server on port {port}")
                    return port
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.info(f"Port {port} is not running ComfyUI: {e}")
    return None

async def parse_port_from_line(line: str) -> Optional[int]:
    """Extract and validate port number from lsof output line.
    
    Example line: "Python    1234 user    4u   IPv4  0x1234567890abcdef      0t0    TCP *:8188 (LISTEN)"
    """
    # Match port number after the last colon in the line
    match = re.search(r':(\d+)\s*\(LISTEN\)$', line)
    if not match:
        logger.info(f"No port number found in line: {line}")
        return None
        
    try:
        return int(match.group(1))
    except ValueError as e:
        logger.info(f"Failed to parse port from match {match.group(1)}: {e}")
        return None

async def find_comfy_port() -> Optional[int]:
    """Find the port where ComfyUI is running using lsof on macOS."""
    logger.info("Starting port search...")
    cmd = ["lsof", "-i", "-P", "-n"]
    
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
        
    # Parse the output to find listening ports
    for line in stdout.decode().splitlines():
        logger.debug(f"Processing line: {line}")
        # Look for Python processes that are listening
        if ("Python" in line or "python" in line) and "LISTEN" in line:
            port = await parse_port_from_line(line)
            if port:
                if await check_port(port):
                    return port
    
    logger.error("No ComfyUI server found in listening ports")
    return None

def update_workflow(workflow: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
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
                        logger.debug(f"Updated node {node_id} ({node_title}) field {field_name} with value from column {column_name}")
                    else:
                        # The node does not have this input field, so raise a helpful error
                        # Find suggestions for similar input fields in the node
                        if "inputs" in node:
                            input_fields = list(node["inputs"].keys())
                            suggestions = difflib.get_close_matches(field_name, input_fields, n=3, cutoff=0.6)
                        else:
                            input_fields = []
                            suggestions = []
                        raise NoSuchInputError(node_title, column_name, suggestions)
    
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