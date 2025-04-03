#!/usr/bin/env python3

from typing import Dict, Any
import json
import sys
import asyncio
import argparse
import csv
from pathlib import Path
import logging
from comfy_client import ComfyClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

async def wait_for_all(futures: Dict[str, asyncio.Future]) -> Dict[str, Any]:
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

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run ComfyUI workflows in batch mode')
    parser.add_argument(
        '--workflow',
        type=Path,
        default=Path('workflow/text2image_from_ui.json'),
        help='Path to the workflow JSON file'
    )
    parser.add_argument(
        '--prompts',
        type=Path,
        default=Path('testdata/text2image.csv'),
        help='Path to the CSV file containing prompts'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port number of the ComfyUI server (optional)'
    )
    args = parser.parse_args()

    if not args.workflow.exists() or not args.prompts.exists():
        print("Error: One or both input files do not exist")
        print(f"Workflow file: {args.workflow}")
        print(f"Prompts file: {args.prompts}")
        sys.exit(1)

    # Initialize client
    client = await ComfyClient.create(port=args.port)

    # Load workflow
    try:
        with open(args.workflow) as f:
            workflow = json.load(f)
        logger.info("Successfully loaded workflow file")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid workflow JSON file: {e}")
        sys.exit(1)

    # Check if workflow is in UI format
    if is_ui_format_workflow(workflow):
        print("Error: The workflow file appears to be in the UI format (saved using Workflow->Save As... or loaded from a .png file)."
        " Please export the workflow in API format using Workflow->Export (API) in the ComfyUI interface instead.")
        sys.exit(1)

    # Read CSV file
    try:
        with open(args.prompts) as f:
            reader = csv.DictReader(f)
            rows = list(reader)  # Convert to list for easier handling
        logger.info(f"Successfully loaded {len(rows)} rows from CSV file")
    except csv.Error as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    try:
        # Queue all prompts from the rows
        futures = client.queue_sheet(workflow, rows)
        
        # Wait for all futures to complete
        results = await wait_for_all(futures)
        
        # Log results
        success_count = 0
        failure_count = 0
        for prefix, result in results.items():
            if isinstance(result, Exception):
                logger.error(f"Failed to generate image with prefix {prefix}: {result}")
                failure_count += 1
            else:
                logger.info(f"Successfully generated image with prefix: {prefix}")
                success_count += 1

        logger.info(f"Completed processing {len(results)} rows")
        if failure_count > 0:
            logger.error(f"Summary: {failure_count} of {len(results)} rows failed")
        else:
            logger.info(f"Summary: All {len(results)} rows completed successfully")

    finally:
        # Clean up client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 