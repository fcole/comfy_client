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
import comfy_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            result = await future
            results[prefix] = result
        except Exception as e:
            results[prefix] = e
    return results

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
    parser.add_argument(
        '--log-file',
        type=Path,
        default=Path('comfy_batch_log.txt'),
        help='Path to the log file for completed workflow hashes'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port number of the ComfyUI server (optional)'
    )
    args = parser.parse_args()

    # Expand user paths
    args.workflow = args.workflow.expanduser()
    args.prompts = args.prompts.expanduser()
    args.log_file = args.log_file.expanduser()

    if not args.workflow.exists() or not args.prompts.exists():
        print("Error: One or both input files do not exist")
        print(f"Workflow file: {args.workflow}")
        print(f"Prompts file: {args.prompts}")
        sys.exit(1)

    # Initialize client
    client = await ComfyClient.create(port=args.port, log_file=args.log_file)

    # Load workflow
    try:
        with open(args.workflow) as f:
            workflow = json.load(f)
        logger.info("Successfully loaded workflow file")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid workflow JSON file: {e}")
        sys.exit(1)

    # Check if workflow is in UI format
    if comfy_utils.is_ui_format_workflow(workflow):
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
        futures = {}
        skipped_count = 0
        for row in rows:
            try:
                updated_workflow = comfy_utils.update_workflow(workflow, row)
            except (comfy_utils.UnusedColumnsError, comfy_utils.NoSuchInputError) as e:
                logger.error(f"Error processing row {row}: {e}")
                continue

            workflow_hash = comfy_utils.get_workflow_hash(updated_workflow)
            future = client.queue_prompt(updated_workflow)
            # if future is already completed, skip it
            if future.done() and future.result() == "skipped":
                skipped_count += 1
                continue
            futures[workflow_hash] = future

        queued_count = len(futures)
        logger.info(f"Queued {queued_count} new workflows.")

        if not futures:
            logger.info("No new workflows to run.")
            return

        # Wait for all futures to complete
        results = await wait_for_all(futures)
        
        # Log results
        success_count = 0
        failure_count = 0
        for id, result in results.items():
            if isinstance(result, Exception):
                logger.error(f"Failed to generate image with id {id[:8]}: {result}")
                failure_count += 1
            else:
                logger.info(f"Successfully generated image with id: {id[:8]}")
                success_count += 1

        logger.info(f"Completed processing {len(results)} workflows.")
        if failure_count > 0 or skipped_count > 0:
            summary = []
            if success_count > 0:
                summary.append(f"{success_count} succeeded")
            if failure_count > 0:
                summary.append(f"{failure_count} failed")
            if skipped_count > 0:
                summary.append(f"{skipped_count} skipped")
            logger.info(f"Summary: {', '.join(summary)}")
        else:
            logger.info(f"Summary: All {len(results)} rows completed successfully")

    finally:
        # Clean up client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 