#!/usr/bin/env python3

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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    client = ComfyClient(port=args.port)
    try:
        await client.initialize()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

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
        # Start client tasks
        await client.start()

        # Queue all prompts from the rows
        futures = client.queue_sheet(workflow, rows)
        
        # Wait for all futures to complete
        results = await client.wait_for_all(futures)
        
        # Log results
        for prefix, result in results.items():
            if isinstance(result, Exception):
                logger.error(f"Failed to generate image with prefix {prefix}: {result}")
            else:
                logger.info(f"Successfully generated image with prefix: {prefix}")

        logger.info(f"Completed processing {len(results)} rows")
        logger.info("Script finished successfully")

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 