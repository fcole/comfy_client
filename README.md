ComfyUI Command Line Runner

1. Create a workflow and name the important nodes with descriptive titles. Export .json from ComfyUI in API format using Workflow -> Export (API)
2. Create a .csv with columns named <node title>.<parameter>. The rows of this table will replace the node parameter values in the workflow .json
3. Run `python batch_runner.py --workflow=my_workflow.json --prompts=my_prompts.csv`

Example input .csv:

```
i2i.seed,i2i.prompt,save_image.filename_prefix,
42,"A happy capybara swims in the amazon",capybara1,
777,"A happy capybara swims in the amazon",capybara2,
```
