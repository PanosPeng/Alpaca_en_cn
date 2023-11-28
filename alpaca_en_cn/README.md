- Reference
- https://github.com/tatsu-lab/stanford_alpaca
- https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M
```bash
# Create a Conda environment named 'alpaca' with Python 3.9
conda create --name alpaca python=3.9
conda activate alpaca

# Install the required environment; note that there's a modification in openai==0.26.0 compared to the cloned content
pip install -r requirements.txt

# Specify environment variables OPEN_API_KEY & OPENAI_API_BASE
export OPENAI_API_KEY=sk-BWFQ0IcU7IYZLXqZ8cAc3f6928Ea41438c228c5c21E847A3
export OPENAI_API_BASE=https://api.aiguoguo199.com/v1

# Use the default seed task set

# English
python -m generate_instruction generate_instruction_following_data --language en --output_dir ./new_tasks/ --model_name gpt-3.5-turbo-instruct --num_instructions_to_generate 10 --num_prompt_instructions 3 --request_batch_size 2 --num_cpus 8
# Chinese
python -m generate_instruction generate_instruction_following_data --language cn --output_dir ./new_tasks/ --model_name gpt-3.5-turbo-instruct --num_instructions_to_generate 10 --num_prompt_instructions 3 --request_batch_size 2 --num_cpus 8


# Parameters explanation:
# language: Instruction language (en/cn)
# output_dir: Output directory for generated instructions
# seed_tasks_path: Directory for seed instructions
# num_instructions_to_generate: Number of instructions to generate — ensures at least this number (per prompt batch)
# model_name: Model name


# Example of final output:
# The generated instructions are in the ./new_tasks directory
Loaded 175 human-written seed instructions
prompt_batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:13<00:00, 13.66s/it]
 10%|██████████████████                                                                                                                                                                   | 1/10 [00:13<02:03, 13.70s/it]Request 1 took 13.66s, processing took 0.01s
Generated 9 instructions, kept 8 instructions
prompt_batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:12<00:00, 12.72s/it]
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                  | 9/10 [00:26<00:02,  2.57s/it]Request 2 took 12.72s, processing took 0.00s
Generated 10 instructions, kept 10 instructions
18it [00:26,  1.47s/it]
Chinese translated to English

```
