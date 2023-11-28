- 参考项目：https://github.com/tatsu-lab/stanford_alpaca     https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M
```bash
conda create --name alpaca python=3.9
conda activate alpaca

# git clone https://github.com/tatsu-lab/stanford_alpaca.git

cd stanford_alpaca

# 安装所需环境，这里相较于clone的内容有改动openai==0.26.0
pip install -r requirements.txt

# 指定环境变量OPEN_API_KEY&OPEAI_API_BASE
export OPENAI_API_KEY=sk-BWFQ0IcU7IYZLXqZ8cAc3f6928Ea41438c228c5c21E847A3 
export OPENAI_API_BASE=https://api.aiguoguo199.com/v1

# 使用默认种子任务集

# en
python -m generate_instruction generate_instruction_following_data --language en --output_dir ./new_tasks/ --model_name gpt-3.5-turbo-instruct --num_instructions_to_generate 10 --num_prompt_instructions 3 --request_batch_size 2 --num_cpus 8
# cn
python -m generate_instruction generate_instruction_following_data --language cn --output_dir ./new_tasks/ --model_name gpt-3.5-turbo-instruct --num_instructions_to_generate 10 --num_prompt_instructions 3 --request_batch_size 2 --num_cpus 8


# language 指令语言(en/cn)
# output_dir 生成指令输出目录
# seed_tasks_path 种子指令目录
# num_instructions_to_generate 生成指令数量——只能保证多于该数字（分prompt batch）
# model_name 模型名称


# 最终输出类比如下示例
# 运行结果如下,生成指令集在./new_tasks目录下
Loaded 175 human-written seed instructions
prompt_batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:13<00:00, 13.66s/it]
 10%|██████████████████                                                                                                                                                                   | 1/10 [00:13<02:03, 13.70s/it]Request 1 took 13.66s, processing took 0.01s
Generated 9 instructions, kept 8 instructions
prompt_batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:12<00:00, 12.72s/it]
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                  | 9/10 [00:26<00:02,  2.57s/it]Request 2 took 12.72s, processing took 0.00s
Generated 10 instructions, kept 10 instructions
18it [00:26,  1.47s/it]
```