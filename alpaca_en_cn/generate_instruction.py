"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire

# tokenizer for chinese instruction
from gensim.summarization import bm25
from  transformers import AutoTokenizer
checkpoint = "bigscience/bloomz-7b1"
tokenizer_cn = AutoTokenizer.from_pretrained(checkpoint)
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

def encode_prompt(prompt_instructions,language='en'):
    """Encode multiple prompt instructions into a single string."""
    
    prompt_file = "./prompt.txt" if language == 'en' else "./prompt_cn.txt"
    prompt = open(prompt_file).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        if language=='en':
            input = "<noinput>" if input.lower() == "" else input
            prompt += f"###\n"
            prompt += f"{idx + 1}. Instruction: {instruction}\n"
            prompt += f"{idx + 1}. Input:\n{input}\n"
            prompt += f"{idx + 1}. Output:\n{output}\n"
        elif language=='cn':
            input = "<无输入>" if input.lower() == "" else input
            prompt += f"###\n"
            prompt += f"{idx + 1}. 指令: {instruction}\n"
            prompt += f"{idx + 1}. 输入:\n{input}\n"
            prompt += f"{idx + 1}. 输出:\n{output}\n"
    if language=='en':
        prompt += f"###\n"
        prompt += f"{idx + 2}. Instruction:"
    elif language=='cn':
        prompt += f"###\n"
        prompt += f"{idx + 2}. 指令:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response, language='en'):
    if response is None:
        return []
    try: #for gpt-3.5-turbo
        raw_instructions = response["message"]["content"]
    except:
        try:
            raw_instructions = response["text"]  #for text-davinci-003
        except:
            print("ERROR parse!")
    
    if language == 'en':
        if 'Instruction:' not in raw_instructions[0: 10]:
            raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + raw_instructions
    elif language == 'cn':
        if '指令:' not in raw_instructions[0: 10] and '指令：' not in raw_instructions[0: 10]:
             raw_instructions = f"{num_prompt_instructions+1}. 指令:" + raw_instructions
    
    raw_instructions = re.split("###", raw_instructions)
    instructions = []

    # process_gpt3_response_en
    if language == 'en':
        for idx, inst in enumerate(raw_instructions):
            # if the decoding stops due to length, the last example is likely truncated so we discard it
            if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
                continue
            idx += num_prompt_instructions + 1
            splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
            if len(splitted_data) != 7:
                continue
            else:
                inst = splitted_data[2].strip()
                input = splitted_data[4].strip()
                input = "" if input.lower() == "<noinput>" else input
                output = splitted_data[6].strip()
            # filter out too short or too long instructions
            if len(inst.split()) <= 3 or len(inst.split()) > 150:
                continue
            # filter based on keywords that are not suitable for language models.
            blacklist = [
                "image",
                "images",
                "graph",
                "graphs",
                "picture",
                "pictures",
                "file",
                "files",
                "map",
                "maps",
                "draw",
                "plot",
                "go to",
                "video",
                "audio",
                "music",
                "flowchart",
                "diagram",
            ]
            blacklist += []
            if any(find_word_in_string(word, inst, language) for word in blacklist):
                continue
            # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
            # And it's a bit comfusing whether the model need to write a program or directly output the result.
            # Here we filter them out.
            # Note this is not a comprehensive filtering for all programming instructions.
            if inst.startswith("Write a program"):
                continue
            # filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # filter those starting with non-english character
            if not inst[0].isascii():
                continue
            instructions.append({"instruction": inst, "input": input, "output": output})

    # process_gpt3_response_cn
    elif language == 'cn':
        blacklist = ["图像", "图片", "照片", "文件", "图表", "图层", "曲线图", "折线图", "直线图", "柱形图", "饼状图", "链接", "http",'OpenAI', 'chatgpt', 'gpt-3', 'gpt-3.5', 'gpt-4']
        replace_empty_list = ['要求GPT模型能够', '要求GPT能够', '要求GPT模型', '让GPT模型', '使用GPT模型', '请向GPT模型', 'GPT模型应', 'GPT模型应该', '请求GPT模型', '需要GPT模型回答', '请GPT模型'
                          , '请让GPT模型', '训练GPT模型', 'GPT模型需要', '要求GPT', '让GPT', '使用GPT', '请向GPT', 'GPT应', 'GPT应该', '请求GPT', '需要GPT回答', '请GPT', '请让GPT'
                          , '训练GPT', 'GPT需要', '希望GPT模型能够', '希望GPT能够', '以便GPT模型能够', '以便GPT能够', '使得GPT模型能够', '使得GPT能够', '使GPT模型能够', '使GPT能够'
                          , '由GPT模型', '使GPT模型']
        for idx, inst in enumerate(raw_instructions):
            # if the decoding stops due to length, the last example is likely truncated so we discard it
            if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
                continue
            # filter based on keywords that are not suitable for language models.
            if any(find_word_in_string(word, inst, language) for word in blacklist):
                continue
            intruction_pattern = re.compile(r"(?<=(?:" + '|'.join(['指令:', '指令：']) + "))[\s\S]*?(?=" + '|'.join(['输入:', '输入：']) + ")")
            input_pattern = re.compile(r"(?<=(?:" + '|'.join(['输入:', '输入：']) + "))[\s\S]*?(?=" + '|'.join(['输出:', '输出：']) + ")")
            output_pattern = re.compile(r"(?<=(?:" + '|'.join(['输出:', '输出：']) + "))[\s\S]*?(?=$)")
            intruction_match = intruction_pattern.search(inst)
            input_match = input_pattern.search(inst)
            output_match = output_pattern.search(inst)
            if intruction_match and input_match and output_match:
                inst = re.sub(r'\d+\.$', '', intruction_match.group().strip()).strip('\n')
                input = re.sub(r'\d+\.$', '', input_match.group().strip()).strip('\n')
                input = "" if "无输入" in input else input
                output = output_match.group().strip().strip('\n')
                if '指令:' in output and '输入:' in output and '输出:' in output: # 返回若没有以###号区分，取第一条数据
                    output_pattern_new = re.compile(r"(?<=(?:" + "))[\s\S]*?(?=" + '|'.join(['指令:', '指令：']) + ")")
                    output_match_new = output_pattern_new.search(output)
                    if output_match_new:
                        output = re.sub(r'\d+\.$', '', output_match_new.group().strip()).strip('\n')
                # 去掉不合理的instruction
                if len(inst) <= 3:
                    continue
                    
                for item in replace_empty_list:
                    inst = inst.replace(item, "") 
                
                if "GPT" in inst or 'GPT' in input:
                    continue
                    
                if len(input) == 0:  # input无输入
                    instructions.append({"instruction": inst, "input": input, "output": output})
                else:
                    if '示例' in inst or '例子' in inst:  # inst里给例子
                        if len(inst) < 150:
                            instructions.append({"instruction": inst, "input": input, "output": output})
                    else:  # 没给例子
                        if len(inst) < 100:
                            instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s, language):
    if language == "en":
        return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)
    elif language == "cn":
        return w in s
    else:
        raise ValueError("Unsupported language: {0}".format(language))


def generate_instruction_following_data(
    language='en',
    output_dir="./",
    seed_tasks_path="",
    num_instructions_to_generate=100,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    if not seed_tasks_path:
        seed_tasks_path="./seed_tasks.jsonl" if language == 'en' else "./zh_seed_tasks.json"
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]

    if language == 'en':
        # similarities = {}
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    elif language =='cn':
        all_instruction_tokens = [tokenizer_cn.tokenize(inst) for inst in all_instructions]
        bm25Model = bm25.BM25(all_instruction_tokens)

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions, language)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072 if language == 'en' else 1024,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result, language)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            if language == 'en':
                new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
                with Pool(num_cpus) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instruction_tokens),
                        all_instruction_tokens,
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                if max(rouge_scores) > 0.7:
                    continue
                else:
                    keep += 1

            elif language == 'cn':
                new_instruction_tokens = tokenizer_cn.tokenize(instruction_data_entry["instruction"])
                rouge_scores = bm25Model.get_scores(new_instruction_tokens)
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                if max(rouge_scores) >18:
                    continue
                else:
                    keep += 1
            
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
