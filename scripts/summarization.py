import argparse
import vllm
import pandas as pd
import csv
import sys
import os
import random
import numpy as np
import torch
import re


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


system_prompt = "You are an expert at summarizing interviews."
# If modifying the user_prompt, additionally modify parse_summary to extract the themes correctly
user_prompt = """
Task:
1. Summarize the interview in 5–7 sentences, focusing on: 
   "How does this person find meaning in life?"
2. Then provide the following section:
   - Core Values

Output Format:
Summary:
...

Core Values:
- ...
"""

model_patterns = {
    'Qwen/Qwen2.5-7B-Instruct': r"- ([^:\n]+)",
    'meta-llama/Llama-3.2-3B-Instruct': r"\s\*\*(.*?)(?=\*\*:|:\*\*)",
    'meta-llama/Llama-3.1-8B-Instruct': r"(?:- |\* |• |\s\*\*)(.*?)(?=:\s|\*\*:)",
}

def get_responses(text):
    text = ' '.join([x.replace('RESPONDENT: ', '') for x in text.split('\n\n') if x.startswith('RESPONDENT: ')])
    return text.replace('\n', '').replace('\\', '').strip()

def clean_value(text):
    text = text.replace('*', '')
    text = text.lower().strip()
    themes = text.split(' and ')
    return themes

def parse_summary(text, pattern):
    main_text, core_themes = "", ""
    if "Summary:" in text:
        text = text.split("Summary:")[1].strip()

    if "Core Values:" in text:
        main_text = text.split("Core Values:")[0].strip()
        core_themes = text.split("Core Values:")[1].strip()
        matches = re.findall(pattern, core_themes)
        matches = [clean_value(x) for x in matches]
        core_themes = [x for y in matches for x in y]
    return main_text, core_themes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--prompt_style', type=str, default='baseline')
    parser.add_argument('--seed', type=int, default=738294)
    args = parser.parse_args()

    print("Running", args)
    directory = f'summaries/'
    os.makedirs(directory, exist_ok=True)
    output_file = f'{args.model_path.split("/")[1]}_{args.prompt_style}_{args.seed}.csv'
    if os.path.exists(os.path.join(directory, output_file)):
        sys.exit(f"File {output_file} already exists, exiting.")

    llm = vllm.LLM(model=args.model_path, seed=args.seed)
    seed_everything(args.seed)
    sampling_params = vllm.SamplingParams(temperature=.7, max_tokens=6000)

    # -----------------------
    # LOAD DATA AND DEMOGRAPHICS HERE
    #----------------------------------
    input_data = pd.DataFrame([]) # needs fields 'interview', 'Gender', and 'Race'
    # Edit the get_responses function if you need to extract interviewee responses from an interview transcript
    # Otherwise you can have get_responses return the original document

    all_data = {}
    for index, row in input_data.iterrows():
        all_data[index] = {
            'gender': row['Gender'],
            'race': row['Race'],
            'interview': row['interview'],
            'interviewee_responses': get_responses(row['interview'])
        }

    output_data = [["interview_id", "interview", "interviewee_responses", "summary_full", "summary_main", "summary_themes", 'gender', 'race']]
    prompts = []
    for interview_id, interview_dat in all_data.items():
        prompt = f"Interview transcript excerpt:\n{interview_dat['interview']}\n\n"
        if args.prompt_style == 'with_demo':
            prompt += f"The interviewee is a {interview_dat['race']} {interview_dat['gender']}.\n\n"
        prompt += user_prompt
        if args.model_path.startswith('Qwen'):
            prompt += " Encapsulate your response in <response></response> tags."
        prompt = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": prompt},
        ]
        prompts.append(prompt)

    outputs = llm.chat(prompts, sampling_params=sampling_params)

    idx = 0
    for interview_id, interview_dat in all_data.items():
        output = outputs[idx].outputs[0].text
        if f"<response>" in output and f"</response>" in output:
            output = output.split(f"<response>")[1].split(f"</response>")[0]
        summary_main, summary_themes = parse_summary(output, model_patterns[args.model_path])
        ouput_row = [interview_id, interview_dat['interview'], interview_dat['interviewee_responses'], output, summary_main, summary_themes, interview_dat['gender'], interview_dat['race']]
        output_data.append(ouput_row)
        idx += 1

    with open(os.path.join(directory, output_file), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(output_data)














