from pathlib import Path
from enum import Enum
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional
from datasets import load_dataset, concatenate_datasets

import logging
import time
import re
from collections import defaultdict
from mathruler.grader import grade_answer
import base64
from io import BytesIO
from PIL import Image
import ast
import numpy as np
import itertools
from tqdm import tqdm

logger = logging.getLogger(__name__)

MATHVISION_TESTMINI_IDS = {'4', '5', '6', '7', '8', '10', '11', '16', '20', '23', '26', '27', '28', '29', '32', '33', '34', '35', '38', '39', '41', '43', '44', '45', '46', '49', '50', '52', '53', '55', '58', '59', '60', '61', '62', '63', '64', '65', '66', '71', '75', '78', '79', '80', '86', '89', '90', '91', '92', '93', '95', '99', '100', '104', '105', '107', '110', '114', '115', '117', '118', '120', '123', '124', '125', '126', '130', '131', '133', '149', '152', '157', '159', '162', '164', '167', '168', '173', '175', '176', '180', '181', '183', '187', '190', '193', '195', '201', '203', '206', '210', '211', '213', '214', '215', '216', '217', '218', '219', '222', '223', '224', '230', '231', '233', '234', '242', '246', '250', '253', '254', '255', '259', '261', '263', '269', '270', '272', '273', '277', '279', '284', '285', '286', '290', '291', '292', '293', '295', '296', '297', '300', '307', '318', '319', '325', '330', '333', '336', '351', '354', '357', '364', '366', '373', '386', '390', '403', '414', '415', '416', '417', '418', '423', '427', '437', '439', '441', '442', '455', '463', '467', '472', '473', '474', '514', '521', '522', '524', '525', '526', '534', '537', '544', '545', '549', '550', '556', '559', '568', '589', '608', '621', '628', '641', '648', '654', '662', '675', '701', '706', '730', '739', '742', '748', '761', '764', '766', '767', '771', '773', '785', '811', '812', '813', '819', '823', '850', '855', '861', '870', '873', '893', '913', '923', '934', '946', '951', '961', '965', '1008', '1011', '1039', '1050', '1060', '1064', '1101', '1113', '1137', '1167', '1168', '1174', '1203', '1211', '1215', '1222', '1226', '1246', '1250', '1255', '1284', '1301', '1335', '1336', '1343', '1355', '1388', '1389', '1394', '1399', '1414', '1426', '1451', '1512', '1546', '1547', '1559', '1564', '1634', '1668', '1680', '1696', '1709', '1727', '1781', '1787', '1813', '1814', '1831', '1861', '1864', '1889', '1893', '1915', '1920', '1969', '2280', '2288', '2293', '2316', '2398', '2443', '2474', '2513', '2555', '2565', '2588', '2589', '2634', '2636', '2648', '2659', '2665', '2673', '2688', '2741', '2743', '2757', '2773', '2774', '2786', '2955', '2984', '3033'}

# a simple timer class context manager
class Timer:
    def __init__(self, name: str):
        self.name = name
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"{self.name} took {elapsed_time:.2f} seconds.")
        self.elapsed_time = elapsed_time
        
        
def base64_to_pil(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding base64 string: {e}")
        return None
    

class DatasetType(Enum):
    MATHVISTA = "mathvista"
    MATHVERSE = "mathverse"
    MATHVISION = "mathvision"
    MMMU_VAL = "mmmu_val"
    R1_ONEVISION = "r1_onevision"
    MMMU_PRO = "mmmu_pro"

@dataclass
class DatasetConfig:
    name: str
    split: str
    image_field: str
    instruction_field: str
    response_field: str
    choices_field: Optional[str] = None
    options_field: Optional[str] = None
    question_for_eval_field: Optional[str] = None
    question_type_field: Optional[str] = None
    image_field_prefix: Optional[str] = None   # for multi-image support
    skip_multi_choice: Optional[bool] = False   # for mathvision testmini, we skip multi-choice question for pass@k evaluation

@dataclass
class ModelConfig:
    model_name: str
    max_new_tokens: int = 2048
    top_p: float = 0.001
    top_k: int = 1
    temperature: float = 0.01
    repetition_penalty: float = 1.0

# force_tp when do contrast with 72b - 7b
def model_name_to_tp(args, model_name):
    if args.force_tp is not None:
        return args.force_tp
    if '72b' in model_name.lower() or '32b' in model_name.lower():
        return 8
    elif '7b' in model_name.lower() or '3b' in model_name.lower():
        return 4
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    
def get_dataset_config(dataset_type: DatasetType, split='testmini', skip_multi_choice=False) -> DatasetConfig:
    configs = {
        DatasetType.MATHVISTA: DatasetConfig(
            name="AI4Math/MathVista",
            split=split,
            image_field="decoded_image",
            instruction_field="query",
            response_field="answer",
            choices_field="choices"
        ),
        DatasetType.MATHVERSE: DatasetConfig(
            name="AI4Math/MathVerse",
            split=split,
            image_field="image",
            instruction_field="query_cot",
            response_field="answer",
            # for gpt judge, you will need question_for_eval field
            question_for_eval_field="question_for_eval",
        ),
        DatasetType.MATHVISION: DatasetConfig(
            name="MathLLMs/MathVision",
            split=split,
            image_field="decoded_image",
            instruction_field="question",
            response_field="answer",
            options_field="options", 
            skip_multi_choice=skip_multi_choice,  # for mathvision testmini, we skip multi-choice question for pass@k evaluation
        ), 
        DatasetType.MMMU_VAL: DatasetConfig(
            name="lmms-lab/MMMU", 
            split="validation", 
            # Done -- MMMU needs multi-image support
            image_field='image_1', # disable image fields for MMMU, 
            image_field_prefix='image_',
            instruction_field="question",
            response_field="answer",
            question_type_field="question_type",
            options_field="options",
        ), 
        DatasetType.R1_ONEVISION: DatasetConfig(
            name="Fancy-MLLM/R1-Onevision-Bench",
            split="train",   # "train" is the only split available
            image_field="image",   # base64 image
            instruction_field="question",  # hint and choices included! 
            response_field="answer",
            choices_field="choices"
        ),
        DatasetType.MMMU_PRO: DatasetConfig(
            name="MMMU/MMMU_Pro", 
            split=split,  # we need overall -- the average of Stanard (10 options) & Vision, choose from `overall`, `standard4`, `standard10`
            image_field='image', 
            image_field_prefix='image_',
            instruction_field="question",  # only in stanard options. 
            options_field="options",
            response_field="answer",
        )
    }
    return configs[dataset_type]


def load_image_dataset(dataset_config: DatasetConfig) -> List[Dict]:
    """
    Load dataset from Hugging Face and extract image URLs and metadata
    """
    try:
        # MATHVERSE needs special treatment:
        if "MathVerse" in dataset_config.name:
            data = load_dataset(dataset_config.name, dataset_config.split, split=dataset_config.split)
        elif "MMMU_Pro" in dataset_config.name and dataset_config.split == "standard4":
            data = load_dataset(dataset_config.name, 'standard (4 options)', split='test')
        elif "MMMU_Pro" in dataset_config.name and dataset_config.split == "standard10":
            data = load_dataset(dataset_config.name, 'standard (10 options)', split='test')
        elif "MMMU_Pro" in dataset_config.name and dataset_config.split == "overall":
            data = concatenate_datasets([
                        load_dataset(dataset_config.name, 'standard (10 options)', split='test'),
                        load_dataset(dataset_config.name, 'vision', split='test')
                ])
        else:
            data = load_dataset(dataset_config.name, split=dataset_config.split)

        items = []
        for item in data:
            dataset_item = {
                'image_url': [item[dataset_config.image_field]],
                'instruction': item.get(dataset_config.instruction_field, ''),
                'response': item.get(dataset_config.response_field, ''),
            }
            if dataset_config.choices_field:  # only for mathvista & R1-Onevision
                dataset_item['choices'] = item.get(dataset_config.choices_field)
            if dataset_config.options_field:  # only for mathvision
                dataset_item['options'] = item.get(dataset_config.options_field, [])
            if "MathVista" in dataset_config.name:  # load metadata we need
                dataset_item['skills'] = item['metadata']['skills']
                dataset_item['question_type'] = item['question_type']
                dataset_item['answer_type'] = item['answer_type']
            if "MathVision" in dataset_config.name:
                dataset_item['id'] = item['id']
            if "MathVerse" in dataset_config.name:
                dataset_item['question_for_eval'] = item['question_for_eval']
            if "MMMU_Pro" in dataset_config.name:
                # MMMU Pro special treatment for overall setting
                dataset_item['options'] = item['options']
                if item.get(dataset_config.image_field) is None:
                    # 1. stanard options -- item['image'] is None, image_X is present
                    dataset_item['image_url'] = [item[k] for k in item.keys() if k.startswith(dataset_config.image_field_prefix) and item[k] is not None]
                    dataset_item['question_type'] = 'standard'
                else:
                    dataset_item['question_type'] = 'vision'
                    # be careful! remember to run `replace_images_tokens` when constructing the prompt
                # else:
                #     # 2. vision -- item['image'] is not None, image_X is None
                #     dataset_item['image_url'] = [item[dataset_config.image_field]]
                
            elif "MMMU" in dataset_config.name:
                dataset_item['question_type'] = item['question_type']
                dataset_item['options'] = item['options']
                all_image_keys = [k for k in item.keys() if k.startswith(dataset_config.image_field_prefix)]
                dataset_item['image_url'] = [item[k] for k in all_image_keys if item[k] is not None]
                # TODO: we need to search <image X> in the instruction field -- sometimes multi-image is not for question
                
            if "R1-Onevision" in dataset_config.name:
                for i in range(len(dataset_item['image_url'])):
                    dataset_item['image_url'][i] = base64_to_pil(dataset_item['image_url'][i])
                # other metadata
                dataset_item['level'] = item['level']
                dataset_item['category'] = item['category']
            
            # post-processing for mathvision pass@k, we skip multi-choice question
            if not dataset_config.skip_multi_choice or not (dataset_item.get('options') or dataset_item.get('choices')):
                items.append(dataset_item)
        
        return items
    except Exception as e:
        # logger.error(f"Failed to load dataset: {str(e)}")
        raise

# only used in generation 
def format_instruction(args, instruction: str, options: Optional[List[str]] = None) -> str:
    # Thinklite style -- no appended hint
    if args.prompt_type == 'thinklite':
        # logger.warning("Using thinklite prompt type...")
        if options and len(options) > 0:
            prompt_hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end."
            choice_list = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            return f"{prompt_hint}\nQuestion: {instruction}\nOptions:\n{choice_list}"
        else:
            prompt_hint = "Hint: Please answer the question requiring an answer."
            return f"{prompt_hint}\nQuestion: {instruction}"
        
    # visionr1 style -- no appended hint
    elif args.prompt_type == 'visionr1':
        # logger.warning("Using visionr1 prompt type...")
        if options and len(options) > 0:
            choice_list = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            return f"{instruction}\nOptions:\n{choice_list}"
        else:
            return instruction
    
    # Fallback: OpenVLThinker / VL-Rethinker style in mathvision
    elif options and len(options) > 0:
        # logger.warning("Using OpenVLThinker prompt type with options...")
        prompt_hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end."
        choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"{prompt_hint}\nQuestion: {instruction}\nChoices:\n{choice_list}"
    
    else:
        # logger.warning("Using OpenVLThinker prompt type without options...")
        prompt_hint = "Hint: Please answer the question requiring an answer."
        return f"{prompt_hint}\nQuestion: {instruction}"

# format from: https://github.com/si0wang/ThinkLite-VL/blob/8a74923/eval/model_mmmu_qwen.py
def format_instruction_mmmu(args, instruction: str, options, question_type) -> str:
    if question_type == 'multiple-choice':
        instruction += " Options: "
        options = eval(options)
        assert len(options) > 0, "Options should not be empty."
        for j, option in enumerate(options):
            instruction += f"\n{chr(ord('A')+j)}. {option}"
        instruction += f"\nAnswer with the option's letter from the given choices."
        
    else:
        instruction += f"\nAnswer the question using a single word or phrase."
        
    return instruction

# format from: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/infer/infer_transformers.py
def replace_images_tokens(input_string):
    image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", input_string)]
    input_string = re.sub(r"<image\s+\d+>", "<image>", input_string)
    return input_string, image_order

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    # question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    question = f"{question}\n{parsed_options}\nAnswer with the option letter from the given choices."  # use instruction_prompt from the model itself
    return question

def origin_mmmu_doc_to_visual(doc, image_order):
    visual = []
    for idx in image_order:
        visual.append(doc[f"image_{idx}"])
    return visual

def vision_mmmu_doc_to_visual(doc):
    return [doc["image"]]

def format_instruction_mmmu_pro(args, instruction: str, options: Optional[List[str]] = None, images=None, question_type=None) -> str:
    doc = {
        "question": instruction,
        "options": options,
    }
    if question_type == 'standard':
        question = construct_prompt(doc)
        prompt, image_order = replace_images_tokens(question)  
        # in prompt, all <image X> are replaced with [image], images are reorganized in the order of `image_order`
        new_images = [images[j - 1] for j in image_order]
    else:
        prompt = "Answer with the option letter from the given choices."
        new_images = images
    return prompt, new_images
    

# above from OpenVLThinker eval_qwen.py
# below from our implementation (OpenVLThinker style)
def process_dataset_to_message(dataset, args, skip_images=False):
    print(f"Processing {len(dataset)} examples of {args.dataset_type.value} to message format...")
    resp_messages = []
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing images"):
        
        if args.num_samples is not None and i >= args.num_samples:
            break
        
        # thinklite also only need formated instruction in mathvision
        if args.dataset_type == DatasetType.MATHVISION:
            instruction = format_instruction(args, item['instruction'], item.get('options'))
        elif args.dataset_type == DatasetType.MMMU_VAL:
            instruction = format_instruction_mmmu(args, item['instruction'], item.get('options'), item['question_type'])
        elif args.dataset_type == DatasetType.MMMU_PRO:
            instruction, item['image_url'] = format_instruction_mmmu_pro(args, item['instruction'], item.get('options'), 
                                                                         item['image_url'], item['question_type'])
        else:
            instruction = item['instruction']
        
        if args.prompt_type == 'bboxed':  # for CD setting this is needed -- most commonly used
            prompt_instruction = '\n\nYour final answer MUST BE put in \\boxed{}.'
            
        elif args.prompt_type == 'answer':  
            prompt_instruction = '\n\nYour final answer MUST BE put between <answer> </answer>.'
            
        elif args.prompt_type == 'r1onevision':
            prompt_instruction = '\n\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.'
            
        elif args.prompt_type in ('none', 'visionr1'):
            prompt_instruction = ''
            
        elif args.prompt_type == 'thinklite':
            prompt_instruction = r"\n\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
            
        elif args.prompt_type == 'vlrethinker':
            prompt_instruction = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        
        elif args.prompt_type == 'vlrethinker72b':
            prompt_instruction = """\n\nGuidelines: 
Please think step by step, and **regularly perform self-questioning, self-verification, self-correction to check your ongoing reasoning**, using connectives such as "Wait a moment", "Wait, does it seem right?", etc. Remember to put your final answer within \\boxed{}.
"""
        
        else:
            raise ValueError(f"Unknown prompt type: {args.prompt_type}")
        
        # multi-image support from: https://docs.vllm.ai/en/latest/getting_started/examples/vision_language_multi_image.html
        image_placeholder = [{"type": "image", "image": url} for url in item['image_url']]
        
        messages = [
            {
                "role": "user",
                "content": [
                    *image_placeholder,
                    {"type": "text", "text": instruction + prompt_instruction},
                ],
            }
        ]
        
        if args.min_pixels is not None or args.max_pixels is not None:
            assert args.min_pixels is not None and args.max_pixels is not None, \
                "Both min_pixels and max_pixels should be set to filter images."
            messages[0]['content'][0]['min_pixels'] = args.min_pixels
            messages[0]['content'][0]['max_pixels'] = args.max_pixels
        
        if skip_images:  # for debugging a text-only model
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": instruction + prompt_instruction}
            ]
        
        resp_messages.append(messages)
        
    return resp_messages


def process_response(response: str, choices: Optional[List[str]], options: Optional[List[str]] = None) -> str:
    if choices is not None:   # mathvista uses `choices` field
        try:
            response_index = choices.index(response)
            return ['A', 'B', 'C', 'D', 'E', 'F', 'G'][response_index]
        except (ValueError, IndexError):
            pass
    if options is not None and len(options) > 0:   # mathvision / MMMU_VAL uses `options` field
        try:
            response_index = options.index(response)
            return ['A', 'B', 'C', 'D', 'E'][response_index]
        except (ValueError, IndexError):
            pass
    # breakpoint()    
    # gt_response can be translated to one of the option letters
    return response

# 0504: advanced parser to handle nested \boxed{}
def extract_boxed_content(text):
    start = text.find(r'\boxed{')
    if start == -1:
        return None

    i = start + len(r'\boxed{')
    brace_level = 1
    content = ''

    while i < len(text) and brace_level > 0:
        char = text[i]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1

        if brace_level > 0:
            content += char
        i += 1

    return content if content else None

def extract_r1_onevision_answer(text):
    matches = re.findall(r"Answer:\s*([A-Z]|\w[\w\.\-\+]*)", text)
    if matches:
        return matches[-1].strip()  # Return the last 'Answer: X' match
    return None

def extract_answer(model_response, args):
    # thinklite prompt should be able to survive this since it's bascially a boxed 
    # try bboxed first
    answer_part = None
    answer_part = extract_boxed_content(model_response)
    
    if answer_part:
        return answer_part.replace("Final Answer:", "").strip()

    # if allow both eval, try to extract the answer from the model response after bboxed failed
    if args.allow_both_eval and "</answer>" in model_response:
        parts = model_response.split("<answer>")
        if len(parts) > 1:
            answer_part = parts[-1].split("</answer>")[0].strip()
            return answer_part.replace("Final Answer:", "").strip()
        
    answer_part = extract_r1_onevision_answer(model_response)

    # FIXED: 0504 we keep all {} -- re-run MMMU results is needed
    answer_part = answer_part.replace("Final Answer:", "").strip() if answer_part else None
    
    return answer_part

# added for pass@k evaluation -- unbiased estimation
def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def run_evaluate_math_pass_at_k(dataset, final_output, args, 
                                k=list(range(1, 129)), use_tqdm=True):
    # consider filter out multi-choice question!
    parsed_final_output = []
    k = list(sorted(k))
    if use_tqdm:
        pbar = tqdm(zip(dataset, final_output), desc="Evaluating pass@k", total=len(dataset))
    else:
        pbar = zip(dataset, final_output)
        
    for i, (input_example, model_output) in enumerate(pbar):
        # skip multi-choice question to avoid guessing
        if input_example.get('options') or input_example.get('choices'):
            result = {'filter': 'multi-choice'}
            parsed_final_output.append(result)
            continue
        
        model_response_group = model_output['generated_text_group']
        num_total_responses = len(model_response_group)
        assert num_total_responses <= k[-1], \
            f"Expect the number of model responses to be less than or equal to {k[-1]}, got {num_total_responses}"
        correct_flag_group = [0] * len(k)
        for j, model_response in enumerate(model_response_group):
            processed_response = process_response(  # convert free-from response to option letter if possible
                input_example['response'],
                input_example.get('choices'),
                input_example.get('options')
            )
            raw_processed_response = None
            if input_example.get('options') and len(processed_response) == 1 and "A" <= processed_response <= "Z":   # convert option letter to the actual answer
                raw_processed_response = input_example['options'][ord(processed_response) - 65]
                
            elif input_example.get('choices') and len(processed_response) == 1 and "A" <= processed_response <= "Z":   # convert option letter to the actual answer
                raw_processed_response = input_example['choices'][ord(processed_response) - 65]
            
            answer = extract_answer(model_response, args)
            if answer:
                if (processed_response.lower() == answer.lower() or 
                        grade_answer(processed_response, answer) or 
                        grade_answer(input_example['response'], answer) or # added to fix OpenVLThinker error
                        grade_answer(raw_processed_response, answer)):   # added to fix mathvision -- **OpenVLThinker evaluation made a mistake**
                    correct_flag_group[j] = 1
                    
        # complete sample-wise evaluation
        num_correct = sum(correct_flag_group)
        result = {}
        result['num_total_responses'] = num_total_responses
        result['num_correct'] = num_correct
        result['correct_flag_group'] = correct_flag_group
        parsed_final_output.append(result)
    
    total = np.array([item['num_total_responses'] for item in parsed_final_output if 'num_total_responses' in item])
    correct = np.array([item['num_correct'] for item in parsed_final_output if 'num_correct' in item])
    # get average pass@k
    avg_pass_at_k = {
        f"pass@{k_i}": estimate_pass_at_k(total, correct, k_i).mean() for k_i in k if (total >= k_i).all()
    }
    avg_pass_at_k['filtered'] = len([item for item in parsed_final_output if 'filter' in item])
    avg_pass_at_k['filtered_ratio'] = avg_pass_at_k['filtered'] / len(parsed_final_output)
    
    output_parsed_path = args.precomputed_json.replace('.jsonl', f'_parsed_pass_k.jsonl')
    output_acc_path = args.precomputed_json.replace('.jsonl', f'_acc_avg_pass_k.json')
    
    with open(output_parsed_path, 'w') as f:
        for item in parsed_final_output:
            f.write(json.dumps(item) + '\n')
            
    with open(output_acc_path, 'w') as f:
        f.write(json.dumps(avg_pass_at_k, indent=4))

    
# Error processing dataset config mathvision-test-answer: 'Namespace' object has no attribute 'dataset_type'
# copied `process_response` and `run_evaluate_math` here for the ease of debugging
def run_evaluate_math(dataset, final_output, args, run_time=None, tokenizer=None):
    # skip MMMU evaluation for now
    if args.dataset_type == DatasetType.MMMU_VAL:
        run_evaluate_mmmu(dataset, final_output, args, run_time)
        return
        # logger.warning("MMMU evaluation is not supported yet.")
        # return
    
    parsed_final_output = []
    # dataset specific counter...
    correct_counter = 0
    per_type_total_counter = defaultdict(int)
    per_type_correct_counter = defaultdict(int)
    error_counter = defaultdict(int)
    
    if not hasattr(args, 'allow_both_eval'):
        args.allow_both_eval = True   # we should relex the template matching 
    
    assert len(dataset) == len(final_output), \
        f"Expect dataset and final_output to have the same length, got {len(dataset)} and {len(final_output)}"
    
    for i, (input_example, model_output) in enumerate(zip(dataset, final_output)):
        model_response = model_output['generated_text']
        correct_flag = 0
        
        input_example_skills = []
        
        if args.dataset_type == DatasetType.MATHVISTA:
            input_example_skills = input_example['skills']
            for skill in input_example_skills:
                per_type_total_counter[skill] += 1
        
        elif args.dataset_type == DatasetType.MATHVISION:
            if input_example['id'] in MATHVISION_TESTMINI_IDS:
                input_example_skills = ['mathvision_testmini']
                per_type_total_counter['mathvision_testmini'] += 1
                
        elif args.dataset_type == DatasetType.R1_ONEVISION:
            input_example_skills = [input_example['level'], input_example['category']]
            for skill in input_example_skills:
                per_type_total_counter[skill] += 1
        
        if model_response:
            if input_example['response'] is None:   # this happens in one sample of r1-onevision 
                logger.warning(f"Response is None for question {i} -- {input_example['instruction']}")
                error_counter['response_none'] += 1
                continue
                
            processed_response = process_response(  # convert free-from response to option letter if possible
                input_example['response'],
                input_example.get('choices'),
                input_example.get('options')
            )
            raw_processed_response = None
            if input_example.get('options') and len(processed_response) == 1 and "A" <= processed_response <= "Z":   # convert option letter to the actual answer
                raw_processed_response = input_example['options'][ord(processed_response) - 65]
                
            elif input_example.get('choices') and len(processed_response) == 1 and "A" <= processed_response <= "Z":   # convert option letter to the actual answer
                raw_processed_response = input_example['choices'][ord(processed_response) - 65]
            
            answer = extract_answer(model_response, args)
            # logger.info(f"Extracted Answer: {answer}")
            if answer:
                if (processed_response.lower() == answer.lower() or 
                        grade_answer(processed_response, answer) or 
                        grade_answer(input_example['response'], answer) or # added to fix OpenVLThinker error
                        grade_answer(raw_processed_response, answer)):   # added to fix mathvision -- **OpenVLThinker evaluation made a mistake**
                    correct_counter += 1
                    correct_flag = 1
                    
                    for skill in input_example_skills:
                        per_type_correct_counter[skill] += 1
            else:
                error_counter['failed_to_extract'] += 1
                # logger.warning(f"Failed to extract answer for question {i} -- {processed_response} -- {model_response}")

        else:
            answer = "Failed to generate."
            error_counter['failed_to_generate'] += 1
            # logger.warning(f"Failed to generate answer for question {i} -- {processed_response} -- {model_response}")
            
        result = {
            'instruction': input_example['instruction'],
            'response': input_example['response'],  # ground truth
            'reasoning': model_response,
            'answer': answer,
            'correct': correct_flag
        }
        
        if tokenizer is not None:
            result['reasoning_token_length'] = len(tokenizer.encode(model_response))
            result['reasoning_str_length'] = len(model_response)
        
        parsed_final_output.append(result)
        
    # write to _acc json and _parsed jsonl file
    acc = {"Total Accuracy": correct_counter / len(dataset)}
    
    if per_type_correct_counter:
        for skill, correct_count in per_type_correct_counter.items():
            acc[f"Accuracy on {skill}"] = correct_count / per_type_total_counter[skill]
            
    if error_counter:
        for error_type, count in error_counter.items():
            acc[f"{error_type}"] = count / len(dataset)
    
    if run_time is not None:
        acc["run_time"] = run_time
        # get the sum of all reasoning_token_length and compute average decoding time
        if tokenizer is not None:
            acc["sum_reasoning_token_length"] = sum([item['reasoning_token_length'] for item in parsed_final_output])
            acc["avg_decoding_speed"] = acc["sum_reasoning_token_length"] / run_time
        
    print(acc)
    
    output_parsed_path = args.precomputed_json.replace('.jsonl', f'_parsed.jsonl')
    output_acc_path = args.precomputed_json.replace('.jsonl', f'_acc.json')
    
    with open(output_parsed_path, 'w') as f:
        for item in parsed_final_output:
            f.write(json.dumps(item) + '\n')
            
    with open(output_acc_path, 'w') as f:
        json.dump(acc, f, indent=2)

# MMMU evaluation adopted from ThinkLite-VL: https://github.com/si0wang/ThinkLite-VL/blob/main/eval/mmmu_score.py
def extract_answer_mmmu(model_response, args):
    # look for answer either in <answer> </answer> or \boxed{}
    predicted_answer = extract_answer(model_response, args)
    if predicted_answer is None:
        return model_response.split('\n')[-1].strip()
    return predicted_answer

# MMMU evaluation could also be from 
    
def run_evaluate_mmmu(dataset, final_output, args, run_time=None, tokenizer=None):
    parsed_final_output = []
    # dataset specific counter...
    correct_counter = 0
    per_type_total_counter = defaultdict(int)
    per_type_correct_counter = defaultdict(int)
    error_counter = defaultdict(int)
    
    if not hasattr(args, 'allow_both_eval'):
        args.allow_both_eval = True   # we should relex the template matching 

    assert len(dataset) == len(final_output), \
        f"Expect dataset and final_output to have the same length, got {len(dataset)} and {len(final_output)}"
    
    for i, (input_example, model_output) in enumerate(zip(dataset, final_output)):
        model_response = model_output['generated_text']
        extracted_answer = extract_answer_mmmu(model_response, args)
        correct_flag = 0
        
        input_example_skills = []
        
        if input_example['question_type'] == 'multiple-choice':
            options = eval(input_example['options'])
            
            input_example['answer_str'] = options[ord(input_example['response']) - 65]
            if (input_example['answer_str'] in extracted_answer or
                    extracted_answer in input_example['answer_str'] or 
                    extracted_answer == input_example['response']):
                correct_flag = 1
                correct_counter += 1
        else:   # open-ended
            try:
                possible_answers = eval(input_example['response'])
            except:
                possible_answers = input_example['response']
                
            if not isinstance(possible_answers, list):
                possible_answers = [str(possible_answers)]
            else:
                possible_answers = [str(ans) for ans in possible_answers]
            
            if (
                extracted_answer in possible_answers or
                any(extracted_answer in ans for ans in possible_answers) or 
                any(grade_answer(extracted_answer, ans) for ans in possible_answers)
            ):
                correct_flag = 1
                correct_counter += 1
        
        result = {
            'instruction': input_example['instruction'],
            'reasoning': model_response,
            'response': input_example['response'],  # ground truth
            'answer': extracted_answer,
            'correct': correct_flag
        }
        if tokenizer is not None:
            result['reasoning_token_length'] = len(tokenizer.encode(model_response))
            result['reasoning_str_length'] = len(model_response)
        
        if 'answer_str' in input_example:
            result['response_str'] = input_example['answer_str']
        
        parsed_final_output.append(result)
    
    acc = {"Total Accuracy": correct_counter / len(dataset)}
    
    if per_type_correct_counter:
        for skill, correct_count in per_type_correct_counter.items():
            acc[f"Accuracy on {skill}"] = correct_count / per_type_total_counter[skill]
    
    if error_counter:
        for error_type, count in error_counter.items():
            acc[f"{error_type}"] = count / len(dataset)
            
    if run_time is not None and isinstance(run_time, float):
        acc["run_time"] = run_time
        
    print(acc)
    
    output_parsed_path = args.precomputed_json.replace('.jsonl', f'_parsed.jsonl')
    output_acc_path = args.precomputed_json.replace('.jsonl', f'_acc.json')
    
    with open(output_parsed_path, 'w') as f:
        for item in parsed_final_output:
            f.write(json.dumps(item) + '\n')
            
    with open(output_acc_path, 'w') as f:
        json.dump(acc, f, indent=2)