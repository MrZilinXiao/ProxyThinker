# To keep consistency with other works, let's use VLMEvalKit-style prompt to re-validate all responses.
# We read from existing raw jsonl files and read `generated_text` -- adopt the same evaluation strategy as VLMEvalKit.

# ------- general matching utils -------
import string
import copy as cp
import os
from vlmeval.smp import dump, load
from collections import defaultdict
from math_common_utils import (
    DatasetType, 
    MATHVISION_TESTMINI_IDS, 
)
from typing import Iterable
import time
import os.path as osp
import json
from latex2sympy2 import latex2sympy   # mandatory for mathvision evaluation
import timeout_decorator
import logging

logger = logging.getLogger(__name__)

FAIL_MSG = 'Failed to obtain answer via API.'

def track_progress_rich(
        func,
        tasks = tuple(),
        nproc: int = 4,
        save=None,
        keys=None,  # indices of tasks
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'
    # res = load(save) if save is not None else {}
    res = {}
    results = [None for _ in range(len(tasks))]

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    results[idx] = futures[idx].result()
                    new_finished.add(idx)
                    if keys is not None:
                        res[keys[idx]] = results[idx]
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            time.sleep(0.1)
        pbar.close()

    if save is not None:
        dump(res, save)
    
    return results   # after auxeval -- containing `log` and `res`


def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                print(f"A might be a quantifier in the string: {answer}.")
                # logger = get_logger('Evaluation')
                # logger.info(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

# ------- general matching utils -------


# ------- math_vista -------

def get_gpt4_mathvista():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def build_mathvista_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_mathvista()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}

# each line need:
# {
    # 'index': 1,
    # 'question_type': 'multi_choice' / 'free_form'
    # 'answer_type': 'integer' / 'float' / 'text'
    # 'question': 'question text'
    # 'prediction': 'model response'
# }
def post_check(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']  # 'prediction' from original model
    # 'res' from gpt-4o judge
    try:
        if line['question_type'] == 'multi_choice':
            ans = line['answer_option']
            # choices = list_to_dict(eval(line['choices']))
            choices = list_to_dict(line['choices'])
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            if line['answer_type'] == 'integer':
                res = int(response)
                ans = int(line['answer'])
            elif line['answer_type'] == 'float':
                res = float(response)
                ans = float(line['answer'])
            else:
                res = str(res)
                ans = str(ans)
    except ValueError:
        pass

    if res == ans:
        return res if prefetch else True
    else:
        return False
    
def mathvista_auxeval(model, line):  
    # need a threadpool to run this in parallel and get new_results of {'res': 'xxx', 'log': 'xxx'}
    prompt = build_mathvista_gpt4_prompt(line)
    log = ''
    retry = 5
    
    if post_check(line, prefetch=True):  # directly get response from GPT-4o
        res = post_check(line, prefetch=True)
        return dict(log='Prefetch succeed', res=res, prompt=prompt)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res, prompt=prompt)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='', prompt=prompt)


# ------- mathverse -------

def get_gpt4_mathverse_extract():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_mathverse_score():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4]


def build_mathverse_gpt4_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    prediction = str(line['prediction'])
    demo_prompt = task_description
    examples = get_gpt4_mathverse_extract()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt

def build_mathverse_gpt4_score_prompt(line):
    task_description = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    question_for_eval = line['question_for_eval']
    extract = line['extract']
    answer = line['answer']
    demo_prompt = task_description
    examples = get_gpt4_mathverse_score()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt

def post_check_score(line, prefetch=False):
    ans = str(line['answer']).strip()
    response = str(line['extract']).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False
    
def MathVerse_auxeval_extract(model, line):
    prompt = build_mathverse_gpt4_extract_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_extract=log, extract=res)
    log += 'All 5 retries failed.\n'
    return dict(log_extract=log, extract='')


def MathVerse_auxeval_score(model, line):
    prompt = build_mathverse_gpt4_score_prompt(line)
    log = ''
    retry = 5
    if post_check_score(line, prefetch=True):
        res = post_check_score(line, prefetch=True)
        return dict(log_score='Prefetch succeed', score=True)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res or res.strip() not in ['0', '1']:
            log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_score=log, score=int(res) == 1)
    log += 'All 5 retries failed.\n'
    return dict(log_score=log, score=False)

# ------- math_vision -------
# @timeout_decorator.timeout(30)
def is_equal(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) != str or not isinstance(gt_asw, str):
        print('Warning: input is not string')
        print(asw, gt_asw)
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    return False


def get_gpt4_mathvision():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]

def build_mathv_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_mathvision()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def post_check_mathvision(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    try:
        # if len(eval(line['choices'])) > 0:
        if len(line['choices']) > 0:
            ans = line['answer']
            # choices = list_to_dict(eval(line['choices']))
            choices = list_to_dict(line['choices'])
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except ValueError:
        pass

    try:
        if is_equal(res, ans):
            return res if prefetch else True
        else:
            return False
    except Exception as err:
        # logging.warning(f'{type(err)}: {err}')
        logger.warning(f'Error in is_equal: {err}')
        return False
    
def MATH_V_auxeval(model, line):
    prompt = build_mathv_gpt4_prompt(line)
    log = ''
    retry = 5
    if post_check_mathvision(line, prefetch=True):
        res = post_check_mathvision(line, prefetch=True)
        return dict(log='Prefetch succeed', res=res)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


def run_evaluate_math_with_judge(dataset, final_output, args, run_time=None, 
                                 judge_model=None):
    # parsed_final_output = []
    # dataset specific counter...
    correct_counter = 0
    per_type_total_counter = defaultdict(int)
    per_type_correct_counter = defaultdict(int)
    # error_counter = defaultdict(int)
    
    assert len(dataset) == len(final_output), \
        f"Expect dataset and final_output to have the same length, got {len(dataset)} and {len(final_output)}"
        
    evaluation_lines = []
        
    for i, (input_example, model_output) in enumerate(zip(dataset, final_output)):
        model_response = model_output['generated_text']
        # correct_flag = 0
        
        input_example_skills = []
        
        # different dataset need different line-format
        if args.dataset_type == DatasetType.MATHVISTA:
            input_example_skills = input_example['skills']
            for skill in input_example_skills:
                per_type_total_counter[skill] += 1
            
            # organize the `line` dict
            line = {
                'index': i + 1, 
                'question_type': input_example['question_type'],
                'answer_type': input_example['answer_type'],
                # 'answer_option': chr(65 + answer_index),
                'question': input_example['instruction'],
                'answer': input_example['response'],
                'prediction': model_response,
                'skills': input_example_skills,
            }
            if input_example['choices'] is not None:
                answer_index = input_example['choices'].index(
                    input_example['response']
                )
                line['answer_option'] = chr(65 + answer_index)
                line['choices'] = input_example['choices']
            
        elif args.dataset_type == DatasetType.MATHVERSE:
            line = {
                'index': i + 1,
                'question_for_eval': input_example['question_for_eval'],
                'prediction': model_response,
                'answer': input_example['response']
                # 'extract' fields will be added in extract step
            }
        
        elif args.dataset_type == DatasetType.MATHVISION:
            if input_example['id'] in MATHVISION_TESTMINI_IDS:
                input_example_skills = ['mathvision_testmini']
                per_type_total_counter['mathvision_testmini'] += 1
                
            line = {
                'index': i + 1,
                'question': input_example['instruction'],
                'prediction': model_response,
                'answer': input_example['response'],
                'choices': input_example['options'],  # mathvision 
                'skills': input_example_skills,
            }
        
        else:
            raise NotImplementedError(f"Dataset {args.dataset_type} not supported.")
        
        evaluation_lines.append(line)


    # ans = {}
    # if osp.exists(tmp_file):
    #     ans = load(tmp_file)
    # submit the lines to the auxeval judge 
    model_line_pairs = [(judge_model, line) for line in evaluation_lines]
    indices = [line['index'] for line in evaluation_lines]
    tmp_file =  args.precomputed_json.replace('.jsonl', '_auxeval.pkl')
    
    if args.dataset_type == DatasetType.MATHVISTA:
        new_results = track_progress_rich(
            mathvista_auxeval,
            model_line_pairs,
            # nproc=16,
            nproc=args.nproc,
            keys=indices, 
            save=tmp_file,
        )
        ans = load(tmp_file)   # 'res' and 'log' from gpt-4o judge
        # insert them back to the original lines
        for j, line in enumerate(evaluation_lines):
            line['res'] = ans[line['index']]['res']
            line['log'] = ans[line['index']]['log']
            # TODO: only for debug usage -- remove in production
            line['prompt'] = ans[line['index']]['prompt']
            
            # if line['log'] == 'Prefetch succeed' or post_check(line, prefetch=False):
            if post_check(line, prefetch=False):    
                line['correct_flag'] = 1
                correct_counter += 1
                for skill in line['skills']:
                    per_type_correct_counter[skill] += 1
            else:
                line['correct_flag'] = 0
        
    elif args.dataset_type == DatasetType.MATHVERSE:
        # mathverse needs dedicated extract and score
        tmp_file_extract = tmp_file.replace('.pkl', '_extract.pkl')
        tmp_file_score = tmp_file.replace('.pkl', '_score.pkl')
        
        # step 1: extract
        new_results = track_progress_rich(
            MathVerse_auxeval_extract, 
            model_line_pairs,
            # nproc=16,
            nproc=args.nproc,
            keys=indices,
            save=tmp_file_extract,
        )
        
        ans = load(tmp_file_extract)
        # insert them back to the original lines
        for j, line in enumerate(evaluation_lines):
            line['extract'] = ans[line['index']]['extract']
            line['log_extract'] = ans[line['index']]['log_extract']
        
        new_model_line_pairs = [(judge_model, line) for line in evaluation_lines]
        # indices remains the same
        # step 2: score
        new_results = track_progress_rich(
            MathVerse_auxeval_score,
            new_model_line_pairs,
            # nproc=16,
            nproc=args.nproc,
            keys=indices,
            save=tmp_file_score,
        )
        ans = load(tmp_file_score)
        for k, line in enumerate(evaluation_lines):
            line['score'] = ans[line['index']]['score']
            line['log_score'] = ans[line['index']]['log_score']
        
            # judgement = 1 --> correct; judgement = 0 --> incorrect
            if line['score']:
                line['correct_flag'] = 1
                correct_counter += 1
            
    elif args.dataset_type == DatasetType.MATHVISION:
        new_results = track_progress_rich(
            MATH_V_auxeval,
            model_line_pairs,
            # nproc=16,
            nproc=args.nproc,
            keys=indices,
            save=tmp_file,
        )
        ans = load(tmp_file)
        # insert them back to the original lines
        for j, line in enumerate(evaluation_lines):
            line['res'] = ans[line['index']]['res']
            line['log'] = ans[line['index']]['log']
            
            if post_check_mathvision(line, prefetch=False):
                line['correct_flag'] = 1
                correct_counter += 1
                for skill in line['skills']:
                    per_type_correct_counter[skill] += 1
    
    else:
        raise NotImplementedError(f"Dataset {args.dataset_type} not supported.")
        
    acc = {'Total Accuracy': correct_counter / len(evaluation_lines)}
    
    if per_type_correct_counter:
        for skill, correct_count in per_type_correct_counter.items():
            acc[f"Accuracy on {skill}"] = correct_count / per_type_total_counter[skill]
            
    print(acc)
            
    # should be done -- save it to _gpt_parsed.jsonl and a final acc file
    output_parsed_path = args.precomputed_json.replace('.jsonl', '_gpt_parsed.jsonl')
    output_acc_path = args.precomputed_json.replace('.jsonl', '_gpt_acc.json')
    
    with open(output_parsed_path, 'w') as f:
        for line in evaluation_lines:
            f.write(f"{line}\n")
            
    with open(output_acc_path, 'w') as f:
        json.dump(acc, f, indent=4)