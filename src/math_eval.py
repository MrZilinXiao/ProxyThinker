# usage: python math_eval.py --precomputed_json results/mathvista-testmini/Qwen2.5-VL-32B-Instruct_bboxed.jsonl
import json
import argparse
from transformers import AutoProcessor

from math_common_utils import (
    DatasetType, 
    get_dataset_config, 
    load_image_dataset,
    run_evaluate_math, 
    run_evaluate_mmmu, 
    run_evaluate_math_pass_at_k
)
from math_judge_utils import run_evaluate_math_with_judge
from vlmeval.dataset.utils.judge_util import build_judge
from logger import setup_logger
import os

logger = None

def parse_arguments():
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed_json", type=str, required=True)
    parser.add_argument("--gpt_judge", action="store_true", default=False)
    parser.add_argument("--passk_judge", action="store_true", default=False)  # use pass@k judge
    parser.add_argument("--skip_multi_choice", action="store_true", default=False)  # skip multi-choice: use this when processed jsonl is also skipped
    parser.add_argument("--nproc", type=int, default=64)   # apply only to gpt judge
    parser.add_argument("--allow_both_eval", action="store_true", default=True)
    parser.add_argument("--tags", type=str, default='external_vis')  # tags for debug usage
    # technically we could parse the dataset, split from the jsonl file
    args = parser.parse_args()
    logger = setup_logger(
        log_dir="eval_logs", 
        log_file=os.path.basename(args.precomputed_json).replace('.jsonl', f'_{args.tags + "_" if args.tags is not None else ""}eval.log'),
    )
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    final_output = []
    
    # read the jsonl file
    with open(args.precomputed_json, "r") as f:
        for line in f:
            item = json.loads(line)
            final_output.append(item)
    
    # get the dataset name and split from the jsonl file
    dataset_name = args.precomputed_json.split("/")[-2]
    dataset_name, split = dataset_name.split("-")
    # init the same dataset
    dataset_type = DatasetType(dataset_name)
    # pass@k evaluation skips multi-choice
    dataset_config = get_dataset_config(dataset_type, split=split, skip_multi_choice=args.skip_multi_choice)
    dataset = load_image_dataset(dataset_config)
    
    assert len(final_output) == len(dataset), f"Length of jsonl file {len(final_output)} does not match dataset {len(dataset)}"
    
    args.dataset_type = dataset_type
    
    # get the prompt type from the jsonl file -- we need this to select the correct parsing method
    if "bboxed" in args.precomputed_json.lower():
        prompt_type = "bboxed"
    else:
        prompt_type = "answer"
    
    args.prompt_type = prompt_type
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    if args.gpt_judge:
        assert dataset_type != DatasetType.MMMU_VAL, f"GPT judge not supported for {dataset_type} dataset"
        judge_kwargs = {
            'model': 'gpt-4o-mini', 
            'nproc': args.nproc, 
            'verbose': False,
        }
        model = build_judge(max_tokens=128, **judge_kwargs)
        run_evaluate_math_with_judge(dataset, final_output, args, judge_model=model)
        exit(0)
    elif args.passk_judge:
        run_evaluate_math_pass_at_k(dataset, final_output, args)
        exit(0)
        
    # non-gpt judge has own tags
    if args.tags is not None:
        args.precomputed_json = args.precomputed_json.replace('.jsonl', f'_{args.tags}.jsonl')
        
    if dataset_type == DatasetType.MMMU_VAL:
        run_evaluate_mmmu(dataset, final_output, args, tokenizer=processor.tokenizer)
    else:
        run_evaluate_math(dataset, final_output, args, tokenizer=processor.tokenizer)