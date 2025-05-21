# use native Python implemnetation of ProxyThinker
import os
import json
import torch
import argparse
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2VLForConditionalGeneration
)
from math_common_utils import (
    Timer, 
    DatasetType, 
    run_evaluate_math,
    get_dataset_config, 
    process_dataset_to_message,
    load_image_dataset,
)

from proxy_thinker_utils import (
    ProxyThinkerWrapper, generate_completions
)

from logger import setup_logger

logger = None

def parse_arguments():
    global logger 
    # native version keeps most of the params the same with vLLM version
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default="/home/runai-home/public_models/Qwen2.5-VL-32B-Instruct")
    parser.add_argument('--positive_model_path', type=str, default=None)
    parser.add_argument('--negative_model_path', type=str, default=None)
    parser.add_argument('--cd_decoding_alpha', type=float, default=1.0)
    parser.add_argument('--cd_put_on_diff_gpus', action='store_true', default=False)
    parser.add_argument('--output_dir', default="cd_native_results/", type=str)
    parser.add_argument('--log_dir', default="logs/", type=str)
    parser.add_argument('--dataset', default="mathvista", type=str)
    parser.add_argument('--split', type=str, choices=["testmini", "test"], default="testmini")
    parser.add_argument('--datasets', nargs='+', help="List of dataset configs in format: dataset-split-prompt_type")
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, choices=["answer", "bboxed", "none"], default="bboxed")  # ignore this field
    parser.add_argument('--num_samples', type=int, default=None) 
    parser.add_argument('--skip_multi_choice', action='store_true', default=False)
    
    parser.add_argument('--max_model_len', default=65536, type=int)  # Max model length
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--min_pixels', type=int, default=None)
    parser.add_argument('--max_pixels', type=int, default=None)
    
    # only native python version needs explicit bs
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_auto', action='store_true', default=False)
    parser.add_argument('--n', type=int, default=1)    # multiple outputs will be generated
    
    args = parser.parse_args()
    logger = setup_logger(
        log_dir=args.log_dir,
        log_file=f"native_{args.base_model_path.split('/')[-1]}-{args.dataset}-{args.split}.log",
    )
    logger.info("Arguments: %s", args)
    return args

def get_device_map(model_path, use_auto=False, is_positive=False):
    if use_auto or '72B' in model_path or '32B' in model_path:
        return "auto"
    elif torch.cuda.device_count() == 1:  # if only on device available, all small models go here
        return {"": 0}
    else:
        return {"": 1} if not is_positive else {"": 0}
    
def process_single_dataset(args, cd_model, processor, dataset_type, split, prompt_type, 
                           stop_id_seqs, banned_begin_ids, model_suffix=None):
    dataset_config = get_dataset_config(dataset_type, split=split)
    
    print(f"Loading dataset {dataset_config}...")
    dataset = load_image_dataset(dataset_config)
    
    # Save the current prompt_type to process the dataset correctly
    original_prompt_type = args.prompt_type
    args.prompt_type = prompt_type
    # args.dataset = dataset_type
    args.dataset_type = dataset_type
    
    resp_messages = process_dataset_to_message(dataset, args)
    with Timer(f"Huggingface Generation for {dataset_type.value}-{split}-{prompt_type}") as t:
        generations = generate_completions(
            cd_model,
            processor,
            resp_messages,
            positive_messages=None, 
            negative_messages=None,
            batch_size=args.batch_size,
            stop_id_seqs=stop_id_seqs,
            banned_id_seqs=None, 
            # banned_begin_ids=banned_begin_ids,
            banned_begin_ids=None, 
            disable_tqdm=False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            do_sample=False, 
            use_cache=True,
            is_instruct=True,   # use qwen_vl_utils to process the images
        )
        
    ret_list = []
    for i, generation in enumerate(generations):
        ret = {
            'prompt': resp_messages[i][0]['content'][-1]['text'],
            'generated_text': generation,
        }
        ret_list.append(ret)
        
    result_path = os.path.join(args.output_dir, 
                             f"{dataset_type.value}-{split}", 
                             f"{model_suffix}_{prompt_type}{'_' + args.tags if args.tags is not None else ''}.jsonl")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    args.precomputed_json = result_path
    
    with open(args.precomputed_json, 'w') as f:
        for item in ret_list:
            f.write(json.dumps(item) + '\n')
            
    run_evaluate_math(dataset, ret_list, args, run_time=t.elapsed_time, tokenizer=processor.tokenizer)
    # Restore the original prompt_type
    args.prompt_type = original_prompt_type
            
    print(f"Results saved to {result_path}")
    return t.elapsed_time
    
    
if __name__ == "__main__":
    args = parse_arguments()
    # Load the model and processor once
    real_model_name = args.base_model_path.split("/")[-1]
    assert real_model_name, f"Expect a non-empty model name, got {real_model_name}"
    
    if args.positive_model_path:
        positive_model_name = args.positive_model_path.split("/")[-1]
        assert positive_model_name, f"Expect a non-empty positive model name, got {positive_model_name}"
        model_suffix = f"native_cd_{args.cd_decoding_alpha}_{real_model_name}-{positive_model_name}"
    else:
        model_suffix = real_model_name
    
    if args.cd_decoding_alpha is not None:
        assert args.base_model_path is not None, "Please provide a base model path"
        assert args.positive_model_path is not None, "Please provide a positive model path"
        
    # ready to init three models
    print("Loading models...")
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=get_device_map(args.base_model_path, 
                                  use_auto=args.use_auto),
    )
    
    positive_model, negative_model = None, None
    if args.positive_model_path is not None:
        positive_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.positive_model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            # device_map={"": 1},
            device_map=get_device_map(args.positive_model_path, 
                                      use_auto=args.use_auto, 
                                      is_positive=True),
        )
    
    if args.negative_model_path is not None:
        negative_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.negative_model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map=get_device_map(args.negative_model_path, 
                                      use_auto=args.use_auto, 
                                      is_positive=False),
        )
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    cd_model = ProxyThinkerWrapper(
        base_model,
        processor,
        positive_model=positive_model,
        negative_model=negative_model,
        do_torch_compile=False,
        alpha=args.cd_decoding_alpha,
    )
    stop_id_seqs = base_model.generation_config.eos_token_id
    if isinstance(stop_id_seqs, list):
        stop_id_seqs = [[stop_id_seq] for stop_id_seq in stop_id_seqs]
    elif isinstance(stop_id_seqs, int):
        stop_id_seqs = [[stop_id_seqs]]
    else:
        raise ValueError(f"Invalid stop_id_seqs: {stop_id_seqs}")
    
    banned_begin_ids = base_model.generation_config.eos_token_id
    if isinstance(banned_begin_ids, int):
        banned_begin_ids = [banned_begin_ids]
    total_time = 0
    
    # load the dataset and run the model
    for dataset_config in args.datasets:
        # try:
        parts = dataset_config.split('-')
        if len(parts) == 3:
            dataset_name, split, prompt_type = parts
        elif len(parts) == 2:
            dataset_name, split = parts
            prompt_type = args.prompt_type
        else:
            raise ValueError(f"Invalid dataset config format: {dataset_config}")
        
        dataset_type = DatasetType(dataset_name)
        elapsed_time = process_single_dataset(
            args, cd_model, processor, dataset_type, split, prompt_type,
            stop_id_seqs=stop_id_seqs, banned_begin_ids=banned_begin_ids, model_suffix=model_suffix
        )
        total_time += elapsed_time
        
    print(f"\nTotal processing time: {total_time:.2f} seconds")