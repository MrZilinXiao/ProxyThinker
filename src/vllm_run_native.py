import os
import json
import argparse
import torch
import asyncio
import time
import uuid
from vllm import AsyncLLMEngine, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.engine.arg_utils import AsyncEngineArgs
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
# from logger import logger
from math_common_utils import (
    Timer, 
    DatasetType, 
    ModelConfig, 
    run_evaluate_math,
    get_dataset_config, 
    model_name_to_tp,
    process_dataset_to_message,
    load_image_dataset,
)

from logger import setup_logger

def parse_arguments():
    global logger 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/runai-home/public_models/Qwen2-VL-72B-Instruct")
    parser.add_argument('--output_dir', default="results/", type=str)
    
    parser.add_argument('--log_dir', default="logs/", type=str)
    parser.add_argument('--dataset', default="mathvista", type=str)
    parser.add_argument('--split', type=str, choices=["testmini", "test"], default="testmini")
    parser.add_argument('--datasets', nargs='+', help="List of dataset configs in format: dataset-split-prompt_type")
    parser.add_argument('--skip_multi_choice', action='store_true', default=False)
    
    parser.add_argument("--precomputed_json", type=str)
    # added back for debug usage -- run on 10 samples to see if it works
    parser.add_argument('--num_samples', type=int, default=None)  # number of samples to process
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--bs', default=500, type=int)
    parser.add_argument('--force_tp', default=None, type=int)
    parser.add_argument('--force_pp', default=1, type=int)
    parser.add_argument('--force_processor', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, choices=["answer", "bboxed", "none"], default="bboxed")
    parser.add_argument('--gpu_utilization', default=0.8, type=float)
    parser.add_argument('--max_concurrency', default=500, type=int)  # Control concurrent requests
    parser.add_argument('--max_model_len', type=int, default=65536)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    # added beam search options
    parser.add_argument('--num_beams', type=int, default=None)   # do not support beam search?
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    # added for min max pixel -- default of Qwen2.5VL
    parser.add_argument('--min_pixels', type=int, default=None)
    parser.add_argument('--max_pixels', type=int, default=None)
    # added for temperature sampling
    parser.add_argument('--n', type=int, default=1)    # multiple outputs will be generated
    parser.add_argument('--enable_graph', action='store_true', help="Enable graph mode")
    parser.add_argument('--enable_prefix_caching', action='store_true', help="Enable prefix caching")
    args = parser.parse_args()
    # setup logger here and write all parameters to the log file
    logger = setup_logger(
        log_dir=args.log_dir,
        log_file=f"{args.model_path.split('/')[-1]}-{args.dataset}-{args.split}-{args.prompt_type}.log",
    )
    
    logger.info("Arguments: %s", args)
    
    return args

async def generate_one(engine, input_data, sampling_params, request_id):
    """Generate output for a single input"""
    try:
        results_generator = engine.generate(
            input_data, 
            sampling_params, 
            request_id=request_id,
        )
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        ret = {
            "prompt": input_data["prompt"],
            "generated_text": final_output.outputs[0].text if final_output and final_output.outputs else ""
        }
        if len(final_output.outputs) > 1:
            ret['generated_text_group'] = [
                output.text for output in final_output.outputs
            ]
        
        return ret
        
    except Exception as e:
        # logger.error(f"Error generating for request {request_id}: {e}")
        return {
            "prompt": input_data["prompt"],
            "generated_text": f"ERROR: {str(e)}",
            "error": str(e)
        }


class AsyncSemaphore:
    """A semaphore to control concurrency with tracking capabilities"""
    def __init__(self, value):
        self.semaphore = asyncio.Semaphore(value)
        self.active = 0
        self.total = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        await self.semaphore.acquire()
        async with self.lock:
            self.active += 1
            self.total += 1
    
    async def release(self):
        self.semaphore.release()
        async with self.lock:
            self.active -= 1
    
    @property
    def active_count(self):
        return self.active
    
    @property
    def total_count(self):
        return self.total


async def process_single_dataset_async(args, engine, processor, dataset_type, split, prompt_type):
    """Process a single dataset configuration asynchronously"""
    dataset_config = get_dataset_config(dataset_type, split=split, skip_multi_choice=args.skip_multi_choice)
    
    model_config = ModelConfig(
        model_name=args.model_path,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    real_model_name = args.model_path.split("/")[-1]
    assert real_model_name, f"Expect a non-empty model name, got {real_model_name}"
    
    # where we save the results
    result_path = os.path.join(args.output_dir, 
                             f"{dataset_type.value}-{split}", 
                             f"{real_model_name}_{prompt_type}{'_' + args.tags if args.tags is not None else ''}.jsonl")

    # if args.precomputed_json is None:
    args.precomputed_json = result_path
    
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if args.num_beams is not None:
        sampling_params = BeamSearchParams(
            beam_width=args.num_beams,
            max_tokens=args.max_new_tokens,
            temperature=model_config.temperature
        )
    else:
        sampling_params = SamplingParams(
            n=args.n,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
            max_tokens=model_config.max_new_tokens,
            repetition_penalty=model_config.repetition_penalty,
        )
        if args.n > 1:   # temperature sampling needs to assert temperature, top_p, top_k
            assert args.temperature > 0.01, "Temperature must be greater than 0.01 for sampling."
            assert args.top_p == 0.95, "Top-p must be 0.95 for sampling."
            assert args.top_k == -1, "Top-k must be -1 for sampling all `top_p` tokens."

    print(f"Loading dataset {dataset_config}")
    dataset = load_image_dataset(dataset_config)
    
    # Save the current prompt_type to process the dataset correctly
    original_prompt_type = args.prompt_type
    args.prompt_type = prompt_type
    # args.dataset = dataset_type
    args.dataset_type = dataset_type
    
    resp_messages = process_dataset_to_message(dataset, args)
    # convert to vllm format
    print(f"Converting {len(resp_messages)} examples to vllm format...")
    
    prompt = processor.apply_chat_template(
        resp_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, _ = process_vision_info(resp_messages)
    
    image_st_ed = []
    image_st = 0
    for i, messages in enumerate(resp_messages):
        image_count = sum(1 for msg in messages[0]['content'] if msg['type'] == 'image')
        image_st_ed.append((image_st, image_st + image_count))
        image_st += image_count

    llm_inputs = []

    for i in range(len(prompt)):
        llm_input_dict = {
            "prompt": prompt[i],
            "multi_modal_data": {
                "image": image_inputs[image_st_ed[i][0]: image_st_ed[i][1]],
            }
        }

        if args.min_pixels is not None or args.max_pixels is not None:
            assert args.min_pixels is not None and args.max_pixels is not None, \
                "Both min_pixels and max_pixels should be specified together."
            llm_input_dict["mm_processor_kwargs"] = {
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
        llm_inputs.append(llm_input_dict)

    print(f"Converted {len(llm_inputs)} examples to vllm format.")
    
    # Pre-allocate results array
    final_output = [None] * len(llm_inputs)
    semaphore = AsyncSemaphore(args.max_concurrency)
    
    async def process_input(idx, input_data):
        await semaphore.acquire()
        try:
            request_id = f"req_{uuid.uuid4()}"
            result = await generate_one(engine, input_data, sampling_params, request_id)
            final_output[idx] = result
            
            # Print progress periodically
            if semaphore.total_count % 10 == 0 or semaphore.total_count == len(llm_inputs):
                progress = semaphore.total_count / len(llm_inputs) * 100
                print(f"Progress: {progress:.2f}% complete ({semaphore.total_count}/{len(llm_inputs)}) ({dataset_type.value} on {real_model_name}), Active: {semaphore.active_count}")
                
            return result
        finally:
            await semaphore.release()
    
    # Process all inputs asynchronously with concurrency control
    with Timer(f"AsyncLLM inference for {dataset_type.value}-{split}-{prompt_type}") as t:
        # Create all tasks at once, but execution will be controlled by the semaphore
        tasks = [process_input(i, input_data) for i, input_data in enumerate(llm_inputs)]
        await asyncio.gather(*tasks)
    
    print(f"Generated {len(final_output)} examples with {t.elapsed_time:.2f} seconds.")
    
    # Filter out any None values that might have occurred due to errors
    final_output = [item for item in final_output if item is not None]
        
    with open(result_path, 'w') as f:
        for item in final_output:
            f.write(json.dumps(item) + '\n')
            
    # run the evaluation
    if args.n == 1:  # run eval only in greedy mode
        run_evaluate_math(dataset, final_output, args, run_time=t.elapsed_time, tokenizer=processor.tokenizer)
    
    # Restore the original prompt_type
    args.prompt_type = original_prompt_type
            
    print(f"Results saved to {result_path}")
    return t.elapsed_time

async def main():
    args = parse_arguments()
    
    # Load the model and processor once
    real_model_name = args.model_path.split("/")[-1]
    
    # Initialize AsyncLLMEngine
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        limit_mm_per_prompt={"image": 7},
        dtype="bfloat16",
        enforce_eager=not args.enable_graph,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=args.gpu_utilization,
        tensor_parallel_size=model_name_to_tp(args, real_model_name),
        pipeline_parallel_size=args.force_pp,
        max_model_len=args.max_model_len,
        disable_log_requests=True, 
    )
    
    print("Initializing AsyncLLMEngine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("AsyncLLMEngine initialized successfully")
    
    if args.force_processor is not None:
        processor = AutoProcessor.from_pretrained(args.force_processor)
    else:
        processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Process datasets
    total_time = 0
    
    if args.datasets:
        for dataset_config in args.datasets:
            parts = dataset_config.split('-')
            if len(parts) == 3:
                dataset_name, split, prompt_type = parts
            elif len(parts) == 2:
                dataset_name, split = parts
                prompt_type = args.prompt_type
            else:
                raise ValueError(f"Invalid dataset config format: {dataset_config}")
            
            dataset_type = DatasetType(dataset_name)
            
            print(f"\nProcessing dataset: {dataset_name}, split: {split}, prompt_type: {prompt_type}")
            elapsed_time = await process_single_dataset_async(args, engine, processor, dataset_type, split, prompt_type)
            total_time += elapsed_time
    else:
        # Use the original single dataset approach
        dataset_type = DatasetType(args.dataset)
        elapsed_time = await process_single_dataset_async(args, engine, processor, dataset_type, args.split, args.prompt_type)
        total_time = elapsed_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"Total script execution time: {time.time() - start_time:.2f} seconds")
    