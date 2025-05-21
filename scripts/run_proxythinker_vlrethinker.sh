mkdir -p /home/runai-home/hf_home
export HF_HOME='/home/runai-home/hf_home'
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200
export OMP_NUM_THREADS=16
export VLLM_USE_V1=0
export PYTHONPATH=./src:$PYTHONPATH

# then run contrast with r1onevision prompt
python src/vllm_run_proxythinker.py \
    --output_dir proxythinker_results/ \
    --base_model_path /home/runai-home/public_models/Qwen2.5-VL-72B-Instruct \
    --positive_model_path /home/runai-home/public_models/VL-Rethinker-7B \
    --negative_model_path /home/runai-home/public_models/Qwen2.5-VL-7B-Instruct \
    --datasets mathvista-testmini-vlrethinker mathvision-test-vlrethinker mathverse-testmini-vlrethinker \
    --gpu_utilization 0.6 --force_tp 8 --force_cd_tp 4 \
    --cd_decoding_alpha 0.5 \
    --max_concurrency 300 --max_model_len 32768 --cd_put_on_diff_gpus