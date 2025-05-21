## ProxyThinker: Test-Time Guidance through Small Visual Reasoners

This repository contains the code for the paper "ProxyThinker: Test-Time Guidance through Small Visual Reasoners" by Zilin Xiao, Jaywon Koo, Siru Ouyang, Jefferson Hernandez, Yu Meng and Vicente Ordonez. 



### Installing vLLM

The core implementation of ProxyThinker is at `vllm_proxythinker/vllm/contrast_decode/contrast_decode_worker.py`. We build this based on the contrastive decoding implementation by [simonucl](https://github.com/simonucl/vllm/tree/contrastive-decoding). 

We highly recommend installing the vLLM implementation of ProxyThinker for better performance. To do so, please follow the instructions below:
```bash
cd vllm_proxythinker
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/57/49/207364110b96d76139a4e80617e5831d46884abe824941b15c8a748ca5e0/vllm-0.8.2-cp38-abi3-manylinux1_x86_64.whl  
# our implementation is based on vLLM 0.8.2, so we use this precompiled wheel to acclerate the installation
pip install -e .
```

### Getting Started

Use this command to run ProxyThinker with vLLM on ProxyThinker-72B on 8GPUs:

```bash
mkdir -p /home/runai-home/hf_home
export HF_HOME='/home/runai-home/hf_home'
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200
export OMP_NUM_THREADS=16
export VLLM_USE_V1=0
export PYTHONPATH=./src:$PYTHONPATH

python src/vllm_run_proxythinker.py \
    --output_dir proxythinker_results/ \
    --base_model_path /home/runai-home/public_models/Qwen2.5-VL-72B-Instruct \
    --positive_model_path /home/runai-home/public_models/VL-Rethinker-7B \
    --negative_model_path /home/runai-home/public_models/Qwen2.5-VL-7B-Instruct \
    --datasets mathvista-testmini-vlrethinker mathvision-test-vlrethinker mathverse-testmini-vlrethinker \
    --gpu_utilization 0.6 --force_tp 8 --force_cd_tp 4 \
    --cd_decoding_alpha 0.5 \
    --max_concurrency 300 --max_model_len 32768 --cd_put_on_diff_gpus
```

If you'd like to try ProxyThinker without vLLM, you can use the native PyTorch implementation in src/.

```bash
PYTHONPATH=src/ python src/run_proxythinker.py \
  --base_model_path <base-model> \
  --positive_model_path <positive-model> \
  --negative_model_path <negative-model> \
  --cd_decoding_alpha 1.0 \
  --dataset <dataset_name> --split <split_name>
```

For more details, refer to `scripts` to see other examples of running ProxyThinker with different models and datasets.
