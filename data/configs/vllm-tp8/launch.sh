#!/bin/bash
python -m dynamo.frontend --http-port 8000 --router-mode round-robin --kv-cache-block-size 64 > ${RUNS_DIR}/vllm-tp8/frontend.log 2>&1 &
env DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081  python -m dynamo.vllm --model ${MODEL_DIR}/Qwen3-235B-A22B-Instruct-2507-FP8 --served-model-name qwen235 --max-model-len 40960 --gpu-memory-utilization 0.92 --block-size 64 --max-num-seqs 1024 --enable-prefix-caching --tensor-parallel-size 8 > ${RUNS_DIR}/vllm-tp8/worker0.log 2>&1 &
wait
