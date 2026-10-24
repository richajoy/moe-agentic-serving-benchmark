#!/bin/bash
python -m dynamo.frontend --http-port 8000 --router-mode round-robin --kv-cache-block-size 64 > ${RUNS_DIR}/trtllm-tp8ep8/frontend.log 2>&1 &
env DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081  python -m dynamo.trtllm --model ${MODEL_DIR}/Qwen3-235B-A22B-Instruct-2507-FP8 --served-model-name qwen235 --extra-engine-args ${CONFIG_DIR}/trtllm-tp8ep8.yaml --publish-kv-events --publish-metrics > ${RUNS_DIR}/trtllm-tp8ep8/worker0.log 2>&1 &
wait
