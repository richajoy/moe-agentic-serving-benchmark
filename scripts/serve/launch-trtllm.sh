#!/bin/bash
# launch-trtllm.sh: TensorRT-LLM 1.3.0rc29 (PyTorch backend), TP8 attention + EP8 experts, one worker.
: "${MODEL_DIR:?set MODEL_DIR}" "${CONFIG_DIR:?set CONFIG_DIR to the repo deploy/ directory}"
IMG=${IMG:-nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.6.0} WORKER_MOD=dynamo.trtllm FRONT_ARGS="--kv-cache-block-size 64" \
  "$(dirname "$0")/up.sh" "${RUN:-trtllm-tp8ep8}" round-robin \
  "|--model $MODEL_DIR/Qwen3-235B-A22B-Instruct-2507-FP8 --served-model-name qwen235 --extra-engine-args $CONFIG_DIR/trtllm-tp8ep8.yaml --publish-kv-events --publish-metrics"
