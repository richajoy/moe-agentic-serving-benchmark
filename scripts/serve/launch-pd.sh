#!/bin/bash
# launch-pd.sh: P/D 1 prefill : 1 decode on one 8-GPU node.
#   decode  worker: TP4+EP4 on GPUs 0-3, NixlConnector kv_consumer
#   prefill worker: TP4+EP4 on GPUs 4-7, NixlConnector kv_producer
# KVDT=fp8 (default) or auto (BF16 KV). RUN names the run directory.
: "${MODEL_DIR:?set MODEL_DIR}"
M=$MODEL_DIR/Qwen3-235B-A22B-Instruct-2507-FP8
COMMON="--model $M --served-model-name qwen235 --max-model-len 40960 --gpu-memory-utilization 0.92 --block-size 64 --enable-prefix-caching --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype ${KVDT:-fp8}"
CONS='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
PROD='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
FRONT_ARGS="--kv-cache-block-size 64" "$(dirname "$0")/up.sh" "${RUN:-vllm-pd-1p1d-fp8kv}" round-robin \
 "PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NIXL_SIDE_CHANNEL_PORT=20096 UCX_TLS=cuda_ipc,cuda_copy,tcp|$COMMON --max-num-seqs 512 --disaggregation-mode decode --kv-transfer-config '$CONS'" \
 "PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 UCX_TLS=cuda_ipc,cuda_copy,tcp|$COMMON --max-num-seqs 256 --disaggregation-mode prefill --kv-transfer-config '$PROD'"
