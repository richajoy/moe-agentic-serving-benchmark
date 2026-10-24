#!/bin/bash
# up.sh NAME ROUTER_MODE "WORKER1_ENV|WORKER1_ARGS" ["WORKER2_ENV|WORKER2_ARGS" ...]
# Starts the Dynamo frontend plus one or more workers in one container named srv. Logs and the exact
# launch script go to $RUNS_DIR/NAME/ (cmd.txt is what data/configs/<run>/cmd.txt records).
#   ROUTER_MODE   round-robin | kv
#   WORKER_MOD    dynamo.vllm (default) or dynamo.trtllm
#   FRONT_ARGS    extra frontend flags, e.g. "--kv-cache-block-size 64"
#   IMG           container image (default: the Dynamo 1.6.0 vLLM runtime)
set -u
: "${RUNS_DIR:?set RUNS_DIR}" "${MODEL_DIR:?set MODEL_DIR}" "${CACHE_DIR:?set CACHE_DIR}"
NAME=$1; MODE=$2; shift 2; D=$RUNS_DIR/$NAME; mkdir -p "$D"
IMG=${IMG:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0}
{ echo "#!/bin/bash"
  echo "python -m dynamo.frontend --http-port 8000 --router-mode $MODE ${FRONT_ARGS:---kv-cache-block-size 64} > $D/frontend.log 2>&1 &"
  i=0; for w in "$@"; do env="${w%%|*}"; args="${w#*|}"; port=$((8081+i))
    echo "env DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=$port $env python -m ${WORKER_MOD:-dynamo.vllm} $args > $D/worker$i.log 2>&1 &"; i=$((i+1)); done
  echo "wait"; } > "$D/launch.sh"
chmod +x "$D/launch.sh"; cp "$D/launch.sh" "$D/cmd.txt"
docker rm -f srv >/dev/null 2>&1; until ! docker ps -a -q -f name=^srv$ | grep -q .; do sleep 2; done
# Known issue: vLLM 0.31.0 can hang in FlashInfer MoE autotune on a warm restart of an EP/DP worker
# (vllm#57423, fixed in vLLM 0.32.0). Starting every launch with an empty autotune cache avoids it.
[ -d "$CACHE_DIR/vllm/flashinfer_autotune_cache" ] && mv "$CACHE_DIR/vllm/flashinfer_autotune_cache" "$CACHE_DIR/vllm/flashinfer_autotune_cache.old-$$"
docker run -d --name srv --gpus all --ipc=host --network host --shm-size 200g --ulimit memlock=-1 \
  -v "$MODEL_DIR:$MODEL_DIR" -v "$RUNS_DIR:$RUNS_DIR" -v "$CACHE_DIR:$CACHE_DIR" ${CONFIG_DIR:+-v "$CONFIG_DIR:$CONFIG_DIR"} \
  -e HF_HOME="$CACHE_DIR/hf" -e VLLM_CACHE_ROOT="$CACHE_DIR/vllm" -e TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/inductor" \
  -e FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer" -e TRITON_CACHE_DIR="$CACHE_DIR/triton" \
  "$IMG" bash "$D/launch.sh" >/dev/null
echo "started $NAME"
