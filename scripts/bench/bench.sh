#!/bin/bash
# bench.sh RUN SHAPE C [TURNS]   one AIPerf point against localhost:8000; results -> $RUNS_DIR/RUN/SHAPE-cC[-tT]
#   SHAPE: AG (agentic sessions), S1 (512 in / 4,096 out), S2 (16,384 in / 256 out), S3 (16,384 in / 4,096 out)
#   AG: 2,000-token shared system prompt, +600 in / 200 out per turn, TURNS +- TURNS/4 turns, 1,000 +- 250 ms tool gap,
#       sessions = max(C, 64); default stop rule = sessions x TURNS requests (count-capped)
#   DUR / GRACE: switch to a duration cap (seconds) plus grace period
#   METRICS_URLS: worker /metrics endpoints AIPerf scrapes (P/D: both workers)
: "${RUNS_DIR:?set RUNS_DIR}" "${MODEL_DIR:?set MODEL_DIR}"
RUN=$1; SH=$2; C=$3; T=${4:-8}; TOK=$MODEL_DIR/Qwen3-235B-A22B-Instruct-2507-FP8; AIPERF=${AIPERF:-aiperf}
MODEL=$(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
MET=${METRICS_URLS:-"http://localhost:8081/metrics"}
N=$(( C > 100 ? C : 100 ))
case $SH in
 S1) A="--isl 512 --osl 4096 --request-count $N --num-dataset-entries $N"; O=4096;;
 S2) A="--isl 16384 --osl 256 --request-count $N --num-dataset-entries $N"; O=256;;
 S3) A="--isl 16384 --osl 4096 --request-count $N --num-dataset-entries $N"; O=4096;;
 AG) S=$(( C > 64 ? C : 64 )); A="--shared-system-prompt-length 2000 --isl 600 --osl 200 --conversation-num $S --conversation-turn-mean $T --conversation-turn-stddev $(( T/4 )) --conversation-turn-delay-mean 1000 --conversation-turn-delay-stddev 250 --request-count $(( S*T ))"; O=200;;
 *) echo "unknown shape $SH"; exit 2;;
esac
[ -n "$DUR" ] && A=$(echo "$A" | sed -E "s/--request-count [0-9]+/--benchmark-duration $DUR --benchmark-grace-period ${GRACE:-180}/")
OUT=$RUNS_DIR/$RUN/$SH-c$C$([ "$SH" = AG ] && echo "-t$T"); mkdir -p "$OUT"
echo "START $RUN $SH c=$C $A" | tee -a "$RUNS_DIR/$RUN/bench.log"
timeout "${TMO:-900}" "$AIPERF" profile -m "$MODEL" --tokenizer "$TOK" --url http://localhost:8000 --endpoint-type chat --streaming \
  $A --osl-stddev 0 --isl-stddev 0 --extra-inputs ignore_eos:true --extra-inputs min_tokens:$O --concurrency "$C" --random-seed 42 \
  --export-level records --server-metrics $MET --ui-type none --artifact-dir "$OUT" > "$OUT/aiperf.log" 2>&1
echo "END rc=$? $OUT" | tee -a "$RUNS_DIR/$RUN/bench.log"
