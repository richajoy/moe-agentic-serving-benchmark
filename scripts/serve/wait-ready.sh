#!/bin/bash
# wait-ready.sh: wait until the frontend lists a model AND a short chat completion succeeds; prints seconds.
# A model can stay listed after its worker died, so the completion must return HTTP 200 as well.
s=$(date +%s)
until curl -sf localhost:8000/v1/models | grep -q '"id"'; do
  sleep 5
  [ $(( $(date +%s) - s )) -gt "${TMO:-1200}" ] && { echo TIMEOUT; exit 1; }
  docker ps -q -f name=^srv$ | grep -q . || { echo DEAD; exit 2; }
done
M=$(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
code=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8}")
[ "$code" = 200 ] || { echo "NOT READY: chat completion returned HTTP $code"; exit 3; }
echo "READY $(( $(date +%s) - s ))s"
