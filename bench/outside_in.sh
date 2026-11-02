#!/bin/bash
# Outside-in baselines on Loci's exact GGUF with stock llama.cpp. Run with Loci quit.
set -u
M="${LOCI_GGUF:-$HOME/Library/Application Support/LociRust/Models/artifacts/16c647ff3318b79cbb0bd202b1198da69c1e95bc01c51b17e6cab888b9f7ff50/gemma-4-12b-it-Q4_K_M.gguf}"
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/../results"; mkdir -p "$OUT"; T=$(date +%Y%m%d-%H%M%S)
llama-server --version 2>&1 | head -1 | tee "$OUT/$T-llamacpp-version.txt"

if [ "${SKIP_BENCH:-0}" != 1 ]; then
echo "== llama-bench: prefill (pp) and decode (tg) =="
llama-bench -m "$M" -p 512,2048,4096 -n 128 -fa 1 -r 3 -o md | tee "$OUT/$T-llama-bench.md"
fi

for mode in default swa-full; do
  echo "== llama-server ($mode): multiturn + cache =="
  extra=(); [ "$mode" = swa-full ] && extra=(--swa-full)
  llama-server -m "$M" -c 8192 -np 1 -fa on --port 8090 ${extra[@]+"${extra[@]}"} > "$OUT/$T-server-$mode.log" 2>&1 & P=$!
  until curl -s http://127.0.0.1:8090/health | grep -q ok; do sleep 0.5; done
  for t in multiturn cache; do
    python3 "$HERE/loci_bench.py" $t --force --base-url http://127.0.0.1:8090/v1 --no-auth --pid "$P"
  done
  kill $P; wait $P 2>/dev/null
done

echo "== cache_equiv: does cache reuse change outputs? =="
python3 "$HERE/cache_equiv.py"
