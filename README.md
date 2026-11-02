# Loci performance take-home: scripts and data

Scripts and raw measurements behind my Loci take-home report. Everything was run on a MacBook Pro M3 Pro (18-core GPU,
36 GB), macOS 27.0, Loci 0.1.28, with Loci's default chat model for this Mac (Gemma 4 12B, `gemma-4-12b-it-Q4_K_M.gguf`).

The short version: Loci decodes as fast as stock llama.cpp on the same model file, but the wait before the first word
is 17–27 s on most messages because the hidden prompt gets re-processed instead of reused. `RESULTS.md` has the
numbers and which file each one comes from.

## What's here

| path | what it is |
|---|---|
| `bench/loci_bench.py` | Times Loci through its local OpenAI-compatible API: first token, prefill rate, prompt-cache reuse, multi-turn growth, decode, determinism, idle unload, sustained load, cold start. Samples memory while it runs. Python stdlib only. |
| `bench/outside_in.sh` | The control: runs stock llama.cpp (`llama-bench`, `llama-server`) on Loci's exact GGUF, then the same multi-turn and cache tests against it. |
| `bench/cache_equiv.py` | Checks whether reusing the prompt cache changes the model's output. Compares cached vs full recompute at temperature 0: token agreement, KL on top-20 logprobs, and the top-2 margin wherever a token differs. |
| `results/` | Raw output: one JSON line per request plus memory samples, llama-bench tables, server logs. |
| `screenshots/` | Loci's own per-message stats panel during the in-app tests. |
| `RESULTS.md` | Summary tables, each pointing at its raw file. |

## Running it

Needs Python 3 and, for the baseline, Homebrew llama.cpp (`brew install llama.cpp`; I used build 10964).

```bash
# Loci: turn on Settings → Advanced → Agent Access first. The script reads the URL and token from
# ~/Library/Application Support/LociRust/agent-api.json and never prints the token.
python3 bench/loci_bench.py probe            # check the API answers
python3 bench/loci_bench.py all              # ttft, prefill, cache, multiturn, decode, determinism (~20 min)
python3 bench/loci_bench.py idle --waits 60,300,900
python3 bench/loci_bench.py cold --runs 3 --yes   # quits and relaunches Loci

# Stock llama.cpp on the same file (quit Loci first). LOCI_GGUF overrides the model path.
bash bench/outside_in.sh

# Does cache reuse change answers?
python3 bench/cache_equiv.py --swa-full --history-tokens 600
```

`loci_bench.py` refuses to run when load average > 3 or swap > 1 GB, since timings would be off. `--force` skips that
check; a few runs here used it and `RESULTS.md` says which.

## Caveats

- One Mac, one model. Phone numbers in the report come from Loci's public speed page, not from me.
- API requests only carry my text, not the app's hidden prompt, so API runs measure the engine and the in-app stats
  measure what a user actually waits for.
- Some runs were on a busy machine (load 3–4 from macOS indexing). That slows absolute times but doesn't change
  whether the cache was reused.
