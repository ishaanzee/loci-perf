# Results

M3 Pro (18-core GPU, 36 GB, ~150 GB/s), macOS 27.0, Loci 0.1.28, Gemma 4 12B Q4_K_M (7.12 GB), AC power.
Stock llama.cpp is Homebrew build 10964 on the same GGUF file, with Loci quit.

## In the app (Loci's stats panel)

Oct 31, `screenshots/2026-10-31-step*.png`:

| step | prompt tokens | from cache | first word |
|---|---|---|---|
| 1. new chat, "What is a vector subspace?" | 4,207 | 0 | 24.43 s |
| 2. same chat, follow-up | 4,825 | 0 | 27.09 s |
| 3. same chat, follow-up | 5,751 | 5,731 (99%) | 0.64 s |
| 4–5. new chat, "How do I boil an egg?" (done twice) | 4,208 | 0 | 22.41 / 21.95 s |
| 6. memories off, new chat, same question | 3,284 | 0 | 16.87 s |
| 7. relaunched, memories + tools off | 3,284 | 0 | 16.89 s |

Nov 1, memories off, one new chat with 10 short messages, `screenshots/2026-11-01-memories-off-*.png`
(load ~3.9, so times are a bit slow; cache hit or miss isn't affected):

| message | prompt tokens | from cache | first word |
|---|---|---|---|
| 2 | 3,914 | 0 | 27.70 s |
| 3 | 3,981 | 3,960 (99%) | 1.47 s |
| 5 | 4,034 | 4,013 (99%) | 0.63 s |
| 10 | 4,218 | 4,196 (99%) | 0.61 s |

Nov 1, other features (memories off, load ~4):

| test | prompt tokens | from cache | first word | total |
|---|---|---|---|---|
| PDF attached, "summarize in 3 bullets" (`2026-11-01-pdf.png`) | 14,457 | 0 | 96.68 s | 109.74 s |
| dictated short question (`2026-11-01-dictation.png`) | 3,322 | 0 | 30.55 s | 31.34 s |
| Wi-Fi off, short question (`2026-11-01-offline.png`) | 3,283 | 0 | 19.70 s | 20.44 s |

## Through Loci's API (`results/20261030-160651-all.jsonl`)

| test | result |
|---|---|
| first word, short question, warm | median 0.36 s (5 reps), 16.7 tok/s |
| prefill | 174 → 4,472 prompt tokens: 0.97 → 22.7 s, linear, ~196 tok/s |
| same 3,284-token prompt twice | 16.63 s, then 16.64 s (no reuse) |
| multi-turn | turn 1 0.40 s → turn 10 10.23 s (2,028 tokens): +4.95 ms per history token, a full re-prefill every turn |
| long answer | 16.7 tok/s, about 3% slower by the end |
| determinism, temperature 0 | 5/5 identical |

Cold start (`results/20261030-155925-cold.jsonl`): the first request after each of 3 launches ran at ~1.4 tok/s
(32 tokens in ~24 s). The second request ran at 15.6–15.9 tok/s.

Idle (`results/20261101-201233-idle*.jsonl`, `--force`, load ~3.9): memory footprint dropped from ~5.7 GB to 0.9 GB
about 120 s after each request, so the model unloads after ~2 min idle. With the file still in the OS cache, reloading
added ~0.4 s (first word 0.34 → 0.71 s). After ~1 h idle on Oct 30 it took 8.23 s.

## Stock llama.cpp on the same file

`results/20261030-161505-llama-bench.md` (flash attention on, 3 reps):

| test | tok/s |
|---|---|
| pp512 | 215.6 |
| pp2048 | 208.4 |
| pp4096 | 203.2 |
| tg128 | 16.77 |

Multi-turn with thinking off, same test as Loci's (`results/20261030-184810-multiturn.jsonl` default cache,
`results/20261030-185026-multiturn.jsonl` `--swa-full`, `--force`, load 3.3–4):

| server | turn 1 | turn 10 |
|---|---|---|
| Loci 0.1.28 | 0.40 s | 10.23 s |
| llama-server, default cache | 0.48 s | 1.50 s |
| llama-server, `--swa-full` | 0.33 s | 1.27 s |

Earlier llama-server cache runs (`results/20261030-18[23]*`) ran while the machine was busy, and the first multi-turn
ones had empty assistant turns (the model's thinking used up the answer), so I reran them as above.

## Does cache reuse change answers? (`bench/cache_equiv.py`)

Five conversations, greedy, 128 tokens each, cached prefill vs full recompute:

| run | prompt | reused | prefill without → with cache | output tokens | max logprob shift | max KL |
|---|---|---|---|---|---|---|
| `cache_equiv-20261030-183956.jsonl` | ~0.9k | 883–906 | 4.43 → 0.37 s | identical 5/5 | 0.0018 | 2e-6 |
| `cache_equiv-20261030-185245.jsonl` (`--swa-full`) | ~2.5k | 2,384–2,489 | ~11.8 → ~0.20 s | identical 5/5 | 0.015 | 1.9e-4 |

Full recompute vs full recompute was bit-identical both times, so the small shifts come from the cached path.
