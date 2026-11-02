#!/usr/bin/env python3
"""Time Loci's local model through its agent API (OpenAI-compatible, streaming).

Loci's direct-download Mac build serves the model on 127.0.0.1 (port 5624 when free). The URL
and a local token are in ~/Library/Application Support/LociRust/agent-api.json. Turn the API on
in Settings > Advanced > Agent Access. The token is read from that file and never printed.

Each request is streamed and logged as one JSON line with:
  ttft_s      seconds to the first visible answer text (Loci's "first word")
  ttf_any_s   seconds to the first streamed token of any kind, hidden reasoning included
              (prefill, plus any model load that happens inside the request)
  total_s     seconds to the end of the stream
  decode_tps  (tokens - 1) / (last token time - first token time)
Token counts come from the server's `usage` (or llama-server style `timings`) when it sends
them. Otherwise rows fall back to stream chunks, and `chars4_tps` uses characters / 4, the rule
Loci's speed page uses on phones. `token_source` says which one a row used.

Tests ("all" runs probe, ttft, prefill, cache, multiturn, decode, determinism):
  probe        list models, send one short request, show which fields come back
  ttft         short new prompts, warm, repeated
  prefill      prompt-length sweep, unique text each time -> prefill tok/s and fixed cost
  cache        same long prompt twice, then the same text with a new question
  multiturn    a growing conversation -> does time to first token grow with history?
  decode       one long answer -> tok/s early vs late in the answer
  determinism  the same prompt repeated at temperature 0 -> identical text?
  idle         warm request, wait, request again -> does Loci unload the model when idle?
  sustained    back-to-back long answers for N minutes -> thermal throttling
  cold         QUITS Loci, relaunches it, times API-ready and the first two requests (needs --yes)

A background thread samples Loci's memory during each test: footprint (what Activity Monitor
shows, via `top`) and RSS (via `ps`). Run on a quiet Mac; the script refuses if load or swap
say otherwise, unless you pass --force.

Stdlib only, Python 3.9+. Examples:
  python3 loci_bench.py probe
  python3 loci_bench.py all
  python3 loci_bench.py prefill --sizes 256,1024,4096,8192
  python3 loci_bench.py cold --runs 3 --purge --yes
  python3 loci_bench.py all --extra-json '{"chat_template_kwargs": {"enable_thinking": false}}'
  python3 loci_bench.py multiturn --base-url http://127.0.0.1:8090/v1 --no-auth --pid "$(pgrep -x llama-server)"
"""

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import random
import re
import statistics
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path

API_FILE = Path.home() / "Library/Application Support/LociRust/agent-api.json"
DEFAULT_BASE = "http://127.0.0.1:5624/v1"
BUNDLE_ID = "ai.askloci.rust"
PROCESS = "loci"
APP_PLIST = "/Applications/Loci.app/Contents/Info.plist"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"

WORDS = (
    "river model window garden signal market planet orange silver music teacher doctor village "
    "bridge morning winter paper number system ocean forest mountain city train letter coffee "
    "street summer engine battery kitchen school island camera office story table"
).split()

SHORT_QS = [
    "In two sentences, what is a hash table?",
    "In two sentences, why do leaves change colour in autumn?",
    "In two sentences, what does a compiler do?",
    "In two sentences, how does a refrigerator keep food cold?",
    "In two sentences, what is inflation?",
    "In two sentences, why is the sea salty?",
    "In two sentences, what is a black hole?",
    "In two sentences, how do vaccines work?",
]

TURN_QS = [
    "I'm planning a week-long trip to Japan in spring. Which cities should I consider?",
    "How should I split the week between those cities?",
    "What's the best way to travel between them?",
    "Roughly what should I budget per day, not counting flights?",
    "Which foods should I try in each city?",
    "What etiquette mistakes should I avoid?",
    "Suggest a rainy-day plan for the second city.",
    "What should I pack for the weather at that time of year?",
    "Which Japanese phrases would be most useful?",
    "Summarize the whole plan we discussed as a short list.",
    "If I add two more days, where would you go?",
    "What would you cut if I only had five days?",
]


def nonce():
    return "%08x" % random.getrandbits(32)


def filler(n_words, seed):
    """Deterministic plain-English-ish text, about one token per word."""
    rng = random.Random(seed)
    out = []
    for i in range(n_words):
        w = rng.choice(WORDS)
        out.append(w.capitalize() if i % 12 == 0 else w)
        if i % 12 == 11:
            out[-1] += "."
    return " ".join(out)


def visible_text(raw):
    """Answer text with a leading <think>...</think> block removed ("" while still thinking)."""
    s = raw.lstrip()
    if s.startswith("<think>"):
        end = s.find("</think>")
        return "" if end < 0 else s[end + len("</think>"):].strip()
    if "<think>".startswith(s):  # empty, or a partial tag so far
        return ""
    return raw.strip()


def inline_think_chars(raw):
    s = raw.lstrip()
    if not s.startswith("<think>"):
        return 0
    end = s.find("</think>")
    return len(s) - 7 if end < 0 else end - 7


# ---------------------------------------------------------------- endpoint + client


class Endpoint:
    """Reads Loci's endpoint file on every call, since the token can change when Loci restarts."""

    def __init__(self, api_file, base_override=None, use_token=True):
        self.api_file = Path(api_file)
        self.base_override = base_override
        self.use_token = use_token

    def read(self):
        info = {}
        if self.api_file.exists():
            try:
                info = json.loads(self.api_file.read_text())
            except ValueError:
                info = {}
        if not isinstance(info, dict):
            info = {}
        token = next((info[k] for k in ("token", "api_key", "apiKey", "key", "bearer")
                      if isinstance(info.get(k), str)), None)
        token_file = self.api_file.with_name("agent-api.token")  # Loci also keeps the bare token here
        if token is None and token_file.exists():
            token = token_file.read_text().strip() or None
        if not self.use_token:
            token = None
        base = self.base_override
        if not base:
            for k in ("openai_base_url", "openai_url", "base_url", "baseUrl", "api_url", "http_url", "url"):
                v = info.get(k)
                if isinstance(v, str) and v.startswith("http") and "mcp" not in urllib.parse.urlsplit(v).path:
                    base = v
                    break
        if not base and isinstance(info.get("mcp_url"), str):
            u = urllib.parse.urlsplit(info["mcp_url"])
            base = f"{u.scheme}://{u.netloc}/v1"
        if not base and info.get("port"):
            base = f"http://127.0.0.1:{info['port']}/v1"
        base = (base or DEFAULT_BASE).rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if urllib.parse.urlsplit(base).path in ("", "/"):
            base += "/v1"
        return base, token, sorted(info)


class ApiError(RuntimeError):
    pass


class Api:
    def __init__(self, endpoint, model, extra, timeout, include_usage=True):
        self.endpoint = endpoint
        self.model = model
        self.extra = extra
        self.timeout = timeout
        self.include_usage = include_usage

    def _open(self, accept):
        base, token, _ = self.endpoint.read()
        u = urllib.parse.urlsplit(base)
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=self.timeout)
        headers = {"Content-Type": "application/json", "Accept": accept}
        if token:
            headers["Authorization"] = "Bearer " + token
        return conn, u.path.rstrip("/"), headers

    def get(self, path):
        conn, prefix, headers = self._open("application/json")
        try:
            conn.request("GET", prefix + path, headers=headers)
            r = conn.getresponse()
            body = r.read()
            try:
                return r.status, (json.loads(body) if body else None)
            except ValueError:
                return r.status, body[:300].decode("utf-8", "replace")
        finally:
            conn.close()

    def chat(self, messages, max_tokens, temperature=0.0, keep_times=False):
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": max_tokens, "temperature": temperature}
        if self.include_usage:
            body["stream_options"] = {"include_usage": True}
        body.update(self.extra)
        conn, prefix, headers = self._open("text/event-stream")
        try:
            t0 = time.perf_counter()
            conn.request("POST", prefix + "/chat/completions", body=json.dumps(body).encode(), headers=headers)
            r = conn.getresponse()
            t_headers = time.perf_counter() - t0
            if r.status != 200:
                err = r.read()[:400].decode("utf-8", "replace")
                if self.include_usage and r.status in (400, 422) and "stream_options" in err:
                    self.include_usage = False  # server doesn't take it; retry once without
                    return self.chat(messages, max_tokens, temperature, keep_times)
                raise ApiError(f"HTTP {r.status}: {err}")
            return self._read_stream(r, t0, t_headers, keep_times)
        finally:
            conn.close()

    @staticmethod
    def _read_stream(r, t0, t_headers, keep_times):
        content, reasoning, times = [], [], []
        t_any = t_visible = t_last = None
        usage = timings = finish = None
        n_content = n_reasoning = 0
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            now = time.perf_counter() - t0
            usage = obj.get("usage") or usage
            timings = obj.get("timings") or timings
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                rs = d.get("reasoning_content") or d.get("reasoning") or d.get("thinking")
                c = d.get("content")
                if rs:
                    reasoning.append(rs)
                    n_reasoning += 1
                if c:
                    content.append(c)
                    n_content += 1
                if rs or c:
                    t_any = now if t_any is None else t_any
                    t_last = now
                    if keep_times:
                        times.append(round(now, 4))
                    if c and t_visible is None and visible_text("".join(content)):
                        t_visible = now
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        total = time.perf_counter() - t0
        raw = "".join(content)
        text = visible_text(raw)
        reasoning_chars = sum(len(x) for x in reasoning) + inline_think_chars(raw)

        pt = ct = None
        if isinstance(usage, dict):
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(timings, dict):  # llama-server style
            pt = pt if pt is not None else timings.get("prompt_n")
            ct = ct if ct is not None else timings.get("predicted_n")
        n_tok, source = (ct, "usage") if ct else (n_content + n_reasoning, "chunks")
        span = (t_last - t_any) if (t_any is not None and t_last is not None) else None
        decode = (n_tok - 1) / span if (n_tok and n_tok > 1 and span and span > 0) else None
        chars = len(raw) + sum(len(x) for x in reasoning)
        chars4 = (chars / 4 - 1) / span if (span and span > 0 and chars > 8) else None
        return {
            "ttft_s": t_visible, "ttf_any_s": t_any, "headers_s": t_headers, "total_s": total,
            "prompt_tokens": pt, "completion_tokens": ct, "token_source": source,
            "decode_tps": decode, "chars4_tps": chars4,
            "content_chunks": n_content, "reasoning_chunks": n_reasoning, "reasoning_chars": reasoning_chars,
            "finish_reason": finish, "usage": usage, "server_timings": timings,
            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "chunk_times": times if keep_times else None,
        }


# ---------------------------------------------------------------- machine + memory


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def pids_named(name):
    return [int(p) for p in sh("pgrep", "-x", name).split()]


_UNIT_MB = {"B": 1 / 2**20, "K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 2.0**20}


def footprint_mb(pid):
    lines = sh("top", "-l", "1", "-pid", str(pid), "-stats", "mem").strip().splitlines()
    m = re.match(r"^\s*([\d.]+)([BKMGT])[+-]?\s*$", lines[-1]) if lines else None
    return float(m.group(1)) * _UNIT_MB[m.group(2)] if m else None


def rss_mb(pid):
    out = sh("ps", "-o", "rss=", "-p", str(pid)).strip()
    return int(out) / 1024 if out.isdigit() else None


class MemSampler:
    """Samples memory of every process named `loci` (or one --pid) until stopped."""

    def __init__(self, pid_fn, interval, out_file):
        self.pid_fn, self.interval, self.out_file = pid_fn, interval, out_file
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self, test):
        self.test = test
        if self.interval > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            pids = self.pid_fn()
            fps = [f for f in (footprint_mb(p) for p in pids) if f is not None]
            rss = [r for r in (rss_mb(p) for p in pids) if r is not None]
            s = {"t": round(time.time(), 2), "test": self.test, "pids": pids,
                 "footprint_mb": round(sum(fps), 1) if fps else None, "rss_mb": round(sum(rss), 1) if rss else None}
            self.samples.append(s)
            with open(self.out_file, "a") as f:
                f.write(json.dumps(s) + "\n")
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        fp = [s["footprint_mb"] for s in self.samples if s["footprint_mb"] is not None]
        rs = [s["rss_mb"] for s in self.samples if s["rss_mb"] is not None]
        if not fp and not rs:
            return None
        return {"samples": len(self.samples),
                "footprint_mb": {"start": fp[0], "min": min(fp), "peak": max(fp), "end": fp[-1]} if fp else None,
                "rss_mb": {"start": rs[0], "min": min(rs), "peak": max(rs), "end": rs[-1]} if rs else None}


def machine_state():
    swap = re.search(r"used = ([\d.]+)M", sh("sysctl", "-n", "vm.swapusage"))
    batt = sh("pmset", "-g", "batt").splitlines()
    return {"load1": round(os.getloadavg()[0], 2),
            "swap_used_mb": float(swap.group(1)) if swap else None,
            "power": batt[0] if batt else None,
            "therm": sh("pmset", "-g", "therm").strip().splitlines()[:3]}


def loci_version():
    v = sh("plutil", "-extract", "CFBundleShortVersionString", "raw", "-o", "-", APP_PLIST).strip()
    return v or None


# ---------------------------------------------------------------- helpers


class Ctx:
    def __init__(self, api, endpoint, args, out):
        self.api, self.endpoint, self.args, self.out = api, endpoint, args, out

    def write(self, row):
        with open(self.out, "a") as f:
            f.write(json.dumps(row) + "\n")


def fmt(v, nd=2):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def ask(ctx, test, label, messages, max_tokens, temperature=0.0, keep_times=False):
    when = dt.datetime.now().isoformat(timespec="seconds")
    try:
        res = ctx.api.chat(messages, max_tokens, temperature, keep_times)
    except (ApiError, OSError, http.client.HTTPException) as e:
        res = {"error": f"{type(e).__name__}: {e}"}
    res.update({"kind": "request", "test": test, "label": label, "time": when, "max_tokens": max_tokens,
                "prompt_chars": sum(len(m["content"]) for m in messages)})
    ctx.write(res)
    if "error" in res:
        print(f"  {label:<26} ERROR {res['error'][:200]}")
    else:
        print(f"  {label:<26} prompt {fmt(res['prompt_tokens']):>6} tok | first token {fmt(res['ttf_any_s']):>6} s"
              f" | first text {fmt(res['ttft_s']):>6} s | {fmt(res['decode_tps'], 1):>6} tok/s ({res['token_source']})"
              f" | total {fmt(res['total_s']):>6} s | {res['finish_reason'] or '-'}")
    return res


def ok(rows):
    return [r for r in rows if "error" not in r]


def stats(rows, keys=("ttf_any_s", "ttft_s", "total_s", "decode_tps", "prompt_tokens", "completion_tokens")):
    out = {"n": len(ok(rows)), "errors": len(rows) - len(ok(rows))}
    for k in keys:
        v = [r[k] for r in ok(rows) if r.get(k) is not None]
        if v:
            out[k] = {"median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3)}
    return out


def fit(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 2:
        return None
    mx = statistics.fmean(x for x, _ in pts)
    my = statistics.fmean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return {"slope_ms_per_token": round(b * 1000, 4), "intercept_s": round(my - b * mx, 3),
            "prefill_tps": round(1 / b, 1) if b > 0 else None, "points": len(pts)}


def prompt_size(r):
    """Prompt tokens from the server, else a chars/4 estimate."""
    return r.get("prompt_tokens") if r.get("prompt_tokens") is not None else r["prompt_chars"] / 4


def user(text):
    return [{"role": "user", "content": text}]


def warmup(ctx, test):
    """One short request first, so a model load doesn't land on the first measured point."""
    ask(ctx, test, "warmup", user(SHORT_QS[-1] + f" (ref {nonce()})"), 8)


# ---------------------------------------------------------------- tests


def test_probe(ctx):
    base, token, keys = ctx.endpoint.read()
    print(f"  endpoint {base} | token {'found' if token else 'MISSING'} | keys in endpoint file: {keys or 'no file'}")
    if not ctx.endpoint.api_file.exists():
        print(f"  {ctx.endpoint.api_file} doesn't exist yet: turn on Settings > Advanced > Agent Access in Loci")
    try:
        st, models = ctx.api.get("/models")
    except (OSError, http.client.HTTPException) as e:
        print(f"  GET /models failed: {e}. Is Loci running with Agent Access on?")
        return {"error": str(e)}
    ids = [m.get("id") for m in models.get("data", [])] if isinstance(models, dict) else models
    print(f"  GET /models -> {st}: {ids}")
    if st == 401:
        print("  token rejected: toggle Settings > Advanced > Agent Access off and on, then retry")
    r = ask(ctx, "probe", "short", user("Reply with one short sentence: what is 2 + 2?"), 64)
    if "error" not in r:
        print(f"  server sent: usage={bool(r['usage'])} timings={bool(r['server_timings'])} "
              f"reasoning={r['reasoning_chars'] > 0} finish={r['finish_reason']}")
        print(f"  answer: {r['text'][:200]!r}")
        if r["reasoning_chars"] > 0 and r["ttft_s"] is None:
            print("  note: the model spent the whole budget reasoning. Raise max_tokens, or try "
                  "--extra-json '{\"chat_template_kwargs\": {\"enable_thinking\": false}}'")
    return {"models_status": st, "models": ids, "loci_version": loci_version()}


def test_ttft(ctx):
    ask(ctx, "ttft", "warmup", user(SHORT_QS[0] + f" (ref {nonce()})"), 16)
    rows = [ask(ctx, "ttft", f"rep{i + 1}", user(SHORT_QS[(i + 1) % len(SHORT_QS)] + f" (ref {nonce()})"), 128)
            for i in range(ctx.args.reps)]
    s = stats(rows)
    print(f"  median: first token {s.get('ttf_any_s', {}).get('median')} s, first text "
          f"{s.get('ttft_s', {}).get('median')} s, decode {s.get('decode_tps', {}).get('median')} tok/s")
    return s


def test_prefill(ctx):
    warmup(ctx, "prefill")
    rows = []
    for rep in range(ctx.args.prefill_reps):
        order = ctx.args.sizes if rep % 2 == 0 else ctx.args.sizes[::-1]
        for n in order:
            text = (f"Reference {nonce()}.\n" + filler(n, seed=rep * 100003 + n)
                    + "\n\nIn one sentence, what is the text above about?")
            rows.append(ask(ctx, "prefill", f"{n} words", user(text), 8))
    good = [r for r in ok(rows) if r.get("ttf_any_s") is not None]
    f = fit([prompt_size(r) for r in good], [r["ttf_any_s"] for r in good])
    src = "server prompt_tokens" if good and good[0].get("prompt_tokens") is not None else "chars/4 estimate"
    if f:
        print(f"  fit: prefill ≈ {f['prefill_tps']} tok/s, fixed cost ≈ {f['intercept_s']} s per request "
              f"(x = {src}, {f['points']} points)")
    return {"fit": f, "x": src, "by_size": {str(n): stats([r for r in rows if r["label"] == f"{n} words"],
                                                           ("ttf_any_s", "prompt_tokens")) for n in ctx.args.sizes}}


def test_cache(ctx):
    n = ctx.args.cache_words
    base = f"Reference {nonce()}.\n" + filler(n, seed=4242)
    q1 = "\n\nIn one sentence, what is the text above about?"
    q2 = "\n\nName three words that appear often in the text above."
    warmup(ctx, "cache")
    rows = {
        "first": ask(ctx, "cache", "first", user(base + q1), 8),
        "identical": ask(ctx, "cache", "identical", user(base + q1), 8),
        "same_text_new_question": ask(ctx, "cache", "same_text_new_question", user(base + q2), 8),
        "control_new_text": ask(ctx, "cache", "control_new_text",
                                user(f"Reference {nonce()}.\n" + filler(n, seed=4343) + q1), 8),
    }
    ref = rows["control_new_text"].get("ttf_any_s")
    out = {k: v.get("ttf_any_s") for k, v in rows.items()}
    if ref:
        for k in ("identical", "same_text_new_question"):
            if out[k] is not None:
                out[k + "_vs_control"] = round(out[k] / ref, 3)
        print(f"  first token vs a fresh prompt of the same length: identical {out.get('identical_vs_control')}x, "
              f"same text + new question {out.get('same_text_new_question_vs_control')}x "
              "(well below 1 means the prefix was reused)")
    return out


def test_multiturn(ctx):
    warmup(ctx, "multiturn")
    msgs, rows = [], []
    for i in range(ctx.args.turns):
        q = TURN_QS[i % len(TURN_QS)] + (f" (ref {nonce()})" if i == 0 else "")
        msgs.append({"role": "user", "content": q})
        r = ask(ctx, "multiturn", f"turn {i + 1}", list(msgs), 200)
        rows.append(r)
        msgs.append({"role": "assistant", "content": r.get("text") or "(no answer)"})
    good = [r for r in ok(rows) if r.get("ttf_any_s") is not None]
    f = fit([prompt_size(r) for r in good], [r["ttf_any_s"] for r in good])
    if f:
        print(f"  first token grows {f['slope_ms_per_token']} ms per prompt token over the conversation "
              "(about 0 means the history is reused; about 1000/prefill-rate means it's recomputed every turn)")
    return {"fit": f, "turns": [{"turn": i + 1, "prompt_tokens": r.get("prompt_tokens"),
                                 "ttf_any_s": r.get("ttf_any_s"), "ttft_s": r.get("ttft_s")}
                                for i, r in enumerate(rows)]}


def test_decode(ctx):
    q = ("Write a detailed explanation, about 800 words, of how a CPU cache hierarchy works: L1, L2 and L3, "
         "cache lines, associativity, write policies, and why misses are expensive. Use headings.")
    r = ask(ctx, "decode", "long answer", user(q + f" (ref {nonce()})"), ctx.args.decode_tokens, keep_times=True)
    out = {"decode_tps": r.get("decode_tps"), "completion_tokens": r.get("completion_tokens")}
    t = r.get("chunk_times") or []
    if len(t) >= 40:
        k = len(t) // 4
        out["chunks_per_s_first_quarter"] = round((k - 1) / (t[k - 1] - t[0]), 2)
        out["chunks_per_s_last_quarter"] = round((k - 1) / (t[-1] - t[-k]), 2)
        print(f"  chunks/s: first quarter {out['chunks_per_s_first_quarter']}, "
              f"last quarter {out['chunks_per_s_last_quarter']} (a drop = KV growth or heat)")
    return out


def test_determinism(ctx):
    q = "In about 120 words, explain why the sky is blue."
    rows = [ask(ctx, "determinism", f"rep{i + 1}", user(q), 200, temperature=0.0) for i in range(ctx.args.det_reps)]
    texts = [r["text"] for r in ok(rows)]
    if not texts:
        return {"error": "no answers"}
    diffs = []
    for t in texts[1:]:
        if t != texts[0]:
            diffs.append(next((i for i, (a, b) in enumerate(zip(t, texts[0])) if a != b), min(len(t), len(texts[0]))))
    same = sum(t == texts[0] for t in texts)
    print(f"  {same}/{len(texts)} answers identical to the first at temperature 0"
          + (f"; first differing character at {diffs}" if diffs else "")
          + " (repeats may also hit the prompt cache, so this checks cached = uncached too)")
    return {"identical": same, "total": len(texts), "first_diff_chars": diffs}


def test_idle(ctx):
    ask(ctx, "idle", "warm", user(SHORT_QS[0] + f" (ref {nonce()})"), 16)
    rows = []
    for w in ctx.args.waits:
        print(f"  idle {w} s (watch memory in the samples) ...", flush=True)
        time.sleep(w)
        rows.append(ask(ctx, "idle", f"after {w} s idle", user(SHORT_QS[1] + f" (ref {nonce()})"), 16))
    return {"after_wait_ttf_any_s": {str(w): r.get("ttf_any_s") for w, r in zip(ctx.args.waits, rows)}}


def test_sustained(ctx):
    q = "Write a long, detailed story, at least 600 words, about a lighthouse keeper during a storm."
    end = time.time() + ctx.args.minutes * 60
    rows, i = [], 0
    while time.time() < end:
        i += 1
        rows.append(ask(ctx, "sustained", f"run {i}", user(q + f" (ref {nonce()})"), 512))
    rates = [r["decode_tps"] for r in ok(rows) if r.get("decode_tps")]
    out = {"runs": len(rows), "decode_tps": [round(x, 2) for x in rates], "therm_after": machine_state()["therm"]}
    if len(rates) >= 4:
        print(f"  decode tok/s: first two {rates[:2]}, last two {rates[-2:]}")
    return out


def test_cold(ctx):
    if not ctx.args.yes:
        print("  cold quits Loci and relaunches it (the model reloads, several GB of memory). "
              "Re-run with --yes when the Mac is free.")
        return {"skipped": "needs --yes"}
    results = []
    for run in range(ctx.args.runs):
        if pids_named(PROCESS):
            subprocess.run(["osascript", "-e", f'tell application id "{BUNDLE_ID}" to quit'], capture_output=True)
            t = time.time()
            while pids_named(PROCESS) and time.time() - t < 60:
                time.sleep(0.5)
            if pids_named(PROCESS):
                print("  Loci didn't quit within 60 s. Stopping here (not force-killing it).")
                break
        if ctx.args.purge:
            subprocess.run(["sudo", "purge"])  # drops the file cache so weights come from disk
        time.sleep(3)
        t0 = time.perf_counter()
        if subprocess.run(["open", "-b", BUNDLE_ID], capture_output=True).returncode != 0:
            print(f"  couldn't launch {BUNDLE_ID}. Is Loci installed in /Applications?")
            break
        ready = None
        while time.perf_counter() - t0 < 300:
            try:
                st, _ = ctx.api.get("/models")
                if st == 200:
                    ready = time.perf_counter() - t0
                    break
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.25)
        if ready is None:
            print("  API didn't answer within 300 s of launch. Is Agent Access on?")
            break
        sent = time.perf_counter() - t0
        first = ask(ctx, "cold", f"run {run + 1} first", user(SHORT_QS[2] + f" (ref {nonce()})"), 32)
        second = ask(ctx, "cold", f"run {run + 1} second", user(SHORT_QS[3] + f" (ref {nonce()})"), 32)
        row = {"run": run + 1, "api_ready_s": round(ready, 2),
               "launch_to_first_token_s": round(sent + first["ttf_any_s"], 2) if first.get("ttf_any_s") else None,
               "first_ttf_any_s": first.get("ttf_any_s"), "second_ttf_any_s": second.get("ttf_any_s")}
        print(f"  run {run + 1}: API ready {row['api_ready_s']} s, launch → first token "
              f"{row['launch_to_first_token_s']} s, first request {fmt(row['first_ttf_any_s'])} s, "
              f"second {fmt(row['second_ttf_any_s'])} s")
        results.append(row)
    return {"runs": results}


TESTS = {"probe": test_probe, "ttft": test_ttft, "prefill": test_prefill, "cache": test_cache,
         "multiturn": test_multiturn, "decode": test_decode, "determinism": test_determinism,
         "idle": test_idle, "sustained": test_sustained, "cold": test_cold}
ALL = ["probe", "ttft", "prefill", "cache", "multiturn", "decode", "determinism"]


def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=list(TESTS) + ["all"])
    ap.add_argument("--base-url", help="override, e.g. http://127.0.0.1:5624/v1")
    ap.add_argument("--api-file", default=str(API_FILE))
    ap.add_argument("--model", default="default", help="model id (default: Loci's current model)")
    ap.add_argument("--extra-json", default="{}", help="fields merged into every request body")
    ap.add_argument("--no-usage", action="store_true", help="don't send stream_options.include_usage")
    ap.add_argument("--no-auth", action="store_true",
                    help="don't send Loci's token (use with --base-url for llama-server / mlx_lm.server)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=str(OUT_DIR), help="results folder")
    ap.add_argument("--pid", type=int, help="sample memory of this pid instead of processes named 'loci'")
    ap.add_argument("--mem-interval", type=float, default=1.0, help="seconds between memory samples (0 = off)")
    ap.add_argument("--reps", type=int, default=5, help="ttft repeats")
    ap.add_argument("--sizes", type=ints, default=[128, 512, 1024, 2048, 4096], help="prefill sizes in words")
    ap.add_argument("--prefill-reps", type=int, default=2)
    ap.add_argument("--cache-words", type=int, default=3000)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--decode-tokens", type=int, default=1024)
    ap.add_argument("--det-reps", type=int, default=5)
    ap.add_argument("--waits", type=ints, default=[60, 300, 900], help="idle waits in seconds")
    ap.add_argument("--minutes", type=float, default=5, help="sustained duration")
    ap.add_argument("--runs", type=int, default=3, help="cold launches")
    ap.add_argument("--purge", action="store_true", help="cold: `sudo purge` before each launch")
    ap.add_argument("--yes", action="store_true", help="cold: allow quitting and relaunching Loci")
    ap.add_argument("--force", action="store_true", help="run even if the Mac looks busy")
    args = ap.parse_args()

    endpoint = Endpoint(args.api_file, args.base_url, use_token=not args.no_auth)
    api = Api(endpoint, args.model, json.loads(args.extra_json), args.timeout, include_usage=not args.no_usage)
    state = machine_state()
    print(f"machine: load {state['load1']}, swap used {state['swap_used_mb']} MB, {state['power']}")
    if (state["load1"] > 3 or (state["swap_used_mb"] or 0) > 1024) and not args.force:
        print("This Mac looks busy (load > 3 or swap > 1 GB), so timings would be skewed. "
              "Finish other work (reboot clears swap) or pass --force.")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ctx = Ctx(api, endpoint, args, out_dir / f"{stamp}-{args.test}.jsonl")
    mem_file = out_dir / f"{stamp}-{args.test}-memory.jsonl"
    ctx.write({"kind": "header", "time": stamp, "args": vars(args), "machine": state,
               "endpoint": endpoint.read()[0], "loci_version": loci_version()})
    pid_fn = (lambda: [args.pid]) if args.pid else (lambda: pids_named(PROCESS))

    for name in (ALL if args.test == "all" else [args.test]):
        print(f"\n== {name} ==")
        sampler = MemSampler(pid_fn, args.mem_interval, mem_file).start(name)
        try:
            result = TESTS[name](ctx)
        except KeyboardInterrupt:
            sampler.stop()
            print("\ninterrupted")
            break
        mem = sampler.stop()
        ctx.write({"kind": "summary", "test": name, "result": result, "memory": mem, "machine_after": machine_state()})
        if mem and mem.get("footprint_mb"):
            m = mem["footprint_mb"]
            print(f"  memory (footprint): start {m['start']} MB, min {m['min']}, peak {m['peak']}, end {m['end']}"
                  + (f"; RSS peak {mem['rss_mb']['peak']} MB" if mem.get("rss_mb") else ""))
    print(f"\nresults: {ctx.out}\nmemory samples: {mem_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
