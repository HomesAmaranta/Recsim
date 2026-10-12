"""LLM 客户端：OpenAI 兼容接口 + 线程池并发 + 失败重试。"""
import json, time, urllib.request, urllib.error, threading
import concurrent.futures as cf
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "fail": 0}
_lock = threading.Lock()

def chat(messages, max_tokens=512, temperature=0.0, retries=3, logprobs=False):
    """logprobs=True 时返回 (文本, {token: logprob} 首 token 的 top-k)，否则只返回文本。

    首 token 的 A/B 对数概率差就是 LLM 的连续置信度，比「只输出一个字母」的
    1 bit 投票信息量高得多，且不额外花 token。
    """
    p = {"model": C.LLM_MODEL, "messages": messages,
         "max_tokens": max_tokens, "temperature": temperature}
    if logprobs:
        p["logprobs"] = True; p["top_logprobs"] = 10
    body = json.dumps(p).encode()
    last = None
    for attempt in range(retries):
        try:
            h = {"Content-Type": "application/json"}
            if C.LLM_API_KEY:
                h["Authorization"] = "Bearer " + C.LLM_API_KEY
            req = urllib.request.Request(C.LLM_ENDPOINT, data=body, headers=h)
            with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT) as r:
                d = json.loads(r.read())
            u = d.get("usage") or {}
            with _lock:
                _stats["calls"] += 1
                _stats["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                _stats["completion_tokens"] += u.get("completion_tokens", 0) or 0
                if attempt:
                    _stats["retries"] += attempt
            ch = d["choices"][0]
            txt = ch["message"]["content"]
            if not logprobs:
                return txt
            lp = {}
            try:
                first = (ch.get("logprobs") or {}).get("content") or []
                for e in (first[0].get("top_logprobs") or []):
                    lp.setdefault(e["token"].strip(), e["logprob"])
            except Exception:
                pass
            return txt, lp
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    with _lock:
        _stats["fail"] += 1
    raise RuntimeError(f"LLM 调用失败 {retries} 次: {last}")

def chat_many(jobs, max_tokens=512, temperature=0.0, concurrency=None, desc="", logprobs=False):
    """jobs: [(key, messages), ...] -> {key: text or None}；logprobs=True 时值为 (text, lp)"""
    concurrency = concurrency or C.LLM_CONCURRENCY
    out, done, t0, n = {}, [0], time.time(), len(jobs)
    def run(item):
        k, msgs = item
        try:
            r = chat(msgs, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs)
        except Exception as e:
            r = (None, {}) if logprobs else None
        with _lock:
            done[0] += 1
            if done[0] % max(1, n // 20) == 0 or done[0] == n:
                el = time.time() - t0
                print(f"  {desc} {done[0]}/{n}  {el:.0f}s  {done[0]/el:.1f} req/s", flush=True)
        return k, r
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for k, r in ex.map(run, jobs):
            out[k] = r
    return out

def stats():
    return dict(_stats)

def print_stats(tag=""):
    s = _stats
    print(f"[LLM 用量{tag}] 调用 {s['calls']}  prompt {s['prompt_tokens']:,} tok  "
          f"completion {s['completion_tokens']:,} tok  重试 {s['retries']}  失败 {s['fail']}")

def extract_json(text):
    if not text:
        return None
    try:
        s = text[text.index("{"):text.rindex("}") + 1]
        return json.loads(s)
    except Exception:
        return None
