import json, statistics, os, re
from pathlib import Path

FB = Path("data/feedback/feedback.jsonl")
ENV = Path(".env")

def load_feedback():
    if not FB.exists(): return []
    for line in FB.read_text(encoding="utf-8").splitlines():
        try: yield json.loads(line)
        except: pass

def window(feeds, days=7):
    # 简化：不按时间窗口也可，先全量
    return list(feeds)

def compute_uprate(items):
    up = sum(1 for x in items if x.get("label")=="up")
    down = sum(1 for x in items if x.get("label")=="down")
    return up / max(1, up+down)

def rewrite_env(**kv):
    old = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    for k,v in kv.items():
        if f"{k}=" in old:
            old = re.sub(rf"^{k}=.*$", f"{k}={v}", old, flags=re.M)
        else:
            old += f"\n{k}={v}"
    ENV.write_text(old, encoding="utf-8")

if __name__ == "__main__":
    data = list(window(load_feedback(), days=7))
    rate = compute_uprate(data)
    # 读取当前阈值（也可用 settings）
    min_vec = float(os.getenv("MIN_VEC_SIM", "0.40"))
    min_bm25 = float(os.getenv("MIN_BM25_SCORE", "5"))
    # 简单策略
    if rate < 0.7:
        min_vec = min(min_vec + 0.02, 0.60)
        min_bm25 = min_bm25 + 0.5
    elif rate > 0.8:
        min_vec = max(min_vec - 0.02, 0.30)
        min_bm25 = max(min_bm25 - 0.5, 3.0)
    # 写回 .env（下次进程重启或热加载读取）
    rewrite_env(MIN_VEC_SIM=min_vec, MIN_BM25_SCORE=min_bm25)
    print({"up_rate": rate, "MIN_VEC_SIM": min_vec, "MIN_BM25_SCORE": min_bm25})