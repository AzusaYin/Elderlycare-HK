from pathlib import Path
import json, re, collections, statistics, datetime as dt

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data/logs/chat_usage.jsonl"
OUT = ROOT / "data/logs/summary.json"

if not LOG.exists():
    print("[WARN] usage log not found:", LOG)
    raise SystemExit(0)

total = clarified = nohit = 0
durs = []
ans_lens = []
kw = collections.Counter()
per_day = collections.Counter()

for line in LOG.read_text(encoding="utf-8").splitlines():
    if not line.strip(): 
        continue
    try:
        r = json.loads(line)
    except:
        continue
    total += 1
    if r.get("clarified"): clarified += 1
    if not r.get("found"): nohit += 1
    dur = r.get("duration_ms")
    if isinstance(dur, (int, float)) and dur >= 0: durs.append(float(dur))
    al = r.get("answer_len")
    if isinstance(al, int): ans_lens.append(al)
    q = r.get("user_query") or ""
    for w in re.findall(r"[A-Za-z0-9]{3,}", q):
        kw[w.lower()] += 1
    ts = r.get("ts")
    if isinstance(ts, (int, float)):
        day = dt.datetime.utcfromtimestamp(ts/1000.0).strftime("%Y-%m-%d")
        per_day[day] += 1

summary = {
    "total": total,
    "clarified": clarified,
    "nohit": nohit,
    "avg_duration_ms": (statistics.mean(durs) if durs else None),
    "p95_duration_ms": (statistics.quantiles(durs, n=20)[18] if len(durs) >= 20 else (max(durs) if durs else None)),
    "avg_answer_len": (statistics.mean(ans_lens) if ans_lens else None),
    "top_keywords": kw.most_common(50),
    "daily_series": sorted(per_day.items()),
    "generated_at": int(dt.datetime.utcnow().timestamp() * 1000)
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote {OUT}")
