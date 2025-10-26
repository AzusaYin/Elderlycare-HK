from pathlib import Path
import json, collections, re

LOG = Path("data/logs/chat_usage.jsonl")
if not LOG.exists():
    print("No log file found.")
    raise SystemExit

qs = []
clar = cnt_nohit = 0
for line in LOG.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    qs.append(r["user_query"])
    if r.get("context_hits", 0) == 0: cnt_nohit += 1
    if r.get("clarified"): clar += 1

# 统计最常见的问题关键词
words = re.findall(r"\b[A-Za-z]{3,}\b", " ".join(qs))
freq = collections.Counter(words).most_common(20)

print("Top 20 keywords:", freq)
print(f"Total Qs: {len(qs)} | Clarifications: {clar} | No hits: {cnt_nohit}")