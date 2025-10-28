import uvicorn
import numpy as np
import re, json, time, statistics, collections
from pathlib import Path
from typing import List, Dict, Tuple
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .schemas import ChatRequest, ChatAnswer, IngestResponse, FeedbackIn
from .rag import Index, Embedder, hybrid_retrieve, build_prompt, format_citations, ingest_corpus
from .llm_client import smartcare_chat, smartcare_chat_stream, smartcare_translate_to_en
from .security import require_bearer
from .admin_docs import router as admin_docs_router

app = FastAPI(title="ElderlyCare HK — Backend")
app.include_router(admin_docs_router)

_LOG_DIR = Path("data/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_USAGE_LOG = _LOG_DIR / "chat_usage.jsonl"
DEBUG_CLARIFY = True  # 临时打开；确认无误后改为 False

def _dbg(*a):
    if DEBUG_CLARIFY:
        print("[clarify]", *a)

def _append_jsonl_safe(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _now_ms() -> int:
    return int(time.time() * 1000)

_FB_DIR = Path("data/feedback")
_FB_DIR.mkdir(parents=True, exist_ok=True)
_FB_JSONL = _FB_DIR / "feedback.jsonl"         # 已存在：主反馈日志
_FB_METRICS = _FB_DIR / "metrics.json"         # 已存在：聚合指标
_FB_COUNTS = _FB_DIR / "penalty_counts.json"   # 新增：按页计数 { (file,page): {up,down} }
_FB_PENALTY = _FB_DIR / "penalty.json"         # 已存在于 rag；我们会即时生成/更新

def _safe_read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json_atomic(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _recompute_penalty_from_counts(counts: dict) -> dict:
    """
    把 { "file::page": {"up":x,"down":y}, ... } 转成 { "file::page": penalty_float }
    简单规则：penalty = min(1.0, BASE + STEP * max(0, down - up))
    你也可以换成更平滑的函数（例如 sigmoid）。
    """
    BASE, STEP = 0.10, 0.05   # 可按需要微调
    out = {}
    for k, v in counts.items():
        up = int(v.get("up", 0))
        down = int(v.get("down", 0))
        excess = max(0, down - up)
        pen = min(1.0, BASE + STEP * excess)
        if pen > 0:
            out[k] = round(pen, 4)
    return out

def _bump_counts_and_penalty(recs: list[dict]):
    """
    根据这次反馈里附带的 citations 批量更新 counts 与 penalty。
    recs: [{ "file": "...", "page": 12 }, ...]
    """
    counts = _safe_read_json(_FB_COUNTS, {})
    # 每条 citation 累计 up/down；由调用者决定当前反馈是 up 还是 down
    def inc(key: str, field: str):
        row = counts.get(key) or {}
        row[field] = int(row.get(field, 0)) + 1
        counts[key] = row

    # 上层会把 label 透传进来；这里不区分每条 citation 的 label
    for r in recs:
        file = r.get("file")
        page = r.get("page")
        if not file:
            continue
        key = f"{file}::{page}"
        # 占位，实际增量在调用处做（见 feedback_in 内）
        counts.setdefault(key, counts.get(key, {"up": 0, "down": 0}))

    return counts, inc

_CITE_TAG_RE = re.compile(r"\[Source\s+(\d+)\]")
_PRONOUN_PAT = re.compile(r"\b(it|its|this|that|they|their)\b|[它其這該]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")  # 基本漢字

# 简单别名映射（可继续补充）
ALIASES = {
    "OAA": "Old Age Allowance",
    "OALA": "Old Age Living Allowance",
    "LSG": "Lump Sum Grant",
    "LSGSS": "Lump Sum Grant Subvention System",
}

# 用“非字母数字”前后视图替代 \b，避免在中文里失效
# (?<![A-Za-z0-9]) …… (?![A-Za-z0-9])
_ENTITY_EXTRACT_RE = re.compile(
    r"(?<![A-Za-z0-9])("  # ---- 英文正式名稱（多詞短語 + 特定後綴，不匹配單詞本身）----
    # 例如：Old Age Allowance, Disability Allowance, Social Welfare Department Manual ...
    r"(?:Old\s+Age\s+Allowance|Old\s+Age\s+Living\s+Allowance|Disability\s+Allowance|"
    r"Comprehensive\s+Social\s+Security\s+Assistance|Lump\s+Sum\s+Grant\s+Subvention\s+System|"
    r"Lump\s+Sum\s+Grant|Infirmary\s+Care\s+Supplement|Foster\s+Care\s+Allowance|"
    r"Emergency\s+Foster\s+Care\s+Allowance)"
    r")(?![A-Za-z0-9])"
    r"|(?:OAA|OALA|CSSA|DA|LSG|LSGSS)"  # 常見英文縮寫
    r"|(?:長者生活津貼|高齡津貼|老年津貼|傷殘津貼|"
    r"綜合社會保障援助|資助福利服務|統一撥款|整筆撥款|整筆撥款津助制度|統一撥款制度|統一撥款資助|整筆資助|"
    r"資助計劃|援助計劃|補助金|補貼|津貼計劃|津貼安排|"
    r"政策|計劃|手冊|指引|通告|規例|條例|方案|制度|安排)"
    , re.I
)

# 這個比上面的稍窄，用於你的“實體提示/澄清”檢測（不需要過多干擾詞）
# 仅示例，请并入你现有的 _ENTITY_PAT：
_ENTITY_PAT = re.compile(
    r"(?:"
    # ---- English concrete entities (no generic words here!) ----
    r"Old Age Living Allowance|OALA|Old Age Allowance|OAA|"
    r"Disability Allowance|DA|"
    r"Comprehensive Social Security Assistance|CSSA|"
    r"Lump Sum Grant Subvention System|LSG Subvention System|LSGSS|Lump Sum Grant|LSG|"
    r"Infirmary Care Supplement|Foster Care Allowance|Emergency Foster Care Allowance"
    r"|"
    # ---- 中文实体（保留你现有 + 我们之前补的）----
    r"長者生活津貼|高齡津貼|老年津貼|傷殘津貼|綜合社會保障援助|"
    r"整筆撥款|整筆撥款津助制度|統一撥款制度|統一撥款資助|整筆資助|"
    r"資助計劃|援助計劃|補助金|補貼|津貼計劃|津貼安排"
    r")",
    re.I,
)

# --- Helpers for zh-Hant queries ---
def _merge_dedup_hits(h1: list[dict], h2: list[dict], k: int) -> list[dict]:
    """以 (file, page, chunk_id) 去重；分数取较大值；保留来源标记便于调试"""
    seen = {}
    for src, tag in ((h1, "zh"), (h2, "en")):
        for h in (src or []):
            key = (h.get("file"), h.get("page"), h.get("chunk_id"))
            score = float(h.get("score", 0))
            if key not in seen or score > seen[key]["score"]:
                seen[key] = {**h, "score": score, "src": tag}
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:k]

def _expand_aliases_zh(q: str) -> str:
    """把繁中的俗称/简称扩成 (A OR B OR 英文名)；你可把表慢慢补充起来"""
    table = {
        "生果金": ["高齡津貼", "Old Age Allowance", "OAA"],
        "綜援":   ["綜合社會保障援助", "Comprehensive Social Security Assistance", "CSSA"],

        # 🔽 新增：LSG/LSGSS 别名族
        "整筆撥款":           ["統一撥款", "Lump Sum Grant", "LSG"],
        "整筆撥款津助制度":   ["統一撥款制度", "Lump Sum Grant Subvention System", "LSG Subvention System", "LSGSS", "LSG"],
        "統一撥款制度":       ["Lump Sum Grant Subvention System", "LSG Subvention System", "LSGSS", "LSG"],
    }
    for k, vs in table.items():
        if k in q:
            q = q.replace(k, f"({ ' OR '.join([k] + vs) })")
    # 再套用你现有的英文缩写表
    for short, full in ALIASES.items():
        if short in q:
            q = q.replace(short, f"({short} OR {full})")
    return q

def _norm_for_entity(s: str) -> str:
    # 全角->半角 + 去掉多餘空白
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000: code = 0x20
        elif 0xFF01 <= code <= 0xFF5E: code -= 0xFEE0
        out.append(chr(code))
    return re.sub(r"\s+", " ", "".join(out)).strip()

def _extract_focus_phrase(s: str) -> str | None:
    s = _norm_for_entity(s or "")
    # 书名号优先：如《津貼及服務協議》
    m = re.search(r"《(.+?)》", s)
    if m: return m.group(1).strip()
    # 退化：连续大写开头词 + 关键尾词（Accounts/Allowance/Manual/...）
    m = re.search(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,6}\s(?:Accounts?|Allowance|Manual|Programme|Scheme|Policy|System|Subvention))", s)
    return m.group(1).strip() if m else None

def _boost_by_phrase(contexts: list[dict], phrase: str | None, boost: float = 0.35) -> list[dict]:
    if not phrase: 
        return contexts
    for c in contexts:
        t = (c.get("text") 
             or (c.get("meta") or {}).get("text") 
             or "")
        if t and phrase in t:
            c["score"] = float(c.get("score", 0.0)) + boost
    return sorted(contexts, key=lambda x: float(x.get("score", 0.0)), reverse=True)

def _extract_entities_from_text(text: str) -> list[str]:
    if not text: 
        return []
    return [m.group(0).strip() for m in _ENTITY_EXTRACT_RE.finditer(text)]

def _suggest_entities_for(query: str, idx: "Index", emb: "Embedder", top_k: int = 5) -> list[str]:
    """
    用當前查詢在索引裡做一次輕量檢索，從命中的 chunk 文本中抽取實體名，按頻次去重排序返回。
    """
    try:
        contexts = hybrid_retrieve(query, idx, emb, k=20, soft=True)
    except Exception:
        return []
    freq: dict[str, int] = {}
    for c in contexts:
        t = c["meta"].get("text") or ""
        for ent in _extract_entities_from_text(t):
            # 合併大小寫/空白差異
            key = re.sub(r"\s+", " ", ent).strip()
            freq[key] = freq.get(key, 0) + 1
    # 排序 + 去掉過於籠統的詞
    bad = {"Allowance","Scheme","Manual","Programme","Grant","Service","Subvention","System","Policy","計劃","津貼","手冊","政策"}
    items = [e for e,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True) if e not in bad]
    # 保留不同前綴的前 top_k 個
    out = []
    seen_lower = set()
    for e in items:
        low = e.lower()
        if low in seen_lower: 
            continue
        seen_lower.add(low)
        out.append(e)
        if len(out) >= top_k:
            break
    return out

def _looks_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))
# 轻量启发式参数
_MIN_TOKENS = 3
_GENERIC_Q_RE = re.compile(
    r"^(what|how|why|tell me|can you|could you|explain|give me|i want to know|說說|介紹|解釋|請講講|我想知道)\b",
    re.I,
)

def _looks_specific(q: str, contexts: list[dict]) -> bool:
    """
    “已足够具体”的判定：
      1) 命中实体正则（政策/津贴/制度等具体名）。
      2) 含《》書名號。
      3) 中文查詢：CJK 長度 ≥ 6 且 以專有尾詞結尾（津貼/補助/撥款/制度/計劃/指引/服務/津貼計劃/補助金）。
    不再用「上下文出现该短语」作为依据。
    """
    qn = _norm_for_entity(q or "")
    if "《" in qn and "》" in qn:
        _dbg("looks_specific: booktitle")
        return True
    if _ENTITY_PAT.search(qn):
        _dbg("looks_specific: entity")
        return True

    s = _cjk_only(qn)
    if len(s) >= 6 and (
        any(s.endswith(w) for w in ("津貼", "補助", "撥款", "制度", "計劃", "指引", "服務"))
        or s.endswith("津貼計劃") or s.endswith("補助金")
    ):
        _dbg("looks_specific: zh length&suffix")
        return True

    _dbg("looks_specific: False")
    return False

def _tokenize_simple(s: str) -> list[str]:
    return [t for t in re.findall(r"\w+|[\u4e00-\u9fff]", s or "") if t.strip()]

def _is_ambiguous_heuristic(q: str) -> bool:
    """
    仅在非常“短/虚”的情况下返回 True：
      - 空/全是空白
      - 英文长度 < 4 且全是问句词 (what/why/how等)
      - 中文 CJK 长度 ≤ 2 的单字问句
    不再因为包含“政策/津貼”等词就直接判泛。
    """
    if not q or not q.strip():
        return True

    s = q.strip()
    # 英文过短问句
    sl = s.lower()
    if len(sl) <= 3 and sl in {"?", "hi", "hey"}:
        return True

    # 中文极短
    cz = _cjk_only(s)
    if cz and len(cz) <= 2:
        return True

    return False

# 语义模板（极小集合，捕捉“泛问/扩写/解释一下”这类）
_GENERIC_TEMPLATES = [
    "Tell me about it",
    "Tell me about this",
    "What is it",
    "Explain this",
    "Give me details",
    "I want to know more",
    "What about it",
    "請介紹一下",
    "這是什麼",
    "說說看",
]

_GENERIC_EMB: np.ndarray | None = None  # 懒加载缓存

def _ensure_generic_emb() -> np.ndarray:
    global _GENERIC_EMB
    if _GENERIC_EMB is None:
        emb = get_embedder()  # 复用现有的 SentenceTransformer（已做 normalize）
        _GENERIC_EMB = emb.encode(_GENERIC_TEMPLATES)
        if _GENERIC_EMB.ndim == 1:
            _GENERIC_EMB = _GENERIC_EMB.reshape(1, -1)
    return _GENERIC_EMB

def _is_ambiguous_semantic(q: str, thr: float = 0.82) -> bool:
    q = (q or "").strip()
    if not q:
        return True
    emb = get_embedder()
    qe = emb.encode([q])[0]  # 已归一化
    G = _ensure_generic_emb()  # (m, d)
    sims = (G @ qe)  # 余弦相似度
    return float(np.max(sims)) >= thr

_GENERIC_HEADS_ZH = {"津貼", "政策", "制度", "計劃", "手冊", "服務", "指引"}

def _cjk_only(s: str) -> str:
    return "".join(ch for ch in s if "\u4e00" <= ch <= "\u9fff")

def _contains_generic_head_zh(q: str) -> bool:
    """
    只有當查詢“很短”(≤4個 CJK字) 且 恰好是常見泛詞 或 以泛詞結尾(如“介紹津貼”)時，才視為泛問。
    目的是避免把長而具體的中文名詞短語誤判為泛問。
    """
    s = _cjk_only((q or "").strip())
    if not s:
        return False
    if len(s) <= 4 and (s in _GENERIC_HEADS_ZH or any(s.endswith(w) for w in _GENERIC_HEADS_ZH)):
        return True
    return False

# ===== EN generic heads =====
_GENERIC_HEADS_EN = {
    "allowance", "allowances",
    "policy", "policies",
    "scheme", "schemes",
    "program", "programs", "programme", "programmes",
    "service", "services",
    "manual", "handbook", "guideline", "guidelines",
}
_GENERIC_HEADS_EN_RE = re.compile(r"\b(" + "|".join(sorted(_GENERIC_HEADS_EN)) + r")\b", re.I)

def _contains_generic_head_en(q: str) -> bool:
    qn = _norm_for_entity(q or "").lower()
    qn = re.sub(r"[^a-z0-9\s]", " ", qn)  # 去标点
    has_generic = bool(_GENERIC_HEADS_EN_RE.search(qn))
    ent = bool(_ENTITY_PAT.search(qn))    # 注意此处 _ENTITY_PAT 已不含 generic
    return has_generic and not ent

# ===== ZH generic heads (保守) =====
_GENERIC_HEADS_ZH = {"津貼", "政策", "制度", "計劃", "手冊", "服務", "指引", "補助", "補貼"}

def _cjk_only(s: str) -> str:
    return "".join(ch for ch in s if "\u4e00" <= ch <= "\u9fff")

def _contains_generic_head_zh(q: str) -> bool:
    """
    只有當查詢“很短”(≤4個 CJK字) 且 恰好是常見泛詞 或 以泛詞結尾(如“介紹津貼”)時才視為泛問。
    目標：避免將長中文名詞短語誤判為泛問。
    """
    s = _cjk_only((q or "").strip())
    if not s:
        return False
    short = len(s) <= 4 and (s in _GENERIC_HEADS_ZH or any(s.endswith(w) for w in _GENERIC_HEADS_ZH))
    ent = bool(_ENTITY_PAT.search(_norm_for_entity(q or "")))
    _dbg("ZH heads:", short, "entity:", ent, "s=", s)
    return short and not ent

def _should_clarify_smart(user_query: str, lang: str | None = None) -> bool:
    """
    综合启发式与语义判断。
    对中文（zh-Hant）使用更严格阈值，减少误判。
    """
    # 启发式判断：太短/太泛
    if _is_ambiguous_heuristic(user_query):
        _dbg("heuristic=True")
        return True

    # 语言自适应阈值（中文更严格）
    thr = 0.82
    if lang == "zh-Hant":
        thr = 0.86

    amb = _is_ambiguous_semantic(user_query, thr=thr)
    _dbg("semantic:", amb, "thr:", thr)
    return amb

def _clarify_question(user_query: str, lang: str | None) -> str:
    """
    当检测到用户问题模糊时，生成一条追问句，用于提示用户具体化问题。
    """
    if not user_query:
        user_query = "your question"

    # 中文界面
    if lang == "zh-Hant":
        return (
            f"你的問題（「{user_query}」）目前範圍過大。"
            "我在相關的官方文檔中找不到有關該主題的信息。"
        )

    # 英文界面
    return (
        f"Your question (“{user_query}”) is a bit broad. "
        "I couldn't find any information on that topic in documents related to elderly care."
    )

def _clarify_question_smart(user_query: str, lang: str | None, idx: "Index", emb: "Embedder") -> str:
    q = (user_query or "").strip()
    keywords = ["allowance", "policy", "scheme", "manual", "programme", "津貼", "政策", "計劃", "手冊"]
    need_list = any(kw in q.lower() for kw in keywords) or _is_ambiguous_semantic(q)

    if need_list:
        cands = _suggest_entities_for(q, idx, emb, top_k=5)
        if cands:
            if lang == "zh-Hant":
                opts = "、".join(cands[:4])
                return f"你的問題較為籠統（「{q}」）。你是在問 {opts}，還是其他？"
            else:
                opts = ", ".join(cands[:4])
                return f'Your question (“{q}”) is a bit broad. Are you asking about {opts}, or something else?'

        # 🔽 兜底候選（没有抽到实体时仍然给出典型选项）
        fallback_en = ["Old Age Allowance (OAA)", "Old Age Living Allowance (OALA)", "Disability Allowance (DA)"]
        fallback_zh = ["高齡津貼（OAA）", "長者生活津貼（OALA）", "傷殘津貼（DA）"]
        if lang == "zh-Hant":
            return f"你的問題較為籠統（「{q}」）。你是在問 {fallback_zh[0]}、{fallback_zh[1]} 或 {fallback_zh[2]}，還是其他？"
        else:
            return f'Your question (“{q}”) is a bit broad. Are you asking about {fallback_en[0]}, {fallback_en[1]}, or {fallback_en[2]}?'

    # 候選空且不需要列表時，退回通用提示
    return _clarify_question(user_query, lang)

def _expand_aliases(text: str) -> str:
    out = text
    for k, v in ALIASES.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
    return out

def _guess_entity_from_history(msgs: list[dict]) -> str | None:
    """從對話歷史中猜測最近出現的明確政策/津貼名（支援繁中與全角）"""
    for m in reversed(msgs):
        txt = _expand_aliases((m.get("content") or "").strip())
        if not txt:
            continue
        # 🔹在匹配前做正規化
        normed = _norm_for_entity(txt)
        hit = _ENTITY_PAT.search(normed)
        if hit:
            return hit.group(0)

    # 若仍未命中，退回首輪訊息再試
    for m in msgs:
        txt = _expand_aliases((m.get("content") or "").strip())
        normed = _norm_for_entity(txt)
        hit = _ENTITY_PAT.search(normed)
        if hit:
            return hit.group(0)
    return None

_FEEDBACK_DIR = Path("data/feedback")
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
_FEEDBACK_PATH = _FEEDBACK_DIR / "feedback.jsonl"
_METRICS_PATH  = _FEEDBACK_DIR / "metrics.json"   # 用于极简在线指标

def _append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _bump_metrics(label: str):
    # 超迷你在线指标：累计 up/down 次数；可扩展为分桶(相似度区间)统计
    m = {"up": 0, "down": 0}
    if _METRICS_PATH.exists():
        try: m = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception: pass
    m[label] = m.get(label, 0) + 1
    _METRICS_PATH.write_text(json.dumps(m), encoding="utf-8")

def _iter_feedback_jsonl():
    if not _FB_JSONL.exists():
        return
    for line in _FB_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue

import datetime as _dt
def _iter_usage():
    if not _USAGE_LOG.exists():
        return
    for line in _USAGE_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip(): 
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue

@app.get("/metrics/usage")
def metrics_usage(days: int = 7, _auth=Depends(require_bearer)):
    """
    基础使用统计：最近 N 天（默认 7）
    返回总请求数 / clarifications / 无命中 / 平均与P95时延 / 关键词Top-N（英文/数字简单切词）
    """
    now_ms = _now_ms()
    cutoff = now_ms - int(days) * 24 * 60 * 60 * 1000

    total = 0
    clarified = 0
    nohit = 0
    found = 0
    durs = []
    ans_lens = []
    word_counter = collections.Counter()
    per_day = collections.Counter()  # YYYY-MM-DD -> count

    for r in _iter_usage() or []:
        ts = r.get("ts")
        if isinstance(ts, (int, float)) and ts < cutoff:
            continue
        total += 1
        if r.get("clarified"): clarified += 1
        if not r.get("found"): nohit += 1
        else: found += 1
        dur = r.get("duration_ms")
        if isinstance(dur, (int, float)) and dur >= 0:
            durs.append(float(dur))
        al = r.get("answer_len")
        if isinstance(al, int):
            ans_lens.append(al)
        # 简单关键词（英文/数字），如需中英文混合可在此处替换为更好的分词逻辑
        q = r.get("user_query") or ""
        for w in re.findall(r"[A-Za-z0-9]{3,}", q):
            word_counter[w.lower()] += 1

        # day bucket
        if isinstance(ts, (int, float)):
            day = _dt.datetime.utcfromtimestamp(ts/1000.0).strftime("%Y-%m-%d")
            per_day[day] += 1

    avg_dur = statistics.mean(durs) if durs else None
    p95_dur = (statistics.quantiles(durs, n=20)[18] if len(durs) >= 20 else (max(durs) if durs else None))
    avg_ans = statistics.mean(ans_lens) if ans_lens else None

    top_keywords = [{"term": k, "count": v} for k, v in word_counter.most_common(20)]
    series = [{"day": d, "count": c} for d, c in sorted(per_day.items())]

    return {
        "window_days": int(days),
        "total": total,
        "found": found,
        "nohit": nohit,
        "clarified": clarified,
        "avg_duration_ms": round(avg_dur, 2) if avg_dur is not None else None,
        "p95_duration_ms": round(p95_dur, 2) if p95_dur is not None else None,
        "avg_answer_len": round(avg_ans, 2) if avg_ans is not None else None,
        "top_keywords": top_keywords,
        "daily_series": series,
        "updated": now_ms
    }

@app.get("/feedback/metrics")
def feedback_metrics(days: int = 7, _auth=Depends(require_bearer)):
    """
    返回最近 N 天的 up/down 统计与 up-rate。days 默认 7。
    数据源：data/feedback/feedback.jsonl
    """
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - int(days) * 24 * 60 * 60 * 1000

    up = down = total = 0
    for rec in _iter_feedback_jsonl() or []:
        ts = rec.get("ts")  # 我们写入的是毫秒时间戳
        if isinstance(ts, (int, float)):
            if ts < cutoff_ms:
                continue
        # 无时间戳的老数据：可以选择纳入或跳过；这里纳入
        if rec.get("label") == "up":
            up += 1
        elif rec.get("label") == "down":
            down += 1
        total += 1

    uprate = (up / total) if total > 0 else 0.0
    return {
        "days": int(days),
        "up": up,
        "down": down,
        "total": total,
        "uprate": round(uprate, 4),
        "updated": now_ms
    }

@app.post("/feedback")
async def feedback_in(body: FeedbackIn, _auth=Depends(require_bearer)):
    rec = body.model_dump()
    rec["ts"] = int(time.time() * 1000)

    # 1) 先写主日志 & 聚合计数（与你现有逻辑一致）
    try:
        _append_jsonl(_FEEDBACK_PATH, rec)
        _bump_metrics(body.label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write feedback failed: {e}")

    # 2) 即时把这条反馈“转译”为页级 counts & penalty，立刻生效
    try:
        # 载入现有 counts，并得到一个累加器 inc()
        counts, inc = _bump_counts_and_penalty(rec.get("citations") or [])
        # 对本条反馈对应的所有 citation 做 up/down 增量
        fld = "up" if body.label == "up" else "down"
        for c in rec.get("citations") or []:
            file = c.get("file")
            page = c.get("page")
            if not file:
                continue
            key = f"{file}::{page}"
            # 初始化如果不存在
            if key not in counts:
                counts[key] = {"up": 0, "down": 0}
            counts[key][fld] = int(counts[key].get(fld, 0)) + 1

        # 写回 counts
        _write_json_atomic(_FB_COUNTS, counts)

        # 计算/写回 penalty.json —— rag.hybrid_retrieve 会在下一次检索时读取
        penalty = _recompute_penalty_from_counts(counts)
        _write_json_atomic(_FB_PENALTY, penalty)
    except Exception as e:
        # 不中断主流程；只是记录这次“即时惩罚更新”失败
        print(f"[WARN] feedback penalty update failed: {e}")

    return {"ok": True}

@app.middleware("http")
async def add_pna_header(request: Request, call_next):
    # 让 Chrome 的 Private Network Access 预检通过
    response: Response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_index: Index | None = None
_embedder: Embedder | None = None

def _extract_used_indices(answer_text: str, max_k: int) -> List[int]:
    """从答案正文里提取实际被使用的 [Source n]，去重并按首次出现顺序返回。"""
    seen = set()
    order: List[int] = []
    for m in _CITE_TAG_RE.finditer(answer_text or ""):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 1 <= n <= max_k and n not in seen:
            seen.add(n)
            order.append(n)
    return order

def _citations_by_usage(answer_text: str, contexts: List[Dict]) -> List[Dict]:
    """只返回正文里实际使用到的 [Source n] 所对应的 citations。"""
    from .rag import format_citations
    all_cites = format_citations(contexts)  # 顺序与 [Source 1..K] 对应
    used = _extract_used_indices(answer_text, len(all_cites))
    return [all_cites[i - 1] for i in used]  # i 从 1 开始

def _sanitize_inline_citations(text: str, max_k: int) -> str:
    def repl(m):
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= max_k else ""
    return _CITE_TAG_RE.sub(repl, text)

def get_index() -> Index:
    global _index
    if _index is None:
        idx = Index(settings.index_dir)
        idx.load()
        if idx.faiss is None:
            raise HTTPException(status_code=503, detail="Index not built yet. Run /ingest or scripts/ingest.py")
        _index = idx
    return _index

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(settings.embedding_model_name, settings.embedding_device)
    return _embedder

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(_auth=Depends(require_bearer)):
    docs_count, chunks_count = ingest_corpus(settings.docs_dir, settings.index_dir)
    global _index, _embedder
    _index = None
    _embedder = None
    return IngestResponse(documents_indexed=docs_count, chunks_indexed=chunks_count)

def _extract_answer_text(data: dict) -> str:
    # 新增：处理 "answer" 是嵌套字符串 JSON 的情况
    if "answer" in data and isinstance(data["answer"], str):
        try:
            parsed = json.loads(data["answer"].replace("'", '"'))  # 兼容单引号
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed["response"]
        except Exception:
            pass  # fallback below

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        pass
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        c0 = data["choices"][0]
        if isinstance(c0, dict) and "text" in c0 and isinstance(c0["text"], str):
            return c0["text"]
    for key in ("content", "answer", "message", "data"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            for kk in ("content", "text", "answer"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    return vv
    return str(data)

def _normalize_answer_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)

    # 1) 解析 {"response": "..."} 或 {'response': '...'}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("response"), str):
            text = obj["response"]
    except Exception:
        m = re.match(r"\s*\{[^}]*'response'\s*:\s*'(.*?)'\s*\}\s*$", text, flags=re.S)
        if m:
            text = m.group(1)

    # 2) 去掉文末内嵌的 Sources 段
    text = re.sub(r"\n+Sources\s*\n(?:\[[^\n]+\].*\n?)+\s*$", "", text, flags=re.I)

    # 3) 统一换行，并把字面量 \\n 转成真正换行
    text = text.replace("\r\n", "\n")
    if "\\n" in text:
        text = text.replace("\\n", "\n")

    return text.strip()

def _extract_stream_token_preserve(text_line: str) -> str | None:
    line = text_line
    if not line:
        return None

    # 处理 SSE 前缀
    s = line.strip()
    if s.lower() == "[done]":
        return None
    if s.startswith("data:"):
        s = s[5:].strip()

    # 尝试解析 JSON；如果不是 JSON，直接返回原行（不改动）
    try:
        obj = json.loads(s)
    except Exception:
        return line  # 非 JSON，原样返回（保留其中换行/空格）

    # 在常见路径中取字符串值（原样返回）
    def get_path(o, path):
        cur = o
        for p in path:
            if isinstance(p, int):
                if isinstance(cur, list) and len(cur) > p:
                    cur = cur[p]
                else:
                    return None
            else:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
        return cur if isinstance(cur, str) else None

    candidates = [
        ("choices", 0, "delta", "content"),
        ("choices", 0, "text"),
        ("response",),
        ("content",),
        ("answer",),
        ("message", "content"),
        ("data", "content"),
    ]
    for path in candidates:
        val = get_path(obj, path)
        if isinstance(val, str):
            return val  # 原样返回，不做任何替换或去除

    # 没命中就把整行 JSON 丢弃（避免再把 JSON 文本回传前端）
    return None

def _extract_stream_piece(line: str) -> str:
    """
    从 SmartCare 流式每行里提取纯文本。
    兼容以下情况：
      - {"response": " ... "}
      - {"text": "..."} / {"content": "..."} / {"data": "..."}
      - 纯文本（直接返回）
      - SSE 风格的 'data: {...}'（可选兼容）
    """
    if not line:
        return ""

    # 1) 去掉可能的 SSE 前缀
    if line.startswith("data:"):
        line = line[len("data:"):].strip()

    # 2) 优先 JSON 解析
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            for key in ("response", "text", "content", "data"):
                val = obj.get(key)
                if isinstance(val, str):
                    return val
        # 如果是数组或其他结构，这里不处理
    except Exception:
        pass

    # 3) 兼容非严格 JSON，简单用正则兜底（例如 {'response': '...'}）
    m = re.search(r"'response'\s*:\s*'(.*?)'", line)
    if m:
        return m.group(1)

    # 4) 如果 line 不是 JSON，就当作纯文本（少见但安全）
    #    但要排除纯粹的空白/心跳
    if line.strip():
        return line
    return ""

def _not_found_text(lang: str | None) -> str:
    if lang == "zh-Hant":
        return ("抱歉，我在目前納入的社會福利文件中沒有找到你這個問題的具體答案。"
                "你可以嘗試：\n"
                "• 換一種說法或補充更具體的名詞（例如津貼名稱、服務單位）\n"
                "• 指定文件或年份（例如 2024 年 LSG Subvention Manual）\n"
                "如果需要，我可以幫你重述查詢或列出相關章節供你查閱。")
    return ("Sorry, I couldn't find a specific answer to this question in the indexed documents.\n"
            "You can try:\n"
            "• Rephrasing or adding more specific terms (e.g., the exact allowance name/service unit)\n"
            "• Mentioning a document or year (e.g., 2024 LSG Subvention Manual)\n"
            "If you like, I can help refine your query or show nearby sections.")

from .llm_client import smartcare_chat, smartcare_chat_stream, smartcare_rewrite_query
@app.post("/chat", response_model=ChatAnswer)
async def chat(req: ChatRequest, _auth=Depends(require_bearer)):
    idx = get_index()
    emb = get_embedder()

    # 1) 取得 messages（前端现在会带最近若干轮）
    msgs = [m.dict() if hasattr(m, "dict") else m for m in req.messages]

    # 2) 改写为独立问题（有历史时更有效）
    #    如果 messages 很短（只有一条 user），就直接用它
    if len(msgs) >= 2:
        rewritten = await smartcare_rewrite_query(msgs)
        user_query = rewritten
    else:
        user_query = msgs[-1]["content"]

    # ★ 規則兜底：若仍含代詞，嘗試補上最近實體
    if _PRONOUN_PAT.search(user_query):
        ent = _guess_entity_from_history(msgs)
        if ent:
            # 不粗暴替换用户原句，只给检索信号加注释
            user_query = f"{user_query} (about {ent})"
    
    # 3) 用改写后的独立问题进行检索（繁中：原文 + 英译 双通道合并）
    is_followup_pronoun = bool(_PRONOUN_PAT.search(msgs[-1]["content"]))

    if req.language == "zh-Hant":
        # 3.1 别名/俗称扩展（只影响检索，不改动原 messages）
        query_zh = _expand_aliases_zh(user_query)

        # 3.2 同时用繁中与英译检索
        try:
            query_en = await smartcare_translate_to_en(query_zh)
        except Exception:
            query_en = None

        hits_zh = hybrid_retrieve(query_zh, idx, emb, settings.top_k, soft=is_followup_pronoun)
        hits_en = hybrid_retrieve(query_en, idx, emb, settings.top_k, soft=True) if query_en else []

        # 3.3 合并去重（保留较高分，截到 top_k）
        contexts = _merge_dedup_hits(hits_zh, hits_en, settings.top_k)

        # （可选）把“实际用于检索的 query”记下来便于日志排查
        user_query = query_zh
    else:
        contexts = hybrid_retrieve(user_query, idx, emb, settings.top_k, soft=is_followup_pronoun)

    # 从“当前问句”和“上一个问句/回答”中抓一个焦点短语
    focus = (_extract_focus_phrase(msgs[-1]["content"]) or
            (len(msgs) >= 2 and _extract_focus_phrase(msgs[-2]["content"])) or
            _guess_entity_from_history(msgs))

    # 按焦点短语重排（把包含该短语的分片往前推）
    contexts = _boost_by_phrase(contexts, focus, boost=0.35)

    # === 检索信号评估 ===
    max_score = max((c.get("score", 0.0) for c in contexts), default=0.0)
    found_enough = len(contexts) >= settings.min_sources_required
    # 检索“弱命中”的判定阈值（可在 .env 暴露，如 CLARIFY_RETRIEVE_THR）
    retrieve_weak_thr = getattr(settings, "clarify_retrieve_thr", settings.min_vec_sim * 0.4)
    is_weak = max_score < retrieve_weak_thr

    # === 只有在“命中不足或过弱”时，才考虑澄清 ===
    want_clarify = False
    if (not found_enough) or is_weak:
        try:
            # 推荐：_should_clarify_smart 支持 contexts / thr 参数
            want_clarify = _should_clarify_smart(
                user_query, req.language, contexts=contexts,
                thr=getattr(settings, "clarify_thr", 0.65)
            )
        except TypeError:
            # 兼容旧签名
            want_clarify = _should_clarify_smart(user_query, req.language)

        # 语言泛词的轻量规则（英文 “allowance/policy/scheme…”，中文保守泛词）
        if req.language == "en":
            if _contains_generic_head_en(user_query):
                # 英文命中泛词时更倾向于澄清
                want_clarify = True
        else:
            if _contains_generic_head_zh(user_query) and not _looks_specific(user_query, contexts):
                want_clarify = True

        if want_clarify:
            text = _clarify_question_smart(user_query, req.language, idx, emb)
            if req.stream:
                async def event_stream():
                    yield text
                    yield "\nCITATIONS:[]\n"
                return StreamingResponse(event_stream(), media_type="text/plain")
            else:
                return ChatAnswer(answer=text, citations=[])

    # 走到这里：检索并不弱（或虽弱但不需澄清）
    # 如果真的啥都没有，再走“未找到”
    if not found_enough:
        text = _not_found_text(req.language)
        if req.stream:
            async def event_stream():
                yield text
                yield "\nCITATIONS:[]\n"
            return StreamingResponse(event_stream(), media_type="text/plain")
        else:
            return ChatAnswer(answer=text, citations=[])

    # 4) 正常拼 prompt（把原 messages 发给模型，这样它能“按上下文口吻”回答）
    prompt_msgs = build_prompt(msgs, contexts)
    if req.language == "zh-Hant":
        prompt_msgs.insert(0, {
            "role": "system",
            "content": (
                "請使用繁體中文回答所有問題，"
                "語氣親切、句子簡短，避免使用艱深詞彙，"
                "讓長者能容易明白。"
            )
        })
    elif req.language == "en":
        prompt_msgs.insert(0, {
            "role": "system",
            "content": (
                "Please answer in clear, short English sentences within 100 words suitable for older adults. "
                "Avoid jargon and keep the response friendly."
            )
        })

    if req.stream:
        async def event_stream():
            buffer = ""
            async for line in smartcare_chat_stream(prompt_msgs):
                piece = _extract_stream_piece(line)
                if piece:
                    buffer += piece
                    yield piece
            clean_final = _sanitize_inline_citations(buffer, len(contexts))
            cites = _citations_by_usage(clean_final, contexts)
            trailer = "CITATIONS:" + json.dumps(cites, ensure_ascii=False)
            yield "\n" + trailer + "\n"

            # ✅ 发送完毕后记录 usage 日志（此时 clean_final/cites 可用）
            try:
                _append_jsonl_safe(_USAGE_LOG, {
                    "ts": _now_ms(),
                    "lang": req.language,
                    "duration_ms": None,
                    "user_query": user_query,
                    "clarified": False,  # 走到这里代表没有澄清直接回答
                    "context_hits": len(contexts),
                    "found": len(contexts) >= settings.min_sources_required,
                    "answer_len": len(clean_final or ""),
                    "citations_cnt": len(cites or []),
                    "stream": True
                })
            except Exception:
                pass

        return StreamingResponse(event_stream(), media_type="text/plain")

    data = await smartcare_chat(prompt_msgs)
    answer_text = _normalize_answer_text(_extract_answer_text(data))
    answer_text = _sanitize_inline_citations(answer_text, len(contexts))
    citations = _citations_by_usage(answer_text, contexts)

    try:
        _append_jsonl_safe(_USAGE_LOG, {
            "ts": _now_ms(),
            "lang": req.language,
            "duration_ms": None,
            "user_query": user_query,
            "clarified": _should_clarify_smart(user_query) and not _looks_specific(user_query, contexts),
            "context_hits": len(contexts),
            "found": len(contexts) >= settings.min_sources_required,
            "answer_len": len(answer_text or ""),
            "citations_cnt": len(citations or []),
            "stream": False
        })
    except Exception:
        pass

    return ChatAnswer(answer=answer_text, citations=citations)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
    