import React, { useEffect, useState } from "react";

/** ============ Types ============ */
type FeedbackMetrics = {
  days: number;
  up: number;
  down: number;
  total: number;
  uprate: number;   // 0~1
  updated: number;
};

type UsageMetrics = {
  window_days: number;
  total: number;
  found: number;
  nohit: number;
  clarified: number;
  avg_duration_ms: number | null;
  p95_duration_ms: number | null;
  avg_answer_len: number | null;
  top_keywords: { term: string; count: number }[];
  daily_series: { day: string; count: number }[];
  updated: number;
};

type DocItem = { filename: string; size: number; modified: number };
type Status = { status: string; note?: string; last_built?: number };

/** ============ Backend config ============ */
// 统一把 /chat 结尾去掉，避免拼接成 /chat/docs/...
const API_BASE = (import.meta.env.VITE_BACKEND_URL || "http://localhost:8001").replace(/\/chat$/i, "");
const API_TOKEN = import.meta.env.VITE_API_TOKEN || "";
const ACCEPT_FILES = ".md,.markdown,.pdf,text/markdown,application/pdf";

/** ============ Component ============ */
export default function AdminDocs() {
  // 基础 state
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [status, setStatus] = useState<Status>({ status: "loading" });
  const [busy, setBusy] = useState(false);

  const [metrics7d, setMetrics7d] = useState<FeedbackMetrics | null>(null);
  const [usage, setUsage] = useState<UsageMetrics | null>(null);
  const [fileName, setFileName] = useState<string>("");


  async function refresh() {
    try {
      const headers = { Authorization: `Bearer ${API_TOKEN}` };

      const [listRes, statRes, fbRes, usageRes] = await Promise.all([
        fetch(`${API_BASE}/docs/list`,   { headers }),
        fetch(`${API_BASE}/docs/status`, { headers }),
        fetch(`${API_BASE}/feedback/metrics?days=7`, { headers }),
        fetch(`${API_BASE}/metrics/usage?days=7`,    { headers }),
      ]);

      if (!listRes.ok || !statRes.ok || !fbRes.ok || !usageRes.ok) {
        const msg = `HTTP ${listRes.status}/${statRes.status}/${fbRes.status}/${usageRes.status}`;
        setStatus({ status: "error", note: msg });
        return;
      }

      const [list, stat, fb, um] = await Promise.all([
        listRes.json(), statRes.json(), fbRes.json(), usageRes.json()
      ]);

      setDocs(list.docs || []);
      setStatus(stat);
      setMetrics7d(fb);
      setUsage(um);
    } catch (e: any) {
      setStatus({ status: "error", note: e?.message || "network error" });
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;

    // 前端兜底校验：允许 .md / .markdown / .pdf
    const nameOk = /\.(md|markdown|pdf)$/i.test(f.name);
    const mimeOk = (f.type || "").toLowerCase().includes("markdown") ||
                   (f.type || "").toLowerCase().includes("pdf");
    if (!nameOk && !mimeOk) {
      alert("僅支援上傳 Markdown 檔（.md / .markdown）和 PDF 檔（.pdf）");
      e.currentTarget.value = ""; // 清空選擇
      return;
    }

    const fd = new FormData();
    fd.append("file", f);
    setBusy(true);
    await fetch(`${API_BASE}/docs/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${API_TOKEN}` },
      body: fd,
    }).then((r) => r.json());
    setBusy(false);
    await refresh();
    
    // ✅ 上传完成后清空文件名显示
    setFileName("");
  }

  async function onDelete(name: string) {
    if (!confirm(`Delete ${name}?`)) return;
    setBusy(true);
    await fetch(`${API_BASE}/docs/${encodeURIComponent(name)}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${API_TOKEN}` },
    }).then((r) => r.json());
    setBusy(false);
    await refresh();
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h1 className="text-4xl font-bold text-center mb-3">Document Admin</h1>
      {/* ===== 工具栏：两行居中分区 ===== */}
      <div className="mb-4 w-full flex flex-col items-center gap-3 text-sm">

        {/* 第一行：上传 + 状态 + 统计（全部居中） */}
        <div className="w-full flex flex-col items-center gap-2">
          {/* 上传（自定义样式 + 边框） */}
          <div className="flex items-center gap-3">
            <label className="inline-flex items-center gap-2 px-3 py-1.5 border border-gray-300 rounded-md bg-white shadow-sm cursor-pointer hover:bg-gray-50">
              <span>选择文件</span>
              <input
                type="file"
                accept={ACCEPT_FILES}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  setFileName(f?.name || "");
                  onUpload(e);
                }}
                className="sr-only"
                disabled={busy}
              />
            </label>
            <span className="text-xs text-gray-500 min-h-[1.25rem]">
              {fileName ? fileName : "未选择文件"}
            </span>
          </div>

          {/* Status（置中） */}
          <div className="text-sm text-gray-700 text-center">
            Status:{" "}
            <b className={status.status === "error" ? "text-red-600" : ""}>
              {status.status}
            </b>{" "}
            {status.note ? `(${status.note})` : ""}
          </div>

          {/* 两行统计（置中、紧凑） */}
          <div className="text-xs leading-snug text-center">
            {metrics7d && (
              <div className="mt-1">
                <span className="text-gray-700">7-day up-rate: </span>
                <b className={
                  metrics7d.uprate >= 0.8 ? "text-green-600"
                  : metrics7d.uprate >= 0.6 ? "text-yellow-600"
                  : "text-red-600"
                }>
                  {Math.round(metrics7d.uprate * 100)}%
                </b>
                <span className="opacity-70">（up {metrics7d.up} / down {metrics7d.down}, n={metrics7d.total}）</span>
              </div>
            )}
          </div>
        </div>

        {/* 第二行：按钮（带边框） */}
        <div className="flex justify-center gap-4">
          <button
            className="px-3 py-1 border border-gray-300 rounded-md bg-white shadow-sm hover:bg-gray-50 disabled:opacity-50"
            onClick={async () => {
              setBusy(true);
              await fetch(`${API_BASE}/docs/cancel`, {
                method: "POST",
                headers: { Authorization: `Bearer ${API_TOKEN}` },
              }).then((r) => r.json());
              setBusy(false);
              await refresh();
            }}
            disabled={busy || status.status !== "indexing"}
          >
            Cancel
          </button>
          <button
            className="px-3 py-1 border border-gray-300 rounded-md bg-white shadow-sm hover:bg-gray-50 disabled:opacity-50"
            onClick={refresh}
            disabled={busy}
          >
            Refresh
          </button>
        </div>
      </div>


      <table className="w-full text-sm border">
        <thead>
          <tr className="bg-gray-50">
            <th className="p-2 text-left">Filename</th>
            <th className="p-2 text-left">Size</th>
            <th className="p-2 text-left">Modified</th>
            <th className="p-2 text-left">Action</th>
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => (
            <tr key={d.filename} className="border-t">
              <td className="p-2">{d.filename}</td>
              <td className="p-2">{(d.size / 1024).toFixed(1)} KB</td>
              <td className="p-2">
                {new Date(d.modified * 1000).toLocaleString()}
              </td>
              <td className="p-2">
                <button
                  className="px-2 py-1 border rounded"
                  onClick={() => onDelete(d.filename)}
                  disabled={busy}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
          {docs.length === 0 && (
            <tr>
              <td className="p-2 text-gray-500" colSpan={4}>
                No documents.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {usage && usage.top_keywords?.length > 0 && (
        <div className="mt-4 text-sm">
          <div className="font-semibold mb-1">Top keywords (7d)</div>
          <div className="flex flex-wrap gap-2">
            {usage.top_keywords.slice(0, 12).map((t, i) => (
              <span
                key={i}
                className="px-2 py-1 rounded-md border border-gray-200 bg-white/70"
                title={`${t.term} ×${t.count}`}
              >
                {t.term} <span className="opacity-60">×{t.count}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
