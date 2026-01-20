import { useEffect, useMemo, useState } from "react";
import { API_BASE_URL, checkHealth } from "../lib/api";

const LS_KEY = "toefl_api_key";

export function TopBar() {
  const [ok, setOk] = useState<boolean | null>(null);
  const [apiKey, setApiKey] = useState<string>(() => localStorage.getItem(LS_KEY) || "");
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    checkHealth(ac.signal).then(setOk);
    const t = setInterval(() => checkHealth(ac.signal).then(setOk), 8000);
    return () => {
      clearInterval(t);
      ac.abort();
    };
  }, []);

  useEffect(() => {
    localStorage.setItem(LS_KEY, apiKey);
  }, [apiKey]);

  const statusText = useMemo(() => {
    if (ok === null) return "检查中…";
    return ok ? "后端可用" : "后端不可达";
  }, [ok]);

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-slate-900" />
          <div>
            <div className="text-sm font-semibold text-slate-900">TOEFL Speaking Scoring</div>
            <div className="text-xs text-slate-500">本地练习 · 本机评分</div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600">
            <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : ok === false ? "bg-rose-500" : "bg-slate-300"}`} />
            <span>{statusText}</span>
          </div>

          <button
            className="rounded-lg border border-slate-200 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
            onClick={() => setExpanded((v) => !v)}
          >
            设置
          </button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-slate-200 bg-white">
          <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 px-4 py-4 md:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm font-semibold text-slate-900">API Base URL</div>
              <div className="mt-1 text-xs text-slate-600 break-all">{API_BASE_URL}</div>
              <div className="mt-2 text-xs text-slate-500">
                评分接口：{API_BASE_URL}/api/evaluate，健康检查：{API_BASE_URL}/health
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 p-4">
              <div className="text-sm font-semibold text-slate-900">Authorization（可选）</div>
              <div className="mt-1 text-xs text-slate-600">
              </div>
              <input
                className="mt-3 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="输入 API Key（仅本地保存）"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <div className="mt-2 text-xs text-slate-500">已保存到 localStorage：{LS_KEY}</div>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}