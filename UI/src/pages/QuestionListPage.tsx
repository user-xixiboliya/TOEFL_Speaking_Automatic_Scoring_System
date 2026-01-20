import { useEffect, useMemo, useState } from "react";
import { loadManifest } from "../lib/questionBank";
import type { QuestionManifestItem } from "../types";
import { QuestionCard } from "../components/QuestionCard";

export function QuestionListPage() {
  const [items, setItems] = useState<QuestionManifestItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    loadManifest()
      .then((d) => mounted && setItems(d))
      .catch((e) => mounted && setErr(e?.message || String(e)))
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return items;
    return items.filter((it) => it.id.toLowerCase().includes(s) || it.title.toLowerCase().includes(s));
  }, [items, q]);

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <div className="text-lg font-semibold text-slate-900">题库</div>
        <div className="mt-1 text-sm text-slate-600">选择一道题开始练习（本地题库 / 浏览器录音 / 后端评分）</div>

        <div className="mt-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <input
            className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-200 md:max-w-md"
            placeholder="搜索题目（id）"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />

          <div className="text-xs text-slate-500">
            共 <span className="font-semibold text-slate-700">{filtered.length}</span> 题
          </div>
        </div>
      </div>

      {loading && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600">加载中…</div>
      )}

      {err && (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-700">
          加载失败：{err}
          <div className="mt-2 text-xs text-rose-600">
            请先运行 <span className="font-mono">npm run sync:bank</span> 生成 /public/question-bank/questions.json
          </div>
        </div>
      )}

      {!loading && !err && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {filtered.map((it) => (
            <QuestionCard key={it.id} item={it} />
          ))}
        </div>
      )}
    </div>
  );
}
