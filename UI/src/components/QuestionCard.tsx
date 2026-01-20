import { Link } from "react-router-dom";
import type { QuestionManifestItem } from "../types";

export function QuestionCard({ item }: { item: QuestionManifestItem }) {
  return (
    <Link
      to={`/practice/${encodeURIComponent(item.id)}`}
      className="group rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900 group-hover:text-blue-700">ID: {item.id}</div>
        </div>

        {item.category && (
          <span className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-600">
            {item.category}
          </span>
        )}
      </div>

      <div className="mt-3 text-xs text-slate-500">
        点击进入练习：听音频 → 录音 → 上传评分
      </div>
    </Link>
  );
}
