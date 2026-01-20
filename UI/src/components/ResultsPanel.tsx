import type { EvaluateResponse } from "../types";
import { CollapsibleDetail } from "./CollapsibleDetail";

export function ResultsPanel({ result }: { result: EvaluateResponse }) {
  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <div className="text-sm font-semibold text-slate-900">评分结果</div>
        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <div className="text-xs text-slate-500">audio_score (1–6)</div>
            <div className="mt-1 text-3xl font-bold text-slate-900">{result.audio_score.toFixed(2)}</div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <div className="text-xs text-slate-500">content_score (0–10)</div>
            <div className="mt-1 text-3xl font-bold text-slate-900">{result.llm_score.toFixed(2)}</div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <div className="text-sm font-semibold text-slate-900">ASR 转写</div>
          <button
            className="rounded-lg border border-slate-200 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            onClick={() => navigator.clipboard.writeText(result.transcribed_text || "")}
          >
            复制
          </button>
        </div>
        <div className="mt-3 whitespace-pre-wrap text-sm text-slate-700">
          {result.transcribed_text || "(空)"}
        </div>
      </div>

      <CollapsibleDetail
        title="详细分析"
        value={result.llm_score_detail}
        defaultCollapsed={false}
      />

      <CollapsibleDetail
        title="音频评分详情（audio_score_detail）"
        value={result.audio_score_detail}
        defaultCollapsed={true}
      />
    </div>
  );
}