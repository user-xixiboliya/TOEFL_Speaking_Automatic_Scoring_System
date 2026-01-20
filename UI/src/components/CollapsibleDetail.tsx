import { useMemo, useState } from "react";

function toPretty(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function CollapsibleDetail(props: {
  title: string;
  value: unknown;
  defaultCollapsed?: boolean;
}) {
  const { title, value, defaultCollapsed } = props;
  const [open, setOpen] = useState(!defaultCollapsed);
  const text = useMemo(() => toPretty(value), [value]);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-slate-900">{title}</div>
        <div className="flex items-center gap-2">
          <button
            className="rounded-lg border border-slate-200 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            onClick={() => navigator.clipboard.writeText(text)}
            disabled={!text}
          >
            Copy
          </button>
          <button
            className="rounded-lg border border-slate-200 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? "收起" : "展开"}
          </button>
        </div>
      </div>

      {open && (
        <div className="mt-3 max-h-[420px] overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-3">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs text-slate-800">
            {text || "(空)"}
          </pre>
        </div>
      )}
    </div>
  );
}