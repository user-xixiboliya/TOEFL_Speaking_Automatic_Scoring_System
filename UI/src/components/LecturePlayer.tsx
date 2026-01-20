import { useMemo, useRef, useState } from "react";

export function LecturePlayer({ audioUrl }: { audioUrl: string }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [rate, setRate] = useState(1.0);

  const rates = useMemo(() => [0.75, 1.0, 1.25, 1.5], []);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold text-slate-900">Lecture Audio</div>
          <div className="mt-1 text-xs text-slate-500">建议先听完再开始录音</div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">倍速</span>
          <select
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs"
            value={rate}
            onChange={(e) => {
              const v = Number(e.target.value);
              setRate(v);
              if (audioRef.current) audioRef.current.playbackRate = v;
            }}
          >
            {rates.map((r) => (
              <option key={r} value={r}>
                {r}x
              </option>
            ))}
          </select>
        </div>
      </div>

      <audio ref={audioRef} className="mt-3 w-full" controls src={audioUrl} />
    </div>
  );
}