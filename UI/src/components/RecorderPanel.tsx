import { useEffect, useMemo, useRef, useState } from "react";

function pickMimeType(): string {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/wav",
    "audio/mp4"
  ];
  for (const c of candidates) {
    // @ts-ignore
    if (window.MediaRecorder && MediaRecorder.isTypeSupported?.(c)) return c;
  }
  return "audio/webm";
}

export function RecorderPanel(props: {
  disabled?: boolean;
  onRecorded: (blob: Blob, mime: string) => void;
}) {
  const { disabled, onRecorded } = props;

  const [supported, setSupported] = useState(true);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);

  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [mime, setMime] = useState<string>(pickMimeType());

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<number | null>(null);

  const canStart = useMemo(() => !disabled && !recording, [disabled, recording]);
  const canStop = useMemo(() => !disabled && recording, [disabled, recording]);

  useEffect(() => {
    setSupported(Boolean((window as any).MediaRecorder));
  }, []);

  useEffect(() => {
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [blobUrl]);

  function startTimer() {
    stopTimer();
    timerRef.current = window.setInterval(() => setSeconds((s) => s + 1), 1000);
  }
  function stopTimer() {
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
  }

  async function start() {
    if (!supported) return;
    setSeconds(0);
    setMime(pickMimeType());
    chunksRef.current = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream, { mimeType: pickMimeType() });
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      stopTimer();
      setRecording(false);

      // stop tracks
      stream.getTracks().forEach((t) => t.stop());

      const out = new Blob(chunksRef.current, { type: recorder.mimeType });
      const url = URL.createObjectURL(out);
      setBlobUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      onRecorded(out, recorder.mimeType);
    };

    recorder.start();
    setRecording(true);
    startTimer();
  }

  function stop() {
    const r = mediaRecorderRef.current;
    if (!r) return;
    try {
      r.stop();
    } catch {}
  }

  function reset() {
    setSeconds(0);
    chunksRef.current = [];
    setBlobUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
  }

  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900">录音</div>
          <div className="mt-1 text-xs text-slate-500">
            允许浏览器麦克风权限后开始作答（支持 MediaRecorder）
          </div>
        </div>

        <div className="text-sm font-semibold text-slate-900 tabular-nums">
          {mm}:{ss}
        </div>
      </div>

      {!supported && (
        <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
          当前浏览器不支持 MediaRecorder，建议使用最新版 Chrome/Edge。
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          className={`rounded-xl px-4 py-2 text-sm font-semibold text-white ${
            canStart ? "bg-emerald-600 hover:bg-emerald-700" : "bg-slate-300"
          }`}
          disabled={!canStart}
          onClick={start}
        >
          开始录音
        </button>

        <button
          className={`rounded-xl px-4 py-2 text-sm font-semibold text-white ${
            canStop ? "bg-rose-600 hover:bg-rose-700" : "bg-slate-300"
          }`}
          disabled={!canStop}
          onClick={stop}
        >
          停止
        </button>

        <button
          className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          onClick={reset}
          disabled={disabled || recording}
        >
          重录
        </button>
      </div>

      {blobUrl && (
        <div className="mt-4">
          <div className="text-xs text-slate-500">录音回放</div>
          <audio className="mt-2 w-full" controls src={blobUrl} />
          <div className="mt-1 text-[11px] text-slate-400">Mime: {mime}</div>
        </div>
      )}
    </div>
  );
}