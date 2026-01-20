import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { EvaluateResponse, QuestionManifestItem } from "../types";
import { loadLecture, loadManifest, loadTask } from "../lib/questionBank";
import { evaluateSpeaking } from "../lib/api";
import { LecturePlayer } from "../components/LecturePlayer";
import { RecorderPanel } from "../components/RecorderPanel";
import { ResultsPanel } from "../components/ResultsPanel";

const LS_KEY = "toefl_api_key";

export function PracticePage() {
  const { id } = useParams();
  const [manifest, setManifest] = useState<QuestionManifestItem[] | null>(null);
  const [item, setItem] = useState<QuestionManifestItem | null>(null);

  const [taskText, setTaskText] = useState<string>("");
  const [lectureText, setLectureText] = useState<string>("");
  const [loadingTexts, setLoadingTexts] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioMime, setAudioMime] = useState<string>("audio/webm");

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [submitErr, setSubmitErr] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let mounted = true;
    loadManifest()
      .then((m) => {
        if (!mounted) return;
        setManifest(m);
      })
      .catch((e) => setErr(e?.message || String(e)));
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!manifest || !id) return;
    const found = manifest.find((x) => x.id === id) || null;
    setItem(found);
    setResult(null);
    setAudioBlob(null);
    setSubmitErr(null);
  }, [manifest, id]);

  useEffect(() => {
    if (!item) return;
    let mounted = true;
    setLoadingTexts(true);
    setErr(null);

    Promise.all([loadTask(item.taskJsonUrl), loadLecture(item.lectureJsonUrl)])
      .then(([t, l]) => {
        if (!mounted) return;
        setTaskText(t.task || "");
        setLectureText(l.lecture_text || "");
      })
      .catch((e) => mounted && setErr(e?.message || String(e)))
      .finally(() => mounted && setLoadingTexts(false));

    return () => {
      mounted = false;
    };
  }, [item]);

  const canSubmit = useMemo(() => {
    return Boolean(audioBlob) && !submitting && !loadingTexts && Boolean(taskText) && Boolean(lectureText);
  }, [audioBlob, submitting, loadingTexts, taskText, lectureText]);

  async function onSubmit() {
    if (!item || !audioBlob) return;
    setSubmitting(true);
    setSubmitErr(null);
    setResult(null);

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    const apiKey = localStorage.getItem(LS_KEY) || "";

    try {
      const safeId = item.id.replaceAll("/", "_");
      const ext = audioMime.includes("webm") ? "webm" : audioMime.includes("wav") ? "wav" : "mp3";
      const resp = await evaluateSpeaking({
        audioBlob,
        audioFilename: `${safeId}.${ext}`,
        transcriptText: lectureText,
        taskText: taskText,
        apiKey,
        signal: ac.signal
      });
      setResult(resp);
    } catch (e: any) {
      setSubmitErr(e?.message || String(e));
    } finally {
      setSubmitting(false);
    }
  }

  if (err) {
    return (
      <div className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-700">
        出错了：{err}
        <div className="mt-3">
          <Link className="text-rose-700 underline" to="/">返回题库</Link>
        </div>
      </div>
    );
  }

  if (!item) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600">
        {manifest ? "未找到该题目 ID。" : "加载题库中…"}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Link to="/" className="text-sm text-slate-600 hover:text-slate-900">
          ← 返回题库
        </Link>
        <div className="text-xs text-slate-500">Practice · ID: {item.id}</div>
      </div>

      {/* Main */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Left: materials */}
        <div className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-4">
            <div className="text-sm font-semibold text-slate-900">Task</div>
            <div className="mt-2 whitespace-pre-wrap text-sm text-slate-700">
              {loadingTexts ? "加载中…" : taskText}
            </div>
          </div>

          <LecturePlayer audioUrl={item.audioUrl} />

          <details className="rounded-2xl border border-slate-200 bg-white p-4">
            <summary className="cursor-pointer select-none text-sm font-semibold text-slate-900">
              Transcript（可选查看）
            </summary>
            <div className="mt-3 whitespace-pre-wrap text-sm text-slate-700">
              {loadingTexts ? "加载中…" : lectureText}
            </div>
          </details>
        </div>

        {/* Right: record & submit */}
        <div className="space-y-4">
          <RecorderPanel
            disabled={submitting}
            onRecorded={(blob, mime) => {
              setAudioBlob(blob);
              setAudioMime(mime);
              setResult(null);
              setSubmitErr(null);
            }}
          />

          <div className="rounded-2xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-slate-900">提交评分</div>
                <div className="mt-1 text-xs text-slate-500">
                </div>
              </div>

              <button
                className={`rounded-xl px-4 py-2 text-sm font-semibold text-white ${
                  canSubmit ? "bg-blue-600 hover:bg-blue-700" : "bg-slate-300"
                }`}
                disabled={!canSubmit}
                onClick={onSubmit}
              >
                {submitting ? "评分中…" : "开始评分"}
              </button>
            </div>

            {!audioBlob && (
              <div className="mt-3 text-xs text-slate-500">提示：录音完成后才能提交。</div>
            )}

            {submitErr && (
              <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
                {submitErr}
              </div>
            )}
          </div>

          {result && <ResultsPanel result={result} />}
        </div>
      </div>
    </div>
  );
}