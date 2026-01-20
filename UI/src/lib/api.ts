import type { EvaluateResponse } from "../types";

const DEFAULT_BASE =
  "https://agency-thats-removing-connecticut.trycloudflare.com";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.toString() || DEFAULT_BASE;

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal });
    if (!res.ok) return false;
    const json = await res.json();
    return Boolean(json?.ok);
  } catch {
    return false;
  }
}

/**
 * POST /api/evaluate
 * multipart/form-data:
 * - audio: file (required)
 * - transcript_text or transcript file (required one of them)
 * - task_text or task file (required one of them)
 *
 * 服务端可能要求 Authorization: Bearer <TOEFL_API_KEY>（若其设置了 API_KEY）:contentReference[oaicite:6]{index=6}
 */
export async function evaluateSpeaking(args: {
  audioBlob: Blob;
  audioFilename: string;
  transcriptText: string;
  taskText: string;
  apiKey?: string;
  signal?: AbortSignal;
}): Promise<EvaluateResponse> {
  const fd = new FormData();
  fd.append("audio", new File([args.audioBlob], args.audioFilename, { type: args.audioBlob.type || "audio/webm" }));

  // 这里用 text 字段方式，严格匹配后端的 transcript_text / task_text :contentReference[oaicite:7]{index=7}
  fd.append("transcript_text", args.transcriptText);
  fd.append("task_text", args.taskText);

  const headers: Record<string, string> = {};
  if (args.apiKey?.trim()) {
    headers["Authorization"] = `Bearer ${args.apiKey.trim()}`;
  }

  const res = await fetch(`${API_BASE_URL}/api/evaluate`, {
    method: "POST",
    body: fd,
    headers,
    signal: args.signal
  });

  if (!res.ok) {
    // 直接把后端 detail 透出，便于定位（例如缺 transcript/task）:contentReference[oaicite:8]{index=8}
    let msg = `请求失败: ${res.status}`;
    try {
      const j = await res.json();
      msg = j?.detail ? String(j.detail) : msg;
    } catch {}
    throw new Error(msg);
  }

  return (await res.json()) as EvaluateResponse;
}