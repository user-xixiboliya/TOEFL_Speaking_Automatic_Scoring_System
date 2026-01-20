import type { QuestionManifestItem, LectureJson, TaskJson } from "../types";

export async function loadManifest(): Promise<QuestionManifestItem[]> {
  const res = await fetch("/question-bank/questions.json", { cache: "no-store" });
  if (!res.ok) throw new Error("无法加载题库 manifest：/question-bank/questions.json");
  const data = (await res.json()) as QuestionManifestItem[];
  return data;
}

export async function loadLecture(url: string): Promise<LectureJson> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`无法加载 lecture json：${url}`);
  return (await res.json()) as LectureJson;
}

export async function loadTask(url: string): Promise<TaskJson> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`无法加载 task json：${url}`);
  return (await res.json()) as TaskJson;
}
