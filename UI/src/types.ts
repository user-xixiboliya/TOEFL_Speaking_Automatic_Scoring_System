export type QuestionManifestItem = {
  id: string;
  title: string;
  taskJsonUrl: string;
  lectureJsonUrl: string;
  audioUrl: string;
  category?: string;
};

export type LectureJson = {
  lecture_text: string;
};

export type TaskJson = {
  task: string;
};

export type EvaluateResponse = {
  audio_score: number;
  transcribed_text: string;
  llm_score: number;
  audio_score_detail: unknown | null;
  llm_score_detail: unknown | null;
};