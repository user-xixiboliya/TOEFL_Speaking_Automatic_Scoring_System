# TOEFL Speaking Scoring Web UI

A local web application for **TOEFL Speaking practice and automatic scoring**, featuring a question bank, lecture audio playback, browser-based recording, and AI-powered evaluation.

This project uses a **frontend–backend separated architecture**:

- **Frontend**: React + Vite (local UI)
- **Backend**: FastAPI (scoring API)
- **LLM Scoring**: Ollama (local or server-side)

---

## Features

- Local TOEFL Speaking question bank
- Lecture audio playback
- Browser microphone recording (MediaRecorder API)
- One-click submission for scoring (multipart/form-data)
- Detailed scoring results:
  - Audio Score (1–6)
  - Automatic transcription
  - LLM Score (0–10)
  - Detailed speaking feedback (collapsible, scrollable)
- UI accessible via LAN or public tunnel (Cloudflare / ngrok)

---

## Project Structure

```text
.
├── src/
│   ├── components/
│   │   ├── TopBar.tsx
│   │   ├── LecturePlayer.tsx
│   │   ├── RecorderPanel.tsx
│   │   └── ResultsPanel.tsx
│   ├── pages/
│   │   ├── QuestionBankPage.tsx
│   │   └── PracticePage.tsx
│   ├── lib/
│   │   ├── api.ts
│   │   └── questionBank.ts
│   ├── types.ts
│   ├── main.tsx
│   └── App.tsx
├── public/
│   └── question-bank/
│       ├── questions.json
│       ├── task/
│       │   └── 001.json
│       └── lecture/
│           ├── 001.json
│           └── 001.mp3
├── scripts/
│   └── sync-question-bank.mjs
├── vite.config.ts
├── package.json
└── README.md
```

---

## How to Run

### Frontend

```bash
npm install
npm run dev
```

Open:
```
http://localhost:5173
```

LAN access:
```bash
npm run dev -- --host 0.0.0.0 --port 5173
```

---

### Backend

```bash
python api_server.py
```

---

### Ollama (Required)

```bash
ollama serve
ollama pull llama3
```

## Notes

- Ensure `ffmpeg` is installed for audio processing
- Ollama must be reachable at `http://localhost:11434`
- Audio filenames are sanitized to avoid filesystem issues

---

## License

For educational and research purposes.
