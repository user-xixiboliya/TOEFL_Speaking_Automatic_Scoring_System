import fs from "node:fs";
import path from "node:path";

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function copyFile(src, dst) {
  ensureDir(path.dirname(dst));
  fs.copyFileSync(src, dst);
}

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

function listFilesRecursive(dirAbs) {
  const out = [];
  const stack = [dirAbs];
  while (stack.length) {
    const cur = stack.pop();
    const entries = fs.readdirSync(cur, { withFileTypes: true });
    for (const e of entries) {
      const abs = path.join(cur, e.name);
      if (e.isDirectory()) stack.push(abs);
      else out.push(abs);
    }
  }
  return out;
}

function isJson(absPath) {
  return absPath.toLowerCase().endsWith(".json");
}
function isAudio(absPath) {
  const low = absPath.toLowerCase();
  return low.endsWith(".mp3") || low.endsWith(".wav");
}

const ROOT = process.cwd();

const DATA_DIR = path.join(ROOT, "../data");
const TASK_DIR = path.join(DATA_DIR, "task");
const LECTURE_DIR = path.join(DATA_DIR, "lecture");

const OUT_ROOT = path.join(ROOT, "public", "question-bank");
const OUT_TASK = path.join(OUT_ROOT, "task");
const OUT_LECTURE = path.join(OUT_ROOT, "lecture");

if (!fs.existsSync(TASK_DIR)) {
  console.error("找不到 data/task 目录：", TASK_DIR);
  process.exit(1);
}
if (!fs.existsSync(LECTURE_DIR)) {
  console.error("找不到 data/lecture 目录：", LECTURE_DIR);
  process.exit(1);
}

ensureDir(OUT_TASK);
ensureDir(OUT_LECTURE);

const taskAll = listFilesRecursive(TASK_DIR);
for (const abs of taskAll) {
  const rel = path.relative(TASK_DIR, abs); 
  copyFile(abs, path.join(OUT_TASK, rel));
}

const lectureAll = listFilesRecursive(LECTURE_DIR);
for (const abs of lectureAll) {
  if (isJson(abs) || isAudio(abs)) {
    const rel = path.relative(LECTURE_DIR, abs);
    copyFile(abs, path.join(OUT_LECTURE, rel));
  }
}

const taskJsonFiles = taskAll.filter(isJson);

const items = [];
for (const absTaskJson of taskJsonFiles) {
  const relTask = path.relative(TASK_DIR, absTaskJson).replaceAll("\\", "/"); 
  const relNoExt = relTask.replace(/\.json$/i, ""); 

  const taskObj = readJson(absTaskJson);
  const title =
    typeof taskObj.task === "string"
      ? taskObj.task.trim().split("\n")[0].slice(0, 80)
      : `Question ${relNoExt}`;

  const lectureJsonAbs = path.join(LECTURE_DIR, `${relNoExt}.json`);
  const mp3Abs = path.join(LECTURE_DIR, `${relNoExt}.mp3`);
  const wavAbs = path.join(LECTURE_DIR, `${relNoExt}.wav`);

  const lectureJsonUrl = `/question-bank/lecture/${relNoExt}.json`;
  const taskJsonUrl = `/question-bank/task/${relNoExt}.json`;

  let audioUrl = "";
  if (fs.existsSync(mp3Abs)) audioUrl = `/question-bank/lecture/${relNoExt}.mp3`;
  else if (fs.existsSync(wavAbs)) audioUrl = `/question-bank/lecture/${relNoExt}.wav`;

  items.push({
    id: relNoExt, 
    title,
    taskJsonUrl,
    lectureJsonUrl,
    audioUrl
  });
}

items.sort((a, b) => a.id.localeCompare(b.id));

ensureDir(OUT_ROOT);
fs.writeFileSync(
  path.join(OUT_ROOT, "questions.json"),
  JSON.stringify(items, null, 2),
  "utf-8"
);

console.log(`同步完成：共 ${items.length} 题`);
console.log(`- manifest: public/question-bank/questions.json`);
