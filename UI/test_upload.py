import json
import os
import sys
import requests

BASE_URL = "https://mailto-sitting-victorian-representing.trycloudflare.com"
API_URL = f"{BASE_URL}/api/evaluate"

AUDIO_PATH = r".\test\answer.mp3"
TRANSCRIPT_PATH = r".\test\lecture.txt"
TASK_PATH = r".\test\task.txt"

HEADERS = {}

def must_exist(p: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

def main():
    for p in [AUDIO_PATH, TRANSCRIPT_PATH, TASK_PATH]:
        must_exist(p)

    with open(AUDIO_PATH, "rb") as fa, open(TRANSCRIPT_PATH, "rb") as ft, open(TASK_PATH, "rb") as fk:
        files = {
            "audio": (os.path.basename(AUDIO_PATH), fa),
            "transcript": (os.path.basename(TRANSCRIPT_PATH), ft),
            "task": (os.path.basename(TASK_PATH), fk),
        }
        r = requests.post(API_URL, files=files, headers=HEADERS, timeout=600)

    print("HTTP:", r.status_code)
    print(r.text)

    if r.ok:
        try:
            data = r.json()
            with open("result.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("Saved -> result.json")
        except Exception as e:
            print("JSON parse failed:", e)

if __name__ == "__main__":
    main()
