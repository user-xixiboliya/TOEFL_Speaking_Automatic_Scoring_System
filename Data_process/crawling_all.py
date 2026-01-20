import re
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import sync_playwright

BASE = "https://toefl.kmf.com"

START_PAGE = 1
END_PAGE = 11
LIST_URL_TMPL = BASE + "/speak/ets/new-order/{page}/0"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://toefl.kmf.com/",
}

LECTURE_DIR = Path("lecture")
TASK_DIR = Path("task")

DETAIL_RE = re.compile(r"^/detail/speak/[0-9a-zA-Z]+\.html$")

LISTENING_AUDIO_RE = re.compile(
    r"^https://img\.kmf\.com/toefl/listening/audio/[0-9a-f]{32}\.mp3$",
    re.I
)
SPEAKING_TPO_AUDIO_RE = re.compile(
    r"^https://img\.kmf\.com/toefl/speaking/TPO-\d{1,2}-Q6-[A-Za-z0-9_-]+\.mp3$",
    re.I
)

SPEAKING_TPO_AUDIO_IN_HTML_RE = re.compile(
    r"https://img\.kmf\.com/toefl/speaking/TPO-\d{1,2}-Q6-[A-Za-z0-9_-]+\.mp3",
    re.I
)


def get_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_q6_detail_urls(list_html: str):
    soup = BeautifulSoup(list_html, "html.parser")
    q6_urls = []

    q6_pattern = re.compile(r"\bQ\s*6\b(?!\d)")

    for node in soup.find_all(string=q6_pattern):
        container = node
        link = None

        for _ in range(10):
            if container is None:
                break

            if isinstance(container, Tag):
                a = container.find("a", href=re.compile(r"/detail/speak/"))
                if a:
                    link = a
                    break

            container = container.parent

        if link:
            href = link.get("href", "").strip()
            if href:
                full = href if href.startswith("http") else BASE + href
                q6_urls.append(full)

    seen = set()
    out = []
    for u in q6_urls:
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out

def clean_lecture(lecture: str) -> str:
    lecture = lecture.replace("\r\n", "\n").replace("\r", "\n").strip()
    lecture = re.sub(r"^(查看听力原文|听力原文|查看原文|Transcript)\s*\n+", "", lecture)
    lecture = re.sub(r"^<-\s*[A-Z\s]+:\s*->\s*", "", lecture)  
    lecture = re.sub(r"\n{3,}", "\n\n", lecture).strip()
    return lecture


def parse_lecture_task_from_text(text: str):
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    titles = ["查看听力原文", "听力原文", "查看原文", "Transcript"]
    start_pos, start_title = None, None
    for title in titles:
        p = t.find(title)
        if p != -1:
            start_pos, start_title = p, title
            break
    if start_pos is None:
        raise RuntimeError("Cannot find transcript title.")

    m_end = re.search(r"\n问题\s*\n", t[start_pos:], flags=re.S)
    if not m_end:
        raise RuntimeError("Cannot find '问题' delimiter.")

    block = t[start_pos:start_pos + m_end.start()]
    if start_title in block:
        block = block.split(start_title, 1)[-1]

    lecture_text = clean_lecture(block)

    m_task = re.search(r"\n问题\s*\n(.*?)(\n{2,}|$)", t, flags=re.S)
    if not m_task:
        raise RuntimeError("Cannot extract task after '问题'.")
    task = m_task.group(1).strip()

    return lecture_text, task


def parse_detail_q6(detail_url: str, official_id_hint: str | None = None):
    html = get_html(detail_url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)

    official_id = official_id_hint
    if not official_id:
        m = re.search(r"Official\s+(\d+)\s+Q\s*6", text)
        if not m:
            raise RuntimeError("Not Q6 or blocked.")
        official_id = f"{int(m.group(1)):03d}"

    sm = re.search(r"/detail/speak/([0-9a-zA-Z]+)\.html", detail_url)
    slug = sm.group(1) if sm else ""

    lecture_text, task = parse_lecture_task_from_text(text)

    tpo_mp3s = SPEAKING_TPO_AUDIO_IN_HTML_RE.findall(html)
    tpo_mp3 = tpo_mp3s[0] if tpo_mp3s else None

    return {
        "official_id": official_id,
        "slug": slug,
        "detail_url": detail_url,
        "lecture_text": lecture_text,
        "task": task,
        "tpo_mp3_hint": tpo_mp3,
    }


def safe_write_json(path: Path, new_data: dict):
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(old)
        for k, v in new_data.items():
            if k not in merged or merged[k] in (None, "", [], {}):
                merged[k] = v
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dirs(official_id: str):
    (TASK_DIR / official_id).mkdir(parents=True, exist_ok=True)
    (LECTURE_DIR / official_id).mkdir(parents=True, exist_ok=True)

def head_total_size(url: str) -> int:
    r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=20)
    r.raise_for_status()
    cl = r.headers.get("Content-Length")
    return int(cl) if cl and cl.isdigit() else -1


def download_range_complete(url: str, out_path: Path, chunk_size: int = 1024 * 256, max_retries: int = 5):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = head_total_size(url)
    if total <= 0:
        raise RuntimeError("Cannot get Content-Length from HEAD")

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    downloaded = tmp.stat().st_size if tmp.exists() else 0

    retries = 0
    while downloaded < total:
        headers = dict(HEADERS)
        headers["Range"] = f"bytes={downloaded}-"
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Unexpected status {r.status_code}")
            with open(tmp, "ab") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        new_size = tmp.stat().st_size
        if new_size == downloaded:
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(f"Stuck downloading. downloaded={downloaded}, total={total}")
            time.sleep(0.8)
        else:
            retries = 0
            downloaded = new_size

    if downloaded != total:
        raise RuntimeError(f"Size mismatch. got={downloaded}, expected={total}")

    tmp.replace(out_path)
    return total

def capture_audio_url(page, timeout_ms: int = 25000) -> str:
    found = {"url": None}

    def on_request(req):
        u = req.url
        if found["url"] is not None:
            return
        if LISTENING_AUDIO_RE.match(u):
            found["url"] = u
            return
        if SPEAKING_TPO_AUDIO_RE.match(u):
            found["url"] = u
            return

    page.on("request", on_request)

    for txt in ["播放", "开始", "开始练习", "开始精听", "立即练习", "开始听写"]:
        try:
            loc = page.get_by_text(txt)
            if loc.count() > 0:
                loc.first.click(timeout=1500)
                break
        except:
            pass

    try:
        page.keyboard.press("Space")
    except:
        pass

    try:
        page.mouse.click(600, 400)
        page.keyboard.press("Space")
    except:
        pass

    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if found["url"]:
            return found["url"]
        page.wait_for_timeout(200)

    raise RuntimeError("Timed out: did not capture audio request.")


def fetch_audio(browser, official_id: str, slug: str, tpo_mp3_hint: str | None):
    out_mp3 = LECTURE_DIR / official_id / "1.mp3"

    if tpo_mp3_hint and (LISTENING_AUDIO_RE.match(tpo_mp3_hint) or SPEAKING_TPO_AUDIO_RE.match(tpo_mp3_hint)):
        total = download_range_complete(tpo_mp3_hint, out_mp3)
        return tpo_mp3_hint, out_mp3.as_posix(), total

    page_url = f"{BASE}/listening/newdrilling/{slug}"
    page = browser.new_page()
    try:
        page.goto(page_url, wait_until="networkidle", timeout=60000)
        audio_url = capture_audio_url(page, timeout_ms=30000)
        total = download_range_complete(audio_url, out_mp3)
        return audio_url, out_mp3.as_posix(), total
    finally:
        page.close()

def main():
    all_detail_urls = []
    for p in range(START_PAGE, END_PAGE + 1):
        list_url = LIST_URL_TMPL.format(page=p)
        try:
            list_html = get_html(list_url)
        except Exception as e:
            print(f"[list] FAIL page={p} url={list_url} reason:", e)
            continue

        q6_urls = extract_q6_detail_urls(list_html)
        print(f"[list] page={p} Q6={len(q6_urls)}")
        all_detail_urls.extend(q6_urls)
        time.sleep(0.2)

    seen = set()
    detail_urls = []
    for u in all_detail_urls:
        if u not in seen:
            seen.add(u)
            detail_urls.append(u)

    print("[list] total unique Q6 detail urls:", len(detail_urls))

    items_for_audio = []

    for i, detail_url in enumerate(detail_urls, start=1):
        try:
            item = parse_detail_q6(detail_url)
            official_id = item["official_id"]
            slug = item.get("slug", "")

            lecture_json_path = LECTURE_DIR / official_id / "1.json"
            task_json_path = TASK_DIR / official_id / "1.json"
            mp3_path = LECTURE_DIR / official_id / "1.mp3"

            if lecture_json_path.exists():
                try:
                    old = json.loads(lecture_json_path.read_text(encoding="utf-8"))
                    if old.get("lecture_text"):
                        print(f"[detail] skip existing lecture_text official={official_id}")
                        if (not mp3_path.exists()) or (not old.get("audio_url")):
                            items_for_audio.append({
                                "official_id": official_id,
                                "slug": old.get("slug") or slug,
                                "tpo_mp3_hint": old.get("tpo_mp3_hint"),
                            })
                        continue
                except:
                    pass

            (TASK_DIR / official_id).mkdir(parents=True, exist_ok=True)
            (LECTURE_DIR / official_id).mkdir(parents=True, exist_ok=True)

            safe_write_json(task_json_path, {
                "official": official_id,
                "q": 6,
                "slug": slug,
                "source_url": item.get("detail_url", detail_url),
                "task": item["task"],
            })

            safe_write_json(lecture_json_path, {
                "official": official_id,
                "q": 6,
                "slug": slug,
                "source_url": item.get("detail_url", detail_url),
                "lecture_text": item["lecture_text"],
                "audio_url": None,
                "local_audio": None,
                "audio_bytes": None,
                "tpo_mp3_hint": item.get("tpo_mp3_hint"),
            })

            items_for_audio.append(item)
            print(f"[detail] {i}/{len(detail_urls)} OK official={official_id}")
            time.sleep(0.25)

        except Exception as e:
            print("[detail] FAIL:", detail_url, "reason:", e)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for j, item in enumerate(items_for_audio, start=1):
            official_id = item["official_id"]
            lecture_json_path = LECTURE_DIR / official_id / "1.json"
            mp3_path = LECTURE_DIR / official_id / "1.mp3"

            old = {}
            if lecture_json_path.exists():
                old = json.loads(lecture_json_path.read_text(encoding="utf-8"))

            if mp3_path.exists() and mp3_path.stat().st_size > 300_000 and old.get("audio_url"):
                print(f"[audio] skip existing official={official_id}")
                continue

            slug = item.get("slug") or old.get("slug")
            if not slug:
                print(f"[audio] FAIL official={official_id} reason: missing slug")
                continue

            try:
                audio_url, local_audio, total = fetch_audio(
                    browser=browser,
                    official_id=official_id,
                    slug=slug,
                    tpo_mp3_hint=old.get("tpo_mp3_hint") or item.get("tpo_mp3_hint"),
                )

                safe_write_json(lecture_json_path, {
                    "audio_url": audio_url,
                    "local_audio": local_audio,
                    "audio_bytes": total,
                })

                print(f"[audio] {j}/{len(items_for_audio)} OK official={official_id} bytes={total}")
                time.sleep(0.3)

            except Exception as e:
                print(f"[audio] FAIL official={official_id} slug={slug} reason:", e)

        browser.close()

    print("DONE.")

if __name__ == "__main__":
    main()
