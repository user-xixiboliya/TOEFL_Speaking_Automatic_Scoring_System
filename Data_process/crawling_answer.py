import re
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright

LECTURE_ROOT = Path("./lecture")
ANSWER_ROOT = Path("./answer")
ANSWER_ROOT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://toefl.kmf.com/",
}

AUDIO_EXTS = r"(mp3|wav|m4a|aac|ogg|webm|flac)"
USER_AUDIO_RE = re.compile(
    rf"^https?://(?:audio|img)\.kmf\.com/.*?\.{AUDIO_EXTS}(?:\?.*)?$",
    re.I
)

SCORE_RE = re.compile(r"(\d(?:\.\d)?)\s*分")
TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")

MAX_PER_QUESTION = 80
HEADLESS = False


def safe_text(locator) -> str:
    try:
        t = locator.inner_text(timeout=800)
        return re.sub(r"\s+", " ", t).strip()
    except:
        return ""


def extract_score(text: str) -> Optional[float]:
    m = SCORE_RE.search(text)
    return float(m.group(1)) if m else None


def extract_time(text: str) -> Optional[str]:
    m = TIME_RE.search(text)
    return m.group(0) if m else None


def is_user_answer_row_text(row_text: str) -> bool:
    if not SCORE_RE.search(row_text):
        return False
    if not TIME_RE.search(row_text):
        return False
    return True


def guess_ext_from_url(url: str) -> str:
    try:
        path = urlparse(url).path.lower()
        m = re.search(r"\.(mp3|wav|m4a|aac|ogg|webm|flac)$", path)
        if m:
            return m.group(1)
    except:
        pass
    return "mp3"


def download_range_complete(url: str, out_path: Path, chunk_size: int = 1024 * 256):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = -1
    try:
        r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=20)
        if r.status_code < 400:
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit():
                total = int(cl)
    except:
        pass

    if total <= 0:
        with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for ch in r.iter_content(chunk_size=chunk_size):
                    if ch:
                        f.write(ch)
        return out_path.stat().st_size

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    downloaded = tmp.stat().st_size if tmp.exists() else 0

    while downloaded < total:
        headers = dict(HEADERS)
        headers["Range"] = f"bytes={downloaded}-"
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Unexpected status {r.status_code}")
            with open(tmp, "ab") as f:
                for ch in r.iter_content(chunk_size=chunk_size):
                    if ch:
                        f.write(ch)
        downloaded = tmp.stat().st_size

    if downloaded != total:
        raise RuntimeError(f"Size mismatch got={downloaded} expected={total}")

    tmp.replace(out_path)
    return total


def switch_to_best_tab(page):
    for txt in ["优秀录音", "分数最高"]:
        try:
            loc = page.get_by_text(txt)
            if loc.count() > 0:
                loc.first.click(timeout=1500)
        except:
            pass
    page.wait_for_timeout(600)


def get_best_section(page):
    anchor = page.get_by_text("精华内容")
    if anchor.count() == 0:
        anchor = page.get_by_text("优秀录音")
    if anchor.count() == 0:
        return page.locator("body")

    section1 = anchor.first.locator("xpath=ancestor::div[1]")
    section2 = anchor.first.locator("xpath=ancestor::div[2]")
    return section2.first if len(safe_text(section2)) > len(safe_text(section1)) else section1.first


def get_rows_on_current_page(section):
    score_nodes = section.locator("text=/\\d(\\.\\d)?分/")
    n = min(score_nodes.count(), 200)

    rows = []
    for i in range(n):
        node = score_nodes.nth(i)
        # 可见性过滤，避免 hidden DOM 干扰
        try:
            if not node.is_visible():
                continue
        except:
            pass

        row = None
        for sel in ["xpath=ancestor::li[1]", "xpath=ancestor::div[1]", "xpath=ancestor::div[2]"]:
            try:
                cand = node.locator(sel)
                if cand.count() == 0:
                    continue
                r = cand.first
                try:
                    if not r.is_visible():
                        continue
                except:
                    pass
                t = safe_text(r)
                if 0 < len(t) < 900 and is_user_answer_row_text(t):
                    row = r
                    break
            except:
                pass

        if row:
            rows.append(row)

    uniq, seen = [], set()
    for r in rows:
        t = safe_text(r)
        if t and t not in seen:
            seen.add(t)
            uniq.append(r)
    return uniq


def get_play_button_in_row(row):
    selectors = [
        "[class*=play]",
        "i[class*=play]",
        "button[class*=play]",
        "svg[class*=play]",
        "text=▶",
        "[data-action*=play]",
        "[aria-label*=播放]",
        "[title*=播放]",
    ]
    for sel in selectors:
        try:
            loc = row.locator(sel)
            if loc.count() > 0:
                for i in range(min(loc.count(), 6)):
                    cand = loc.nth(i)
                    try:
                        if cand.is_visible():
                            return cand
                    except:
                        pass
                return loc.first
        except:
            pass
    return None


def capture_audio_after_click(page, row, click_fn, timeout_ms=20000) -> Optional[str]:
    found = {"url": None}

    def consider(u: Optional[str]):
        if not u or found["url"] is not None:
            return
        if USER_AUDIO_RE.match(u):
            found["url"] = u

    def on_request(req):
        consider(req.url)

    page.on("request", on_request)
    click_fn()

    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if found["url"]:
            page.remove_listener("request", on_request)
            return found["url"]

        # 兜底：播放器 currentSrc（很多题后续切换不会发新 request）
        try:
            cur = page.evaluate("""() => {
                const a = document.querySelector('audio');
                return a ? (a.currentSrc || a.src || '') : '';
            }""")
            consider(cur)
        except:
            pass

        # 兜底：row html 里直接包含音频链接
        try:
            html = row.inner_html(timeout=300)
            m = re.search(r"https?://(?:audio|img)\.kmf\.com/[^\"']+\.(?:mp3|wav|m4a|aac|ogg|webm|flac)(?:\?[^\"']*)?", html, re.I)
            if m:
                consider(m.group(0))
        except:
            pass

        page.wait_for_timeout(200)

    page.remove_listener("request", on_request)
    return None


# ✅ 关键：自动找到真正滚动容器，或优先点“加载更多”，确保列表加载完整
def ensure_list_fully_loaded(page, section, max_rounds: int = 20):
    """
    目标：让 'X分' 节点数量增长到稳定（>1 且不再增长）。
    1) 优先点击“加载更多/更多/展开”等按钮（如果存在）
    2) 若无按钮或按钮点完仍不增长：从第一个分数节点向上找最近可滚动祖先并滚动它
    """
    def count_scores() -> int:
        try:
            return section.locator("text=/\\d(\\.\\d)?分/").count()
        except:
            return 0

    def click_load_more_once() -> bool:
        # 常见“加载更多”文案（按需可再加）
        candidates = ["加载更多", "更多", "展开更多", "点击加载更多", "查看更多"]
        for txt in candidates:
            try:
                btn = section.get_by_text(txt)
                if btn.count() > 0:
                    b = btn.first
                    try:
                        if b.is_visible():
                            b.click(timeout=1200)
                            page.wait_for_timeout(700)
                            return True
                    except:
                        pass
            except:
                pass
        return False

    last = -1
    stable = 0

    for _ in range(max_rounds):
        cur = count_scores()
        if cur == last and cur > 1:
            stable += 1
        else:
            stable = 0
        last = cur
        if stable >= 2:
            break

        # 1) 先尝试点“加载更多”
        if click_load_more_once():
            continue

        # 2) 找“最近可滚动祖先”并滚动
        try:
            score_loc = section.locator("text=/\\d(\\.\\d)?分/").first
            handle = score_loc.element_handle(timeout=800)
            if handle:
                # 在页面里向上找可滚动祖先
                scrollable = handle.evaluate_handle("""
                    (el) => {
                      function isScrollable(x){
                        if(!x) return false;
                        const st = getComputedStyle(x);
                        const oy = st.overflowY;
                        const can = (oy === 'auto' || oy === 'scroll');
                        return can && x.scrollHeight > x.clientHeight + 10;
                      }
                      let p = el;
                      for(let i=0;i<12;i++){
                        if(isScrollable(p)) return p;
                        p = p.parentElement;
                        if(!p) break;
                      }
                      // 找不到就退回 body
                      return document.scrollingElement || document.body;
                    }
                """)
                # 滚动该容器到底部，触发懒加载
                scrollable.evaluate("""(c) => { c.scrollTop = c.scrollHeight; }""")
                page.wait_for_timeout(700)
                scrollable.evaluate("""(c) => { c.scrollTop = c.scrollHeight; }""")
                page.wait_for_timeout(700)
                continue
        except:
            pass

        # 3) 最后兜底：滚页面
        page.mouse.wheel(0, 1800)
        page.wait_for_timeout(700)


def click_next_page(page, section) -> bool:
    try:
        next_btn = section.get_by_text("下一页")
        if next_btn.count() == 0:
            return False
        btn = next_btn.first
        cls = (btn.get_attribute("class") or "").lower()
        aria = (btn.get_attribute("aria-disabled") or "").lower()
        if "disabled" in cls or aria == "true":
            return False

        before = ""
        try:
            before = safe_text(section.locator("text=/\\d{4}-\\d{2}-\\d{2}/").first)
        except:
            pass

        btn.click(timeout=1500)
        page.wait_for_timeout(900)

        for _ in range(30):
            now = ""
            try:
                now = safe_text(section.locator("text=/\\d{4}-\\d{2}-\\d{2}/").first)
            except:
                pass
            if now and now != before:
                return True
            page.wait_for_timeout(250)
        return True
    except:
        return False


def clean_records(records: List[Dict]) -> List[Dict]:
    good = []
    for r in records:
        if not r.get("mp3_url"):
            continue
        if not r.get("local_mp3"):
            continue
        b = r.get("bytes")
        if not isinstance(b, int) or b <= 0:
            continue
        good.append(r)

    best = {}
    for r in good:
        k = r["mp3_url"]
        if k not in best:
            best[k] = r
        else:
            s1 = best[k].get("score") or -1
            s2 = r.get("score") or -1
            if s2 > s1:
                best[k] = r

    out = list(best.values())
    out.sort(key=lambda x: (x.get("page", 999), x.get("rank_in_page", 999)))
    return out


def load_jobs_from_lecture_dir() -> List[Tuple[str, str]]:
    jobs = []
    for official_dir in sorted([p for p in LECTURE_ROOT.iterdir() if p.is_dir()]):
        official_id = official_dir.name
        jpath = official_dir / "1.json"
        if not jpath.exists():
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except:
            continue
        url = data.get("source_url") or data.get("detail_url")
        if url and "/detail/speak/" in url:
            jobs.append((official_id, url))
    return jobs


def already_done(official_id: str) -> bool:
    out_dir = ANSWER_ROOT / official_id
    out_json = out_dir / "1.json"
    if not out_json.exists():
        return False
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        recs = data.get("records") or []
        if not recs:
            return False
    except:
        return False

    for ext in ["mp3", "wav", "m4a", "aac", "ogg", "webm", "flac"]:
        if len(list(out_dir.glob(f"1_*.{ext}"))) > 0:
            return True
    return False


def crawl_one_question(page, official_id: str, detail_url: str) -> int:
    out_dir = ANSWER_ROOT / official_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "1_raw.json"
    clean_path = out_dir / "1.json"

    page.goto(detail_url, wait_until="networkidle", timeout=60000)
    switch_to_best_tab(page)

    raw_records: List[Dict] = []
    seen_mp3: Set[str] = set()
    downloaded = 0

    page_idx = 1
    while True:
        switch_to_best_tab(page)
        section = get_best_section(page)

        # ✅ 用新方法加载完整列表（解决 rows=1）
        ensure_list_fully_loaded(page, section)

        rows = get_rows_on_current_page(section)
        if len(rows) == 0:
            rows = get_rows_on_current_page(page.locator("body"))

        print(f"[{official_id}] page {page_idx} rows={len(rows)}")

        for ri, row in enumerate(rows, start=1):
            if downloaded >= MAX_PER_QUESTION:
                break

            try:
                row.scroll_into_view_if_needed(timeout=2000)
            except:
                pass
            page.wait_for_timeout(120)

            row_text = safe_text(row)
            score = extract_score(row_text)
            tstr = extract_time(row_text)

            play_btn = get_play_button_in_row(row)

            def do_click():
                pb = play_btn if play_btn is not None else get_play_button_in_row(row)
                if pb:
                    pb.click(timeout=1200, force=True)
                else:
                    row.click(timeout=1200, force=True)

            mp3 = capture_audio_after_click(page, row, do_click, timeout_ms=20000)

            rec = {
                "page": page_idx,
                "rank_in_page": ri,
                "score": score,
                "time": tstr,
                "mp3_url": mp3,
                "local_mp3": None,
                "bytes": None,
                "note": None,
            }

            if not mp3:
                rec["note"] = "No audio captured after click."
                raw_records.append(rec)
                continue

            if mp3 in seen_mp3:
                rec["note"] = "Duplicate audio url (captured)."
                raw_records.append(rec)
                continue

            seen_mp3.add(mp3)
            downloaded += 1

            ext = guess_ext_from_url(mp3)
            out_file = out_dir / f"1_{downloaded:02d}.{ext}"

            try:
                size = download_range_complete(mp3, out_file)
                rec["local_mp3"] = out_file.as_posix()
                rec["bytes"] = size
            except Exception as e:
                rec["note"] = f"Download failed: {e}"

            raw_records.append(rec)
            page.wait_for_timeout(200)

        if downloaded >= MAX_PER_QUESTION:
            break

        moved = click_next_page(page, section)
        if not moved:
            break
        page_idx += 1
        page.wait_for_timeout(800)

    raw_path.write_text(json.dumps({
        "official_id": official_id,
        "detail_url": detail_url,
        "q": 6,
        "records": raw_records,
        "note": "raw records (未清洗，含无音频/错误/重复)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    cleaned = clean_records(raw_records)
    clean_path.write_text(json.dumps({
        "official_id": official_id,
        "detail_url": detail_url,
        "q": 6,
        "records": cleaned,
        "count_raw": len(raw_records),
        "count_clean": len(cleaned),
        "note": "cleaned records（仅保留真实用户回答：捕获+下载成功；按mp3_url去重保留高分）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(cleaned)


def main():
    jobs = load_jobs_from_lecture_dir()
    print("Total jobs:", len(jobs))
    if not jobs:
        print("No jobs found in ./lecture")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page()

        ok, fail, skip = 0, 0, 0
        for idx, (official_id, detail_url) in enumerate(jobs, start=1):
            if already_done(official_id):
                print(f"[{idx}/{len(jobs)}] SKIP {official_id}")
                skip += 1
                continue
            try:
                n = crawl_one_question(page, official_id, detail_url)
                print(f"[{idx}/{len(jobs)}] OK   {official_id} saved={n}")
                ok += 1
                time.sleep(0.2)
            except Exception as e:
                print(f"[{idx}/{len(jobs)}] FAIL {official_id} reason={e}")
                fail += 1
                time.sleep(0.6)

        browser.close()

    print(f"DONE ok={ok} fail={fail} skip={skip}")


if __name__ == "__main__":
    main()
