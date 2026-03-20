import re
import time
import json
import xml.etree.ElementTree as ET
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from itertools import product
from kiwipiepy import Kiwi

# ── 설정 ──────────────────────────────────────────────
SETS   = ["ARTI", "JOUR", "ARTI_CONF"]
YEARS  = range(2016, 2027)
MONTHS = range(1, 13)
DAYS   = range(1, 32)

MAX_WORKERS   = 5       # 동시 요청 수 (서버 부하 고려)
RETRY_COUNT   = 3        # 실패 시 재시도 횟수
RETRY_DELAY   = 2.0      # 재시도 대기(초)
OUTPUT_PATH   = "../json/research_data_kci.jsonl"
kiwi = Kiwi()

# ── API 호출 ───────────────────────────────────────────
def call_oai_api(url: str, session: requests.Session, retry: int = RETRY_COUNT):
    for attempt in range(retry):
        try:
            res = session.get(url, timeout=300)
            if res.status_code != 200:
                return {"success": False, "status_code": res.status_code, "url": url}
            root = ET.fromstring(res.content)
            return {"success": True, "xml_root": root}
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {"success": False, "error": str(e), "url": url}


# ── XML 파싱 ───────────────────────────────────────────
def _merge_nouns(noun_tokens: list[tuple[str, int, int]]) -> list[str]:
        """
        합성어 복원
        형태소단위로 분리된 명사 토큰들을 원문의 띄어쓰기 단위로 복원합니다.
         - 예시: [('여우', 0, 2), ('주연상', 2, 3)] → ['여우주연상']
        원문의 띄어쓰기 단위로 복원합니다.
        띄어쓰기가 잘 못 된 경우에는 오류 발생 가능성 있습니다.
        """
        if not noun_tokens:
            return []

        merged = []
        current_form, current_start, current_len = noun_tokens[0]

        for form, start, length in noun_tokens[1:]:
            if current_start + current_len == start:
                # 붙어있음 → 합성
                current_form = f"{current_form}{form}"
                current_len = (start + length) - current_start
            else:
                # 띄어있음 → 별개의 명사로 간주
                merged.append(current_form)
                current_form, current_start, current_len = form, start, length

        merged.append(current_form)
        return merged

def extract_nouns(corpus:str) -> list[str]:
        """
        corpus에서 명사 추출(띄어쓰기로 단어 구분)
        """
        tokens = kiwi.tokenize(corpus)
        processed_nouns = []
        noun = []
        
        for token in tokens:
            if "N" in token.tag:
                if token.tag == "ETN": # 명사형 전성 어미일 경우 처리 X
                    pass
                elif token.tag == "XPN": # 체언 접두어일 경우 처리 X
                    pass
                elif token.tag == "NNB": # 의존 명사 처리 X
                    pass
                elif token.tag == "NP": # 대명사 처리 X
                    pass
                elif token.tag == "XSN": # 명사 파생 접미사 처리 X
                    pass
                elif token.form[-1] == "." or token.form[-1] == ",":
                    noun.append((token.form.strip(), token.start, token.len))

                else:
                    noun.append((token.form.strip(), token.start, token.len))
            elif token.tag in ["SH", "SL", "SW"]: # 외국어 및 특수 문자 처리
                noun.append((token.form.strip(), token.start, token.len))

            else:
                if len(noun)>0:
                    processed_nouns.extend(_merge_nouns(noun))
                noun = []
        if len(noun)>0:
            processed_nouns.extend(_merge_nouns(noun))
            
        return processed_nouns
def process_records(xml_root):
    ns = {
        "oai":    "http://www.openarchives.org/OAI/2.0/",
        "dc":     "http://purl.org/dc/elements/1.1/",
        "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"
    }
    records_data = []

    for record in xml_root.findall(".//oai:record", ns):
        metadata = record.find("oai:metadata", ns)
        if metadata is None:
            continue
        dc = metadata.find("oai_dc:dc", ns)
        if dc is None:
            continue

        result_metadata = {
            "title": None, "authors": [], "category": None, "keywords": [],
            "date": None, "publisher": None, "url": None, "summary": None
        }
        page_content = None

        for title in dc.findall("dc:title", ns):
            lang = title.attrib.get("lang") or \
                   title.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            if lang == "original":
                result_metadata["title"] = title.text
                result_metadata["keywords"] = extract_nouns(title.text)

        for creator in dc.findall("dc:creator", ns):
            result_metadata["authors"].append(creator.text)
            if len(result_metadata["authors"]) == 1:
                text = creator.text
                result_metadata["authors"] = (
                    text.split(";") if ";" in text else
                    text.split("|") if "|" in text else
                    [text]
                )

        for tag, key in [("dc:subject", "category"), ("dc:date", None),
                         ("dc:publisher", "publisher"), ("dc:url", "url")]:
            el = dc.find(tag, ns)
            if el is not None:
                if tag == "dc:date":
                    result_metadata["date"] = f"{el.text}-01"
                else:
                    result_metadata[key] = el.text

        for description in dc.findall("dc:description", ns):
            lang = description.attrib.get("lang") or \
                   description.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            if lang not in ["english", "en"] and description.text:
                if re.search(r"[가-힣]", description.text):
                    page_content = description.text
                    break

        if page_content is not None:
            result_metadata["summary"] = page_content
            records_data.append({"metadata": result_metadata, "page_content": page_content})


    return records_data


# ── 작업 단위 생성 ─────────────────────────────────────
def build_tasks():
    tasks = []
    for year, month, day, set_name in product(YEARS, MONTHS, DAYS, SETS):
        date_str = f"{year}-{month:02d}-{day:02d}"
        url = (
            f"https://open.kci.go.kr/oai/request"
            f"?verb=ListRecords&set={set_name}"
            f"&from={date_str}&until={date_str}"
            f"&metadataPrefix=oai_dc"
        )
        tasks.append((url, set_name, year, month, day))
    return tasks


# ── 워커 함수 ──────────────────────────────────────────
def fetch_task(args, session):
    url, set_name, year, month, day = args
    result = call_oai_api(url, session)
    if not result["success"]:
        return []
    return process_records(result["xml_root"])


# ── 메인 ──────────────────────────────────────────────
def main():
    tasks       = build_tasks()
    write_lock  = threading.Lock()

    # 중복 제거용 seen set (url 기준)
    seen_urls   = set()
    seen_lock   = threading.Lock()

    # 결과를 즉시 파일에 flush (메모리 절약 + 무결성)
    out_file = open(OUTPUT_PATH, "w", encoding="utf-8")

    def safe_write(papers):
        """중복 제거 후 파일에 즉시 기록"""
        written = 0
        with seen_lock:
            deduped = []
            for p in papers:
                key = p["metadata"].get("url") or p["metadata"].get("title")
                if key and key not in seen_urls:
                    seen_urls.add(key)
                    deduped.append(p)
        if deduped:
            with write_lock:
                for item in deduped:
                    out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_file.flush()
            written = len(deduped)
        return written

    total_written = 0

    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
            max_retries=0          # 재시도는 call_oai_api에서 직접 처리
        )
        session.mount("https://", adapter)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(fetch_task, task, session): task
                for task in tasks
            }

            with tqdm(total=len(tasks), desc="수집 진행") as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        papers = future.result()
                        if papers:
                            written = safe_write(papers)
                            total_written += written
                    except Exception as e:
                        url = task[0]
                        tqdm.write(f"[ERROR] {url} → {e}")
                    finally:
                        pbar.update(1)
                        pbar.set_postfix({"저장됨": total_written})

    out_file.close()
    print(f"\n완료: 총 {total_written}건 저장 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()