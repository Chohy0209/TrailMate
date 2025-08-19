# -*- coding: utf-8 -*-
"""
Chroma RAG Top-2 → 네이버 검색(sim) → 최신순 후보 중 본문 파싱 성공하는 1개만 문서당 채택
DB 메타 '수정날짜'보다 최신인 글만 허용, 본문 파싱 실패 글은 제외
"""

import os
import re
import asyncio
import requests
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# 환경설정
# -----------------------------
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise RuntimeError("NAVER 키 없음: .env에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 설정 필요")

CHROMA_DB_PATH = "./camp_vectorDB_BGE_sentence_per_doc"
EMBEDDING_MODEL_NAME = "dragonkue/BGE-m3-ko"
NAVER_CONCURRENCY = 3
_rate_sem = asyncio.Semaphore(NAVER_CONCURRENCY)

# naver_api.py 상단
ONLY_UTILS = True  # 메인에서만 RAG 쓰고, 여기선 검색/파싱 함수만 쓸 때

if not ONLY_UTILS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# -----------------------------
# 벡터DB 로드 & Top-2 헬퍼
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def get_top2_docs(vdb: Chroma, query: str, camping_type: Optional[str] = None) -> List[Tuple[Document, float]]:
    has_with_score = hasattr(vdb, "similarity_search_with_score")
    if has_with_score:
        try:
            if camping_type:
                return vdb.similarity_search_with_score(query=query, k=2, filter={"캠핑유형": camping_type})
            return vdb.similarity_search_with_score(query=query, k=2)
        except TypeError:
            return vdb.similarity_search_with_score(query=query, k=2)
    docs = vdb.similarity_search(query=query, k=2, filter={"캠핑유형": camping_type} if camping_type else None)
    return [(d, 0.0) for d in docs]

# -----------------------------
# 문자열/날짜 유틸
# -----------------------------
def clean_html_tags(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text or "")
    text = (text.replace('&amp;', '&')
                .replace('&lt;', '<')
                .replace('&gt;', '>')
                .replace('&quot;', '"')
                .replace('&apos;', "'"))
    return text.strip()

def _normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = s.strip()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_any_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        try:
            y, mo, d = int(digits[0:4]), int(digits[4:6]), int(digits[6:8])
            return datetime(y, mo, d, tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        pass
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _item_datetime(item: Dict[str, Any]) -> Optional[datetime]:
    for k in ("pubDate", "postdate"):
        v = item.get(k)
        if not v:
            continue
        dt = _parse_any_date(v)
        if dt:
            return dt
    return None

def _latest_modified_from_meta(meta: Dict[str, Any]) -> Optional[datetime]:
    for k in ("수정날짜", "수정일", "modified_at", "updated_at"):
        dt = _parse_any_date(meta.get(k))
        if dt:
            return dt
    return None

# -----------------------------
# 네이버 API
# -----------------------------
async def _http_get_with_retry(url: str, headers: Dict[str, str], params: Dict[str, Any],
                               timeout: int = 7, max_retries: int = 3) -> requests.Response:
    delay = 0.8
    for _ in range(max_retries):
        try:
            resp = await asyncio.to_thread(requests.get, url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504):
                retry_after = e.response.headers.get("Retry-After") if e.response is not None else None
                try:
                    wait_s = int(retry_after) if retry_after else delay
                except Exception:
                    wait_s = delay
                await asyncio.sleep(wait_s)
                delay *= 2
                continue
            raise
    resp = await asyncio.to_thread(requests.get, url, headers, params, timeout=timeout)
    resp.raise_for_status()
    return resp

async def naver_search_api(query: str, search_type: str, display: int = 20, sort: str = "sim") -> List[Dict[str, Any]]:
    """
    search_type: 'blog' | 'cafearticle' | 'webkr'
    cafearticle은 'sim'만 안정적 → 기본 'sim'
    """
    if search_type not in {"webkr", "blog", "cafearticle"}:
        raise ValueError("search_type must be 'webkr', 'blog', or 'cafearticle'")

    eff_sort = "sim" if search_type == "cafearticle" else sort
    url = f"https://openapi.naver.com/v1/search/{search_type}.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query, "display": display, "start": 1, "sort": eff_sort}

    async with _rate_sem:
        resp = await _http_get_with_retry(url, headers, params, timeout=7)

    items = resp.json().get("items", [])
    out = []
    for it in items:
        out.append({
            "source": f"네이버-{search_type}",
            "title": clean_html_tags(it.get("title", "")),
            "description": clean_html_tags(it.get("description", "")),
            "link": it.get("link", ""),
            "pubDate": it.get("pubDate"),
            "postdate": it.get("postdate"),
            "search_type": search_type,
            "query": query
        })
    return out

# -----------------------------
# 본문 추출 유틸
# -----------------------------
def to_mobile_if_naver(url: str) -> str:
    o = urlparse(url); host = o.netloc.lower(); q = parse_qs(o.query); path = o.path
    if "blog.naver.com" in host and "m.blog.naver.com" not in host:
        blog_id = q.get("blogId", [None])[0]; log_no = q.get("logNo", [None])[0]
        if not (blog_id and log_no):
            m = re.match(r"^/([^/]+)/(\d+)", path)
            if m: blog_id, log_no = m.group(1), m.group(2)
        return f"https://m.blog.naver.com/{blog_id}/{log_no}" if (blog_id and log_no) \
               else url.replace("://blog.naver.com", "://m.blog.naver.com")
    if "cafe.naver.com" in host and "m.cafe.naver.com" not in host:
        clubid = q.get("clubid", [None])[0]; articleid = q.get("articleid", [None])[0]
        if clubid and articleid:
            return f"https://m.cafe.naver.com/ArticleRead.nhn?clubid={clubid}&articleid={articleid}"
        m = re.match(r"^/([^/]+)/(\d+)", path)
        if m: return f"https://m.cafe.naver.com/{m.group(1)}/{m.group(2)}"
        return url.replace("://cafe.naver.com", "://m.cafe.naver.com")
    return url

def extract_main_text_from_html(url: str, html: str) -> str:
    def _clean(s):
        s = re.sub(r"\u200b", "", s); s = re.sub(r"[ \t]+", " ", s)
        return re.sub(r"\s*\n\s*\n\s*", "\n\n", s).strip()

    soup = BeautifulSoup(html, "html.parser"); host = urlparse(url).netloc.lower()

    if "m.blog.naver.com" in host:
        for sel in ["div.se-main-container", "#postViewArea", "#viewTypeSelector"]:
            node = soup.select_one(sel)
            if node: return _clean(node.get_text("\n"))
        return _clean(soup.get_text("\n"))

    if "m.cafe.naver.com" in host:
        for sel in ["div.se-main-container", "#tbody", "#app"]:
            node = soup.select_one(sel)
            if node: return _clean(node.get_text("\n"))
        return _clean(soup.get_text("\n"))

    for sel in ["article", "main", "div#content", "div.article", "section.content"]:
        node = soup.select_one(sel)
        if node: return _clean(node.get_text("\n"))
    return _clean(soup.get_text("\n"))

async def fetch_article_text(url: str, timeout: int = 8, max_chars: int = 4000) -> Optional[str]:
    mob = to_mobile_if_naver(url)
    try:
        resp = await asyncio.to_thread(
            requests.get, mob,
            headers={"User-Agent":"Mozilla/5.0", "Referer":"https://m.naver.com/"},
            timeout=timeout
        )
        resp.raise_for_status()
        text = extract_main_text_from_html(mob, resp.text)
        txt = text[:max_chars].strip() if text else None
        return txt if txt else None
    except Exception as e:
        print("fetch failed:", e)
        return None

# -----------------------------
# 핵심: 문서당 1개만, 본문 있는 글만 선택
# -----------------------------
async def fetch_single_best_with_content_per_doc(
    docs_with_metadata: List[Tuple[Document, float]],
    per_type_display: int = 20,
    include_when_no_local_modified: bool = True,
    enforce_name_in_title: bool = True,
    fetch_timeout: int = 8,
    max_chars: int = 3000,
) -> List[Dict[str, Any]]:
    """
    각 RAG 문서(캠핑장)마다:
      - sim 정렬로 blog/cafe/webkr 넉넉히 수집
      - 날짜 없는 결과 제거 → (옵션) 제목에 캠핑장이름 포함 → 최신순
      - 로컬 '수정날짜' 초과인 후보들 중 '본문 파싱 성공'하는 첫 글만 채택
      - 본문 실패 시 그 후보 버리고 다음 후보 시도
    반환: 문서당 최대 1개, content_text 포함된 항목들
    """
    results: List[Dict[str, Any]] = []
    seen_links = set()

    for doc, _score in docs_with_metadata:
        meta = doc.metadata or {}
        camp_name = _normalize(meta.get("캠핑장이름") or meta.get("name") or "")
        camp_addr = _normalize(meta.get("캠핑장주소") or meta.get("address") or "")
        if not camp_name:
            continue

        base_query = f"{camp_name} {camp_addr}".strip()
        local_dt = _latest_modified_from_meta(meta)

        print(f"\n[DEBUG] 캠핑장: {camp_name} | 주소: {camp_addr} | 로컬 수정일: {local_dt}")

        # sim 정렬 수집
        tasks = [
            naver_search_api(base_query, "blog",        display=per_type_display, sort="sim"),
            naver_search_api(base_query, "cafearticle", display=per_type_display, sort="sim"),
            naver_search_api(base_query, "webkr",       display=per_type_display, sort="sim"),
        ]
        packs = await asyncio.gather(*tasks, return_exceptions=True)
        merged: List[Dict[str, Any]] = []
        for p in packs:
            if isinstance(p, Exception):
                print(f"네이버 오류[{camp_name}]: {p}")
                continue
            merged.extend(p)
        print(f"[DEBUG] {camp_name} 후보 총 {len(merged)}개 수집")
        # 후보 정리: 날짜 없는 항목 제거
        merged = [it for it in merged if _item_datetime(it) is not None]

        # 제목에 캠핑장이름 포함 (옵션)
        if enforce_name_in_title:
            nm = camp_name
            merged = [it for it in merged if nm in _normalize(it.get("title", ""))]
            print(f"[DEBUG] {camp_name} 날짜 있는 후보: {len(merged)}개")

        # 최신순
        merged.sort(key=lambda it: _item_datetime(it), reverse=True)

        # 로컬 수정일 초과 필터
        def _is_fresh(it: Dict[str, Any]) -> bool:
            if local_dt is None:
                return include_when_no_local_modified
            web_dt = _item_datetime(it)
            return (web_dt and web_dt > local_dt)

        candidates = [it for it in merged if _is_fresh(it)]
        print(f"[DEBUG] {camp_name} 날짜 필터 통과 후보: {len(candidates)}개")

        # 본문 파싱 성공하는 첫 글만 채택
        picked = None
        for it in candidates:
            link = it.get("link", "")
            if not link or link in seen_links:
                continue
            content = await fetch_article_text(link, timeout=fetch_timeout, max_chars=max_chars)
            if content:
                it["content_text"] = content
                it["matched_camp_name"] = camp_name
                it["matched_camp_address"] = camp_addr
                it["local_modified_at"] = local_dt.isoformat() if local_dt else None
                results.append(it)
                seen_links.add(link)
                picked = it
                print(f"[DEBUG] {camp_name} 본문 파싱 성공: {link} | 길이: {len(content)}")
                break  # 문서당 1개만
            else:
                print(f"[DEBUG] {camp_name} 본문 파싱 실패: {link}")

                 # 이 문서에서 본문 성공 글이 하나도 없으면 그냥 스킵(아무것도 추가 안 함)
        if not picked:
             print(f"[DEBUG] {camp_name} 본문 파싱 성공 글 없음")
       

    # 전체 결과는 최신순 정렬
    results.sort(key=lambda x: _item_datetime(x), reverse=True)
    print(f"[DEBUG] 최종 선택된 글 개수: {len(results)}\n")
    return results

# -----------------------------
# 데모 실행 (LLM 없음)
# -----------------------------
async def demo_run():
    query = input("질문: ").strip()
    camping_type = input("캠핑유형 필터(엔터=전체, 예:'유료캠핑장'/'글램핑/카라반'/'오지/노지캠핑'): ").strip() or None

    docs_with_metadata = get_top2_docs(vectordb, query=query, camping_type=camping_type)
    print(f"\n[Top-2 로컬문서]\n")
    for i, (doc, score) in enumerate(docs_with_metadata, start=1):
        m = doc.metadata or {}
        print(f"- ({i}) score={score:.4f} | 이름={m.get('캠핑장이름') or m.get('name')} | 수정날짜={m.get('수정날짜')}")
        print(f"  주소={m.get('캠핑장주소') or m.get('address')}\n")

    # 문서당 1개, 본문 있는 글만
    selected_items = await fetch_single_best_with_content_per_doc(
        docs_with_metadata,
        per_type_display=20,
        include_when_no_local_modified=True,
        enforce_name_in_title=True,
        fetch_timeout=8,
        max_chars=1500,
    )

    if selected_items:
        print("\n[선택된 최신 글(문서당 최대 1개) + 본문 일부]\n")
        for i, it in enumerate(selected_items, start=1):
            dt = _item_datetime(it)
            dt_str = dt.strftime("%Y-%m-%d") if dt else "N/A"
            title = it.get("title", "")
            link = it.get("link", "")
            content = (it.get("content_text") or "").strip()
            preview = content[:800].replace("\n", " ").strip()

            print(f"{i}. {it['matched_camp_name']} | {it['source']} | {dt_str}")
            print(f"   제목: {title}")
            print(f"   링크: {link}")
            print(f"   본문: {preview}...\n")
    else:
        print("\n(선택 가능한 최신 글이 없거나, 본문 파싱 실패로 모두 제외됨)\n")

        # =========================================
# 키워드 문장 추출 + 문서당 1개 글 선택 헬퍼
# =========================================
# === 키워드 문장 추출 ===
KEYWORD_PATTERNS = [
    r"가격", r"요금", r"비용",
    r"인원", r"추가인원", r"기본\s*인원",
    r"체크인", r"체크아웃", r"입실", r"퇴실",
    r"운영시간", r"영업시간", r"오픈", r"마감", r"매너\s*타임",
    r"예약", r"문의", r"전화", r"연락처"
]
KEYWORD_REGEX = re.compile("|".join(KEYWORD_PATTERNS), re.I)

def _sent_tokenize_kr(text: str) -> List[str]:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    s = re.sub(r"([\.!\?])\s+", r"\1\n", s)
    sents = [x.strip() for x in s.split("\n") if x.strip()]
    return [x for x in sents if len(x) >= 6]

def extract_keyword_sentences(text: str) -> List[str]:
    if not text:
        return []
    sents = _sent_tokenize_kr(text)
    hits = []
    for s in sents:
        if KEYWORD_REGEX.search(s):
            if re.search(r"\d", s) or re.search(r"(시|분|원|만원)", s):
                hits.append(re.sub(r"\s+", " ", s).strip())
    # 중복 제거
    out, seen = [], set()
    for s in hits:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

# === 문서당 1개 글만 뽑아서 '장소이름/링크주소/본문내용(키워드문장만)'으로 리턴 ===
async def build_snippet_per_doc(
    docs_with_metadata: List[Tuple[Document, float]],
    per_type_display: int = 20,
    fetch_timeout: int = 8,
    max_chars: int = 2000,
    enforce_name_in_title: bool = True,
    include_when_no_local_modified: bool = True,
) -> List[Dict[str, str]]:
    # 1) 각 문서당 본문 파싱 성공하는 최신 글 1개 고르기
    picked_items = await fetch_single_best_with_content_per_doc(
        docs_with_metadata,
        per_type_display=per_type_display,
        include_when_no_local_modified=include_when_no_local_modified,
        enforce_name_in_title=enforce_name_in_title,
        fetch_timeout=fetch_timeout,
        max_chars=max_chars,
    )

    # 2) 본문에서 키워드 포함 문장만 추출
    results: List[Dict[str, str]] = []
    for it in picked_items:
        camp_name = it.get("matched_camp_name") or ""
        link = it.get("link") or ""
        text = (it.get("content_text") or "").strip()
        sents = extract_keyword_sentences(text)
        if not sents:
            continue  # 키워드 문장 없으면 버림
        results.append({
            "장소이름": camp_name,
            "링크주소": link,
            "본문내용": "\n".join(sents)
        })
    return results
def join_keyword_sentences(sentences: List[str], mode: str = "dot") -> str:
    # 문장 끝 불필요한 구두점 제거 후 조인
    cleaned = [re.sub(r'\s+', ' ', s).strip().rstrip('.,;·•-') for s in sentences if s.strip()]
    if not cleaned:
        return ""
    if mode == "comma":
        return ", ".join(cleaned)
    # default: dot
    return ". ".join(cleaned) + "."

def format_snippets_as_text(
    snippets: List[Dict[str, str]],
    style: str = "block",   # "block" | "kv"
    body_sep: str = "dot",  # "dot" | "comma"  ← 추가
) -> str:
    if not snippets:
        return ""

    if style == "block":
        lines = []
        for s in snippets:
            name = s.get("장소이름", "")
            link = s.get("링크주소", "")
            body_raw = s.get("본문내용", "")

            # 본문이 문자열이면 줄바꿈 기준으로 문장 리스트화
            sents = [x.strip() for x in body_raw.splitlines() if x.strip()]
            body = join_keyword_sentences(sents, mode=body_sep)

            lines.append(
                f"[{name}]의 웹검색 결과입니다.\n"
                f"- 링크: {link}\n"
                f"- 키워드 문장: {body}\n"
            )
        return "\n".join(lines).strip()

    if style == "kv":
        kv = {}
        for s in snippets:
            name = s.get("장소이름","")
            if not name:
                continue
            link = s.get("링크주소","")
            body_raw = s.get("본문내용","")
            sents = [x.strip() for x in body_raw.splitlines() if x.strip()]
            body = join_keyword_sentences(sents, mode=body_sep)
            kv[name] = [name, link, body]
        import json
        return json.dumps(kv, ensure_ascii=False, indent=2)

    return format_snippets_as_text(snippets, style="block", body_sep=body_sep)

# === 데모 ===
async def demo_run():
    query = input("질문: ").strip()
    camping_type = input("캠핑유형(엔터=전체, 예:'유료캠핑장'/'글램핑/카라반'/'오지/노지캠핑'): ").strip() or None

    docs_with_metadata = get_top2_docs(vectordb, query=query, camping_type=camping_type)
    print("\n[Top-2 로컬문서]")
    for i, (doc, score) in enumerate(docs_with_metadata, 1):
        m = doc.metadata or {}
        print(f"- ({i}) score={score:.4f} | 이름={m.get('캠핑장이름') or m.get('name')} | 수정날짜={m.get('수정날짜')}")

    snippets = await build_snippet_per_doc(docs_with_metadata, per_type_display=20)

    if not snippets:
        print("\n(선택된 글 없음: 최신 글이 없거나 본문/키워드 매칭 실패)")
        return

    print("\n[최종 결과: 문서당 1개]")
    for row in snippets:
        print(row["장소이름"])
        print(row["링크주소"])
        print("본문내용:")
        print(row["본문내용"])
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(demo_run())
