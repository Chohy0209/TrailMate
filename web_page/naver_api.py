# naver_api.py
import os
import re
import asyncio
import httpx
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.schema import Document

# --- 환경설정 ---
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise RuntimeError(".env에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 설정 필요")

NAVER_CONCURRENCY = 3
_rate_sem = asyncio.Semaphore(NAVER_CONCURRENCY)

# --- 유틸 함수 ---
def clean_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text or "").strip()

def _normalize(s: str) -> str:
    s = re.sub(r"[\(\)\[\]\{\}]", " ", str(s or "").strip())
    return re.sub(r"\s+", " ", s)

def _parse_any_date(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    digits = re.sub(r"\D", "", str(s).strip())
    if len(digits) >= 8:
        try:
            return datetime(int(digits[0:4]), int(digits[4:6]), int(digits[6:8]), tzinfo=timezone.utc)
        except ValueError: pass
    return None

def _item_datetime(item: Dict[str, Any]) -> Optional[datetime]:
    for k in ("pubDate", "postdate"):
        if dt := _parse_any_date(item.get(k)): return dt
    return None

def _latest_modified_from_meta(meta: Dict[str, Any]) -> Optional[datetime]:
    for k in ("수정날짜", "수정일"):
        if dt := _parse_any_date(meta.get(k)): return dt
    return None

# --- 네이버 API ---
async def naver_search_api(query: str, search_type: str, display: int = 20, sort: str = "sim") -> List[Dict[str, Any]]:
    url = f"https://openapi.naver.com/v1/search/{search_type}.json"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    params = {"query": query, "display": display, "start": 1, "sort": sort}

    async with _rate_sem, httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params, timeout=7)
        resp.raise_for_status()

    return [{
        "source": f"네이버-{search_type}", "title": clean_html_tags(it.get("title", "")),
        "description": clean_html_tags(it.get("description", "")), "link": it.get("link", ""),
        "pubDate": it.get("pubDate"), "postdate": it.get("postdate")
    } for it in resp.json().get("items", [])]

# --- 본문 추출 ---
def to_mobile_if_naver(url: str) -> str:
    if "blog.naver.com" in url and "m.blog.naver.com" not in url:
        return url.replace("://blog.naver.com", "://m.blog.naver.com")
    if "cafe.naver.com" in url and "m.cafe.naver.com" not in url:
        return url.replace("://cafe.naver.com", "://m.cafe.naver.com")
    return url

def extract_main_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["div.se-main-container", "#postViewArea", "article", "main", "#tbody"]:
        if node := soup.select_one(sel):
            return re.sub(r"\s*\n\s*\n\s*", "\n\n", node.get_text("\n").strip())
    return ""

async def fetch_article_text(url: str, timeout: int = 8, max_chars: int = 4000) -> Optional[str]:
    mob_url = to_mobile_if_naver(url)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(mob_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
            resp.raise_for_status()
        text = extract_main_text_from_html(resp.text)
        return text[:max_chars].strip() if text else None
    except Exception:
        return None

# --- 핵심 로직: Snippet 생성 ---
async def fetch_single_best_with_content_per_doc(docs_with_metadata, **kwargs) -> List[Dict[str, Any]]:
    results, seen_links = [], set()
    for doc, _ in docs_with_metadata:
        meta = doc.metadata or {}
        camp_name = _normalize(meta.get("캠핑장이름") or "")
        if not camp_name: continue

        local_dt = _latest_modified_from_meta(meta)
        tasks = [naver_search_api(camp_name, t, display=kwargs.get('per_type_display', 20)) for t in ["blog", "cafearticle", "webkr"]]
        packs = await asyncio.gather(*tasks, return_exceptions=True)
        
        merged = [it for p in packs if not isinstance(p, Exception) for it in p]
        merged = [it for it in merged if _item_datetime(it) is not None and camp_name in _normalize(it.get("title", ""))]
        merged.sort(key=_item_datetime, reverse=True)
        
        candidates = [it for it in merged if local_dt is None or ((web_dt := _item_datetime(it)) and web_dt > local_dt)]

        for it in candidates:
            if not (link := it.get("link")) or link in seen_links: continue
            if content := await fetch_article_text(link, timeout=kwargs.get('fetch_timeout', 8), max_chars=kwargs.get('max_chars', 3000)):
                it.update({"content_text": content, "matched_camp_name": camp_name})
                results.append(it)
                seen_links.add(link)
                break
    results.sort(key=_item_datetime, reverse=True)
    return results

def extract_keyword_sentences(text: str) -> List[str]:
    if not text: return []
    keywords = r"가격|요금|비용|인원|추가인원|체크인|체크아웃|입실|퇴실|운영시간|영업시간|예약|문의|전화"
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    return [s for s in sents if re.search(keywords, s) and (re.search(r"\d", s) or re.search(r"(시|분|원)", s))]

async def build_snippet_per_doc(docs_with_metadata, **kwargs) -> List[Dict[str, str]]:
    picked_items = await fetch_single_best_with_content_per_doc(docs_with_metadata, **kwargs)
    results = []
    for it in picked_items:
        text = it.get("content_text", "")
        if sents := extract_keyword_sentences(text):
            results.append({
                "장소이름": it.get("matched_camp_name", ""),
                "링크주소": it.get("link", ""),
                "snippet": ". ".join(sents) + "."
            })
    return results