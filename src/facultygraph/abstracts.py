"""Stage 3: validate abstracts and retrieve only missing ones, with durable results cache.

Sources (in order): OpenAlex, Crossref DOI, Semantic Scholar DOI, Europe PMC,
public HTML landing pages. Never scrape gated content, login pages, or PDFs.
"""
from __future__ import annotations
import html as html_lib
import json
import logging
import os
import re
import ipaddress
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, urlparse
from bs4 import BeautifulSoup
from .common import (normalize_doi, short_id, append_jsonl, decode_inverted_index,
                     read_csv, read_jsonl, write_jsonl)
from .http import FetchError
from .progress import progress_iter
LOG = logging.getLogger(__name__)


def clean_abstract(value):
    if not value: return ""
    txt = BeautifulSoup(html_lib.unescape(str(value)), "html.parser").get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"^(abstract|summary)\s*[:—-]?\s*", "", txt, flags=re.I)
    return txt.strip()

BAD_TEXT = ("access denied", "enable javascript", "page not found", "please log in",
            "sign in to access", "we use cookies", "captcha", "request blocked")

def valid_abstract(value, min_words=25):
    text = clean_abstract(value)
    if len(text.split()) < min_words: return False
    if len(text)> 40000: return False
    if any(s in text[:250].casefold() for s in BAD_TEXT): return False
    return True


def url_is_public_http(url):
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        if p.scheme not in ("http","https") or not host or host=="localhost" or host.endswith((".local", ".internal")): return False
        try:
            ip = ipaddress.ip_address(host)
            return ip.is_global
        except ValueError: pass
        return "." in host
    except ValueError: return False


def title_similar(expected, observed):
    def tokens(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").casefold()).split()
    a, b = " ".join(tokens(expected)), " ".join(tokens(observed))
    return bool(a and b and SequenceMatcher(None,a,b).ratio()>=0.73)


def _json_ld_abstract(value):
    if isinstance(value,list):
        for x in value:
            s = _json_ld_abstract(x)
            if s: return s
    if isinstance(value,dict):
        # JSON-LD may be a list of types or @graph of items.
        typ = value.get("@type", [])
        types = [typ] if isinstance(typ,str) else typ
        if any(t.lower().endswith(("scholarlyarticle", "article")) for t in types):
            if value.get("abstract"): return value["abstract"]
        if value.get("@graph"): return _json_ld_abstract(value["@graph"])
    return ""


def extract_html_abstract(html, expected_title, expected_doi="", min_words=25):
    soup = BeautifulSoup(html, "html.parser")
    # Verify that the page actually describes the expected work.
    meta = {x.get("name",x.get("property", "")).lower():x.get("content","")
            for x in soup.select("meta[content]")}
    page_title = meta.get("citation_title") or meta.get("dc.title") or meta.get("og:title") or (soup.title.get_text(" ",strip=True) if soup.title else "")
    page_doi = normalize_doi(meta.get("citation_doi", "") or meta.get("dc.identifier", ""))
    if page_doi and expected_doi and page_doi != normalize_doi(expected_doi): return ""
    if not page_doi and not title_similar(expected_title, page_title): return ""
    if page_doi and expected_doi and page_doi == normalize_doi(expected_doi): pass
    elif not title_similar(expected_title, page_title): return ""
    # Strict tags first; do not accept generic og:description snippets.
    for k in ("citation_abstract", "dc.description", "dcterms.abstract", "dc.abstract", "description.abstract"):
        a = clean_abstract(meta.get(k))
        if valid_abstract(a,min_words): return a
    for script in soup.select('script[type="application/ld+json"]'):
        try: val = json.loads(script.string or script.get_text())
        except (ValueError,TypeError): continue
        a = clean_abstract(_json_ld_abstract(val))
        if valid_abstract(a,min_words): return a
    # A targeted visible abstract container, not arbitrary page text.
    for selector in ("section.abstract", "div.abstract", "#abstract", "section#abstract", 
                     "[itemprop='abstract']", "[data-test='abstract']", "[aria-label='Abstract']",
                     "blockquote.abstract", "div#abs", ".abstractSection", ".abstractText",
                     "section.article-section__abstract"):
        for el in soup.select(selector):
            a = clean_abstract(el.get_text(" ", strip=True))
            if valid_abstract(a,min_words): return a
    # Generic publisher pages sometimes have a heading and sibling paragraphs,
    # without an id/class on the containing abstract section. Stay conservative.
    for heading in soup.select("h2,h3,h4"):
        if heading.get_text(" ",strip=True).casefold().strip(": ") != "abstract":
            continue
        chunks = []
        for sibling in heading.next_siblings:
            if getattr(sibling, "name", None) in ("h2","h3","h4"):
                break
            if hasattr(sibling, "get_text"):
                chunks.append(sibling.get_text(" ",strip=True))
            if len(" ".join(chunks)) > 6000:
                break
        a = clean_abstract(" ".join(chunks))
        if valid_abstract(a,min_words): return a
    return ""


class AbstractCollector:
    def __init__(self, http, output_dir, settings):
        self.http = http
        self.out = Path(output_dir)
        self.settings = settings
        self.min_words = int(settings.get("abstract_min_words",25))
        self.use_fallbacks = bool(settings.get("abstract_fallbacks", True))
        self.max_pages = int(settings.get("max_landing_pages",4))
        self.fallback_attempts = max(1, int(settings.get("abstract_max_attempts",3)))
        self.fallback_timeout = max(1, float(settings.get("abstract_request_timeout",20)))
        self.results_path = self.out / "state" / "abstract_results.jsonl"
        self.old = {r["paper_id"]:r for r in read_jsonl(self.results_path)}
        self._progress = None

    def _show_attempt(self, work, provider):
        if self._progress is None:
            return
        wid = short_id(work.get("id"), "W")
        title = re.sub(r"\s+", " ", work.get("title") or "").strip()
        if len(title) > 48:
            title = title[:45] + "..."
        self._progress.set_postfix_str(f"{wid} | {provider} | {title}", refresh=True)

    def _get(self, url, params=None, headers=None, text=False, allow_not_found=False):
        return self.http.get(
            url, params=params, headers=headers, text=text,
            allow_not_found=allow_not_found,
            max_attempts=self.fallback_attempts, timeout=self.fallback_timeout,
        )

    def _crossref(self, work):
        doi = normalize_doi(work.get("doi"))
        if not doi: return None
        self._show_attempt(work, "Crossref")
        try:
            data = self._get("https://api.crossref.org/works/" + quote(doi,safe="/"), allow_not_found=True)
            return ((data or {}).get("message") or {}).get("abstract"), "crossref", f"https://api.crossref.org/works/{doi}"
        except FetchError as e:
            LOG.warning("Crossref %s: %s",doi,e); return None

    def _s2(self, work):
        doi = normalize_doi(work.get("doi"))
        if not doi: return None
        self._show_attempt(work, "Semantic Scholar")
        try:
            headers = {}
            if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
                headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
            url = "https://api.semanticscholar.org/graph/v1/paper/DOI:" + quote(doi, safe="/")
            data = self._get(url, {"fields":"title,abstract,year,externalIds"},headers=headers,allow_not_found=True)
            if not data: return None
            # DOI request already identifies the paper, but require title agreement.
            if not title_similar(work.get("title"),data.get("title")): return None
            return data.get("abstract"),"semantic_scholar",url
        except FetchError as e:
            LOG.warning("Semantic Scholar %s: %s",doi,e); return None

    def _europepmc(self, work):
        ids = work.get("ids") or {}
        doi = normalize_doi(work.get("doi"))
        pmid = str(ids.get("pmid") or "").rstrip("/").split("/")[-1]
        pmcid = str(ids.get("pmcid") or "").rstrip("/").split("/")[-1]
        query = f"EXT_ID:{pmid}" if pmid else (f"PMCID:{pmcid}" if pmcid else (f"DOI:{doi}" if doi else ""))
        if not query: return None
        self._show_attempt(work, "Europe PMC")
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        try:
            data = self._get(url, {"query":query, "format":"json", "resultType":"core", "pageSize":5})
            for record in (data.get("resultList") or {}).get("result", []):
                if doi and record.get("doi") and normalize_doi(record["doi"]) != doi: continue
                if not title_similar(work.get("title"),record.get("title")): continue
                return record.get("abstractText"),"europe_pmc",url
        except FetchError as e:
            LOG.warning("Europe PMC %s: %s",query,e)
        return None

    def _can_fetch(self, url):
        return url_is_public_http(url) and self.http.robots_allowed(
            url, max_attempts=self.fallback_attempts, timeout=self.fallback_timeout
        )

    def _landing_pages(self, work):
        locations = [work.get("best_oa_location"),work.get("primary_location")]
        locations += work.get("locations") or []
        urls = []
        for loc in locations:
            if not loc: continue
            url = loc.get("landing_page_url")
            if url and url not in urls and url_is_public_http(url): urls.append(url)
        if work.get("doi"):
            url = "https://doi.org/" + normalize_doi(work["doi"])
            if url not in urls: urls.append(url)
        return urls[:self.max_pages]

    def _html(self, work):
        for url in self._landing_pages(work):
            self._show_attempt(work, "landing page")
            if not self._can_fetch(url): continue
            try:
                page = self._get(url, text=True, allow_not_found=True)
                if not page: continue
                abstract = extract_html_abstract(page,work.get("title"),work.get("doi"),self.min_words)
                if abstract: return abstract, "landing_page",url
            except FetchError as exc:
                LOG.info("Landing page skipped %s: %s", url,exc)
        return None

    def collect_one(self, work, retry_failed=False):
        wid = short_id(work.get("id"),"W")
        cached = self.old.get(wid)
        # OpenAlex-only mode must not leak an abstract recovered by an earlier
        # fallback-enabled run. Cached OpenAlex and missing results remain safe.
        cache_matches_mode = (
            cached is not None
            and (self.use_fallbacks or cached.get("source") in (None, "openalex"))
        )
        if cache_matches_mode and (
            not self.use_fallbacks or not retry_failed or cached["status"] == "found"
        ):
            return cached
        # New compact records store decoded text; retain compatibility with
        # older uncompressed raw records containing an inverted index.
        abstract = work.get("abstract") or decode_inverted_index(work.get("abstract_inverted_index"))
        if valid_abstract(abstract,self.min_words):
            result = {"paper_id":wid,"status":"found","abstract":clean_abstract(abstract),
                      "source":"openalex","source_url":work.get("id")}
        else:
            result = {"paper_id":wid,"status":"missing","abstract":None,
                      "source":None,"source_url":None}
            if self.use_fallbacks:
                for method in (self._crossref,self._s2,self._europepmc,self._html):
                    hit = method(work)
                    if hit and valid_abstract(hit[0],self.min_words):
                        result.update({"status":"found", "abstract":clean_abstract(hit[0]),
                                       "source":hit[1],"source_url":hit[2]})
                        break
        append_jsonl(self.results_path,result)
        self.old[wid] = result
        return result

    def run(self, retry_failed=False):
        compressed = self.out/"raw"/"papers_openalex.jsonl.gz"
        legacy = self.out/"raw"/"papers_openalex.jsonl"
        src = compressed if compressed.exists() else legacy
        if not src.exists(): raise FileNotFoundError("Missing raw OpenAlex papers; run collect-openalex first")
        docs = []
        source_counts = {}
        works = list(read_jsonl(src))
        faculty = read_csv(self.out/"faculty.csv")
        alias_to_faculty = {}
        for row in faculty:
            aliases = (row.get("openalex_ids") or row.get("openalex_id") or "").split("|")
            for alias in aliases:
                aid = short_id(alias, "A")
                if aid: alias_to_faculty[aid] = row["faculty_id"]
        progress = progress_iter(works, settings=self.settings, desc="Abstract recovery",
                                 unit="paper", total=len(works))
        self._progress = progress
        for i, work in enumerate(progress, 1):
            self._show_attempt(work, "checking cache/OpenAlex")
            result = self.collect_one(work,retry_failed=retry_failed)
            self._show_attempt(work, f"done: {result['source'] or 'missing'}")
            source = result["source"] or "missing"
            source_counts[source] = source_counts.get(source,0)+1
            openalex_author_ids = [short_id(a.get("author",{}).get("id"),"A")
                                   for a in work.get("authorships") or [] if a.get("author",{}).get("id")]
            faculty_ids = list(dict.fromkeys(alias_to_faculty[aid] for aid in openalex_author_ids
                                             if aid in alias_to_faculty))
            docs.append({"openalex_id":short_id(work.get("id"),"W"),
                         "doi":normalize_doi(work.get("doi")) or None,
                         "title":work.get("title"),"year":work.get("publication_year"),
                         "date":work.get("publication_date"),"type":work.get("type"),
                         "author_ids":faculty_ids,
                         "openalex_author_ids":openalex_author_ids,
                         "abstract":result["abstract"],"abstract_status":result["status"],
                         "abstract_source":result["source"],"abstract_source_url":result["source_url"],
                         "publication_url":(work.get("primary_location") or {}).get("landing_page_url")})
            if i%200==0: LOG.info("Abstract processing %s papers; sources=%s",i,source_counts)
        write_jsonl(self.out/"papers.jsonl",docs)
        self._progress = None
        LOG.info("Abstract processing completed: %s",source_counts)
        return docs
