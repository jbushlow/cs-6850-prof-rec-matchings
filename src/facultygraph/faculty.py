"""Stage 1: collect public faculty rosters from configurable directory sources."""
from __future__ import annotations
import logging
import re
from urllib.parse import unquote, urljoin, urlparse
from pathlib import Path
from bs4 import BeautifulSoup
from .common import normalize_orcid, orcid_from_href, normalized_name, stable_id, read_csv, write_csv
from .progress import progress_bar

LOG = logging.getLogger(__name__)
FIELDS = ["faculty_id", "name", "roster_names", "institution_name", "institution_id",
          "department", "profile_url", "profile_urls", "orcid", "faculty_source",
          "openalex_id", "openalex_ids", "roster_record_count", "match_status"]


def canonical_profile_url(url):
    """Normalize harmless directory URL aliases for identity comparison only.

    Keep meaningful slug and query differences intact. In particular, this
    collapses legacy ``/index.php/people/...`` links and their percent-encoded
    equivalents without changing the original URL written to faculty.csv.
    """
    if not url:
        return ""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    path = re.sub(r"/index\.php/", "/", path, flags=re.IGNORECASE)
    path = re.sub(r"/{2,}", "/", path)
    path = path.rstrip("/") or "/"
    return parsed._replace(
        scheme=parsed.scheme.casefold(),
        netloc=parsed.netloc.casefold(),
        path=path,
        fragment="",
    ).geturl()


def _faculty_record(name, dept, url, orcid, institution, source):
    name = " ".join((name or "").split())
    if not name: return None
    iid = institution["openalex_id"].split("/")[-1]
    return {"faculty_id": stable_id(iid, normalized_name(name), url or ""),
            "name":name, "institution_name":institution["name"], "institution_id":iid,
            "department":(dept or "").strip(), "profile_url":url or "",
            "orcid":normalize_orcid(orcid), "faculty_source":source,
            "openalex_id":"", "match_status":"unmatched"}


def parse_directory_html(html, url, institution, source):
    soup = BeautifulSoup(html, "html.parser")
    selector = source["card_selector"]
    cards = soup.select(selector)
    if not cards:
        raise ValueError(f"No '{selector}' cards at {url}. Check CSS/JS-rendered directory.")
    page_dept_el = soup.select_one(source["department_page_selector"]) if source.get("department_page_selector") else None
    page_dept = page_dept_el.get_text(" ", strip=True) if page_dept_el else ""
    results = []
    for card in cards:
        faculty_only = source.get("faculty_selector")
        if faculty_only and not card.select_one(faculty_only): continue
        el = card.select_one(source["name_selector"])
        if not el: continue
        name = el.get_text(" ", strip=True)
        if source.get("name_format") == "last_comma_first" and "," in name:
            last, first = name.split(",", 1)
            name = f"{first.strip()} {last.strip()}"
        link = card.select_one(source.get("profile_link_selector", "a[href]"))
        href = link.get("href", "").strip() if link else ""
        profile = urljoin(url, href) if href else ""
        dept_el = card.select_one(source["department_selector"]) if source.get("department_selector") else None
        dept = dept_el.get_text(" ", strip=True) if dept_el else page_dept or source.get("default_department", "")
        orcid_link = next((a for a in card.select('a[href]') if orcid_from_href(a.get("href"))), None)
        orcid = orcid_from_href(orcid_link.get("href")) if orcid_link else ""
        r = _faculty_record(name, dept, profile, orcid, institution, url)
        if r: results.append(r)
    return results, soup


def load_html_source(inst, source, http, settings=None):
    url = source["url"]
    seen = set()
    output = []
    max_pages = int(source.get("max_pages", 20))
    with progress_bar(settings=settings, desc=source.get("label", "HTML directory"),
                      unit="page", total=None) as bar:
        for _ in range(max_pages):
            if url in seen: break
            seen.add(url)
            if not http.robots_allowed(url):
                raise PermissionError(f"Directory blocked by robots.txt or not publicly accessible: {url}")
            html = http.get(url, text=True)
            records, soup = parse_directory_html(html, url, inst, source)
            output.extend(records)
            bar.update(1)
            bar.set_postfix(records=len(output), refresh=False)
            nxtsel = source.get("next_page_selector")
            link = soup.select_one(nxtsel) if nxtsel else None
            if not link or not link.get("href"): break
            url = urljoin(url, link["href"])
        else:
            LOG.warning("Directory hit max_pages=%s for %s; verify pagination completeness", max_pages, source["url"])
    return output


def _page_matches_source(soup, source):
    """Apply an optional, configuration-driven filter to a discovered detail page."""
    selector = source.get("required_text_selector")
    if not selector:
        return True
    element = soup.select_one(selector)
    expected = source.get("required_text", "")
    return bool(element and expected.casefold() in element.get_text(" ", strip=True).casefold())


def load_indexed_html_source(inst, source, http, settings=None):
    """Discover homogeneous detail pages from an index, then parse each as a directory.

    This is intentionally institution-agnostic: the index/detail selectors, optional
    page-type filter, and faculty selectors all live in configuration.
    """
    index_url = source["index_url"]
    if not http.robots_allowed(index_url):
        raise PermissionError(f"Directory blocked by robots.txt or not publicly accessible: {index_url}")
    index_html = http.get(index_url, text=True)
    index_soup = BeautifulSoup(index_html, "html.parser")
    links = index_soup.select(source["detail_link_selector"])
    if not links:
        raise ValueError(
            f"No '{source['detail_link_selector']}' detail links at {index_url}. Check CSS selector."
        )

    detail_urls = []
    seen = set()
    for link in links:
        href = link.get("href")
        if not href:
            continue
        detail_url = urljoin(index_url, href)
        if detail_url not in seen:
            seen.add(detail_url)
            detail_urls.append(detail_url)

    max_pages = int(source.get("max_detail_pages", len(detail_urls)))
    if len(detail_urls) > max_pages:
        LOG.warning(
            "Indexed directory discovered %s pages at %s but max_detail_pages=%s",
            len(detail_urls), index_url, max_pages,
        )

    output = []
    matched_pages = 0
    selected_urls = detail_urls[:max_pages]
    with progress_bar(settings=settings, desc=source.get("label", "Indexed directory"),
                      unit="page", total=len(selected_urls)) as bar:
        for detail_url in selected_urls:
            if not http.robots_allowed(detail_url):
                LOG.warning("Skipping detail page blocked by robots.txt: %s", detail_url)
                bar.update(1)
                continue
            html = http.get(detail_url, text=True)
            soup = BeautifulSoup(html, "html.parser")
            if _page_matches_source(soup, source):
                matched_pages += 1
                try:
                    records, _ = parse_directory_html(html, detail_url, inst, source)
                except ValueError:
                    if source.get("allow_empty_detail_pages", False):
                        LOG.warning("Skipping directory page with no faculty cards: %s", detail_url)
                        records = []
                    else:
                        raise
                output.extend(records)
            bar.update(1)
            bar.set_postfix(memberships=len(output), matched=matched_pages, refresh=False)

    if not matched_pages:
        raise ValueError(f"No discovered pages at {index_url} matched the configured page filter")
    LOG.info(
        "Indexed directory: %s matching pages yielded %s roster memberships",
        matched_pages, len(output),
    )
    return output


def load_rendered_html_source(inst, source, http, settings=None):
    """Parse a client-rendered directory using headless Chromium via Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "rendered_html sources require Playwright. Install project dependencies and run "
            "'playwright install chromium'."
        ) from exc

    url = source["url"]
    seen = set()
    output = []
    max_pages = int(source.get("max_pages", 20))
    wait_selector = source.get("wait_selector", source["card_selector"])
    timeout_ms = int(source.get("timeout_ms", 30000))
    delay_ms = int(source.get("page_delay_ms", 500))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            with progress_bar(settings=settings, desc=source.get("label", "Rendered directory"),
                              unit="page", total=None) as bar:
                for _ in range(max_pages):
                    if url in seen:
                        break
                    seen.add(url)
                    if not http.robots_allowed(url):
                        raise PermissionError(
                            f"Directory blocked by robots.txt or not publicly accessible: {url}"
                        )
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    page.wait_for_selector(wait_selector, state="visible", timeout=timeout_ms)
                    if delay_ms:
                        page.wait_for_timeout(delay_ms)
                    html = page.content()
                    records, soup = parse_directory_html(html, url, inst, source)
                    output.extend(records)
                    bar.update(1)
                    bar.set_postfix(records=len(output), refresh=False)
                    nxtsel = source.get("next_page_selector")
                    link = soup.select_one(nxtsel) if nxtsel else None
                    if not link or not link.get("href"):
                        break
                    url = urljoin(url, link["href"])
                else:
                    LOG.warning(
                        "Rendered directory hit max_pages=%s for %s; verify pagination completeness",
                        max_pages, source["url"],
                    )
        finally:
            browser.close()
    return output


def load_csv_source(inst, source, config_dir):
    path = Path(source["path"])
    if not path.is_absolute(): path = config_dir / path
    rows = read_csv(path)
    result = []
    for item in rows:
        r = _faculty_record(
            item.get(source.get("name_column", "name"), ""),
            item.get(source.get("department_column", "department"), source.get("default_department", "")),
            item.get(source.get("profile_url_column", "profile_url"), ""),
            item.get(source.get("orcid_column", "orcid"), ""),
            inst, f"csv:{path}")
        if r: result.append(r)
    return result


def collect_faculty(cfg, http, config_dir, output_dir):
    all_rows = []
    settings = cfg.get("settings", {})
    cross_host_aliases = {
        inst["openalex_id"].split("/")[-1]: bool(
            inst.get("allow_cross_host_profile_aliases", False)
        )
        for inst in cfg["institutions"]
    }
    for inst in cfg["institutions"]:
        if not inst.get("faculty_sources"):
            raise ValueError(f"No faculty_sources supplied for {inst['name']}")
        sources = inst["faculty_sources"]
        for source_no, source in enumerate(sources, 1):
            typ = source["type"]
            label = source.get("label", typ)
            LOG.info("Faculty source %s/%s for %s: %s", source_no, len(sources), inst["name"], label)
            before = len(all_rows)
            if typ == "csv": all_rows.extend(load_csv_source(inst, source, Path(config_dir)))
            elif typ == "html": all_rows.extend(load_html_source(inst, source, http, settings))
            elif typ == "indexed_html": all_rows.extend(load_indexed_html_source(inst, source, http, settings))
            elif typ == "rendered_html": all_rows.extend(load_rendered_html_source(inst, source, http, settings))
            else: raise ValueError(f"Unsupported directory source type {typ!r}")
            LOG.info("Completed %s: %s roster memberships", label, len(all_rows) - before)
    # Two departments may list the same faculty member; retain one row per
    # institution/name and merge department memberships.
    dedup = {}
    for r in all_rows:
        key = (r["institution_id"], normalized_name(r["name"]))
        if key not in dedup:
            dedup[key] = r
        else:
            existing = dedup[key]
            if (existing["profile_url"] and r["profile_url"]
                    and canonical_profile_url(existing["profile_url"])
                    != canonical_profile_url(r["profile_url"])):
                existing_host = (urlparse(existing["profile_url"]).hostname or "").casefold()
                incoming_host = (urlparse(r["profile_url"]).hostname or "").casefold()
                aliases_allowed = cross_host_aliases.get(existing["institution_id"], False)
                # Some institutions publish one profile per school. Treat those as
                # aliases only across hosts; two distinct profiles on one directory
                # remain a possible same-name collision requiring review.
                if not (aliases_allowed and existing_host and incoming_host
                        and existing_host != incoming_host):
                    existing["match_status"] = "directory_conflict"
            if r["department"] and r["department"] not in existing["department"]:
                existing["department"] = "; ".join(x for x in (existing["department"],r["department"]) if x)
            if not existing["orcid"]: existing["orcid"] = r["orcid"]
            if not existing["profile_url"]: existing["profile_url"] = r["profile_url"]
            # Conflicting ORCIDs are not resolved silently.
            if existing["orcid"] and r["orcid"] and existing["orcid"] != r["orcid"]:
                existing["match_status"] = "directory_conflict"
    out = sorted(dedup.values(), key=lambda r: (r["institution_name"],r["name"]))
    write_csv(Path(output_dir)/"faculty.csv", out, FIELDS)
    LOG.info("Stage 1: %s faculty rows across %s institutions", len(out), len(cfg['institutions']))
    return out
