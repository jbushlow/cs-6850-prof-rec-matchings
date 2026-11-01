"""Bounded retries, host pacing and persistent HTTP GET cache. No scraping bypasses."""
from __future__ import annotations
import hashlib
import json
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import ipaddress
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import requests

class FetchError(RuntimeError):
    def __init__(self, status, url):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url

class Http:
    def __init__(self, cache_dir, email="", refresh=False, session=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.refresh = refresh
        self.session = session or requests.Session()
        self.last_request = {}
        self.robots = {}
        self.rates = {"api.openalex.org":0.2, "api.crossref.org":0.5,
                      "api.semanticscholar.org":1.2, "www.ebi.ac.uk":0.5}
        self.headers = {"User-Agent": f"FacultyNetworkResearch/0.1 (contact: {email or 'unspecified'}; academic metadata research)", "Accept": "application/json, text/html;q=0.9"}

    def get(self, url, params=None, headers=None, text=False, allow_not_found=False, cache=True,
            max_attempts=5, timeout=35):
        # Keys must never be embedded in the cache path, saved to outputs, or logged.
        params = {k:v for k,v in (params or {}).items() if v is not None}
        public_params = {k:v for k,v in params.items() if k not in {"api_key", "key", "token"}}
        fingerprint = json.dumps([url, sorted(public_params.items()), text], sort_keys=True)
        key = hashlib.sha256(fingerprint.encode()).hexdigest()
        path = self.cache_dir / key[:2] / (key + (".txt" if text else ".json"))
        if cache and not self.refresh and path.exists():
            return path.read_text(encoding="utf-8") if text else json.loads(path.read_text(encoding="utf-8"))
        host = urlparse(url).hostname or ""
        rate = self.rates.get(host, 1.5)
        h = dict(self.headers); h.update(headers or {})
        max_attempts = max(1, int(max_attempts))
        timeout = max(1, float(timeout))
        for attempt in range(max_attempts):
            delay = rate - (time.monotonic() - self.last_request.get(host, -100))
            if delay > 0: time.sleep(delay)
            self.last_request[host] = time.monotonic()
            try:
                r = self.session.get(url, params=params, headers=h, timeout=timeout, allow_redirects=True)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == max_attempts - 1: raise FetchError("network-failure", url) from exc
                time.sleep(min(2**attempt + random.uniform(0,0.3), 20)); continue
            if r.status_code == 404 and allow_not_found: return None
            if r.status_code in (429,500,502,503,504):
                if attempt == max_attempts - 1: raise FetchError(r.status_code, url)
                retry = r.headers.get("Retry-After", "")
                try: wait = float(retry)
                except ValueError:
                    try: wait = max(0, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError): wait = min(2**attempt, 30)
                time.sleep(min(max(wait, 1), 60)); continue
            if not r.ok: raise FetchError(r.status_code, url)
            result = r.text if text else r.json()
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result if text else json.dumps(result), encoding="utf-8")
            return result
        raise FetchError("unknown", url)

    def robots_allowed(self, url, max_attempts=5, timeout=35):
        """Check robots.txt for public HTTP HTML URLs; fail closed on fetch errors."""
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in ("http","https") or not hostname or hostname == "localhost" or hostname.endswith((".local",".internal")):
            return False
        try:
            if not ipaddress.ip_address(hostname).is_global: return False
        except ValueError:
            if "." not in hostname: return False
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self.robots:
            try:
                rules = self.get(base+"/robots.txt", text=True, allow_not_found=True,
                                 max_attempts=max_attempts, timeout=timeout)
                parser = RobotFileParser()
                parser.parse((rules or "").splitlines())
                self.robots[base] = parser
            except FetchError:
                self.robots[base] = False
        parser = self.robots[base]
        return parser is not False and parser.can_fetch("FacultyNetworkResearch",url)

class OpenAlex:
    BASE = "https://api.openalex.org"
    def __init__(self, http):
        self.http = http
        self.key = os.getenv("OPENALEX_API_KEY", "").strip()

    def get(self, endpoint, params=None, allow_not_found=False):
        p = dict(params or {})
        if self.key: p["api_key"] = self.key
        return self.http.get(self.BASE + "/" + endpoint.lstrip("/"), params=p, allow_not_found=allow_not_found)

    def pages(self, endpoint, filters, select=None, max_results=None):
        params = {"filter": filters, "per_page": 100, "cursor": "*"}
        if endpoint.strip("/") == "works": params["corpus"] = "core"
        if select: params["select"] = ",".join(select)
        count = 0
        while True:
            data = self.get(endpoint, params)
            rows = data.get("results", [])
            for row in rows:
                yield row
                count += 1
                if max_results is not None and count >= max_results: return
            nxt = (data.get("meta") or {}).get("next_cursor")
            if not rows or not nxt: return
            if not self.key:
                raise RuntimeError("OpenAlex cursor pagination beyond page one requires OPENALEX_API_KEY")
            params["cursor"] = nxt
