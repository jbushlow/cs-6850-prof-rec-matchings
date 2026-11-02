"""Shared identifiers, JSONL/CSV helpers and safe text normalization."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse


def short_id(value, entity=None):
    if not value: return ""
    s = str(value).strip().rstrip("/")
    s = s.split("/")[-1] if s.startswith(("https://openalex.org/", "http://openalex.org/", "https://api.openalex.org/")) else s
    if entity and not re.fullmatch(entity + r"\d+", s): return ""
    return s


def normalized_name(value):
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.casefold()
    value = re.sub(r"\b(dr|prof|professor|phd|md|jr|sr)\b", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    return " ".join(value.split())


def identity_name_key(value):
    """Return a conservative key for deciding whether roster names are aliases.

    Keep the first complete given name and surname, ignoring middle names and
    initials. A leading initial therefore does not hide the useful given name
    in ``Y. Samuel Wang`` or ``B. Franklin Pugh``. The deliberately small
    nickname map covers variants observed in the Cornell directories without
    turning a shared first initial into evidence that two people are the same.
    """
    raw = (value or "").strip()
    if "," in raw:
        last, given = raw.split(",", 1)
        raw = f"{given.strip()} {last.strip()}"
    tokens = normalized_name(raw).split()
    if len(tokens) < 2:
        return tuple(tokens)
    surname = tokens[-1]
    given = next((token for token in tokens[:-1] if len(token) > 1), tokens[0])
    given = {"frank": "franklin"}.get(given, given)
    return given, surname


def normalize_orcid(value):
    value = (value or "").strip().rstrip("/").split("/")[-1].upper()
    return value if re.fullmatch(r"\d{4}-\d{4}-\d{4}-[\dX]{4}", value) else ""


def orcid_from_href(value):
    parsed = urlparse(value or "")
    return normalize_orcid(parsed.path) if parsed.hostname in ("orcid.org", "www.orcid.org") else ""


def safe_filename(value):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", value)[:100]


def stable_id(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def read_jsonl(path):
    path = Path(path)
    if not path.exists(): return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def decode_inverted_index(index):
    """Reconstruct plain text from an OpenAlex abstract inverted index."""
    if not index: return ""
    slots = {}
    for token, positions in index.items():
        for position in positions:
            if isinstance(position, int) and position >= 0:
                slots[position] = token
    if not slots or max(slots) > 20000: return ""
    return " ".join(slots[position] for position in sorted(slots))


def write_csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows: writer.writerow(row)
    tmp.replace(path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def normalize_doi(value):
    value = unquote(str(value or "")).strip()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
    return value.lower().strip()
