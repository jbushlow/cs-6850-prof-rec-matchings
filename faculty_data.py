"""Input adapter for faculty CSVs and OpenAlex paper JSONL / JSONL.GZ files.

Authorship rows are professor-paper bipartite relationships (scientific evidence).
Faculty-edge rows are *observed* professor-professor coauthorship (network).
The two independent sources are cross-checked instead of conflated.
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


MISSING_ABSTRACT_CHOICES = ("title", "skip", "error")


def _clean(value):
    return value.strip() if isinstance(value, str) else ""


def _csv_rows(path, fields):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        yield from reader


def _paper_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield row


def apply_missing_abstract_policy(data, policy="title"):
    """Apply the same abstract policy to existing professor JSON datasets.

    Keep coauthorship edges untouched even when papers are excluded from scoring.
    """
    if policy not in MISSING_ABSTRACT_CHOICES:
        raise ValueError(f"Unknown missing-abstract policy: {policy}")
    output = deepcopy(data)
    for prof in output["professors"]:
        filtered = []
        for paper in prof["papers"]:
            title, abstract = _clean(paper.get("title")), _clean(paper.get("abstract"))
            if not abstract:
                if policy == "error":
                    raise ValueError(f"Missing abstract for {paper['id']}")
                if policy == "skip" or not title:
                    continue
            paper["title"], paper["abstract"] = title, abstract
            paper["text_source"] = "abstract" if abstract else "title"
            filtered.append(paper)
        prof["papers"] = filtered
    return output


def load_faculty_files(nodes_path, authorships_path, papers_path, faculty_edges_path, *,
                       missing_abstract="title"):
    """Build the original recommend() dataset from the four real input files.

    IDs are joined exactly, without inferring affiliations/expertise from names.
    All network edges survive the text filter. Bad provenance fails visibly.
    """
    if missing_abstract not in MISSING_ABSTRACT_CHOICES:
        raise ValueError(f"MISSING_ABSTRACT must be one of {MISSING_ABSTRACT_CHOICES}")

    # The current nodes.csv is one row per faculty member. Its faculty_id is
    # the key used by authorships.author_id and faculty_edges.author_a/author_b.
    # A professor may have multiple OpenAlex IDs; these are *aliases*, not
    # separate professors. Keep unmatched roster members as faculty nodes.
    professors = {}
    openalex_to_faculty = {}
    node_total = 0
    for row in _csv_rows(nodes_path, ["faculty_id", "name", "openalex_ids"]):
        node_total += 1
        pid, name = _clean(row["faculty_id"]), _clean(row["name"])
        if not pid or not name or pid in professors:
            raise ValueError(f"Invalid/duplicate faculty node: {pid!r}")
        aliases = sorted(set(x.strip() for x in _clean(row["openalex_ids"]).split("|") if x.strip()))
        for alias in aliases:
            previous = openalex_to_faculty.setdefault(alias, pid)
            if previous != pid:
                raise ValueError(f"OpenAlex author ID {alias} maps to multiple faculty: {previous}, {pid}")
        professors[pid] = {
            "id": pid, "name": name, "papers": [],
            "faculty_institution_ids": _clean(row.get("institution_ids")),
            "openalex_ids": aliases,
            "departments": _clean(row.get("departments")),
            "match_status": _clean(row.get("match_status")),
        }
    if not professors:
        raise ValueError("No faculty nodes found in nodes.csv")

    author_by_paper = defaultdict(set)
    authorship_metadata = {}
    authorship_row_count = 0
    duplicate_authorship_rows = 0
    for row in _csv_rows(authorships_path, ["paper_id", "author_id"]):
        authorship_row_count += 1
        paper_id, author_id = _clean(row["paper_id"]), _clean(row["author_id"])
        if not paper_id or not author_id:
            raise ValueError(f"Invalid authorship row {authorship_row_count}")
        if author_id not in professors:
            continue  # only eligible faculty participate in this recommendation pool
        key = (paper_id, author_id)
        metadata = {"byline_index": _clean(row.get("byline_index")),
                    "author_position": _clean(row.get("author_position")),
                    "institution_ids": _clean(row.get("institution_ids")),
                    "raw_affiliation": _clean(row.get("raw_affiliation"))}
        if key in authorship_metadata:
            # The supplied export contains repeated faculty-paper pairs with
            # different byline positions. Treat as one authorship for scoring,
            # but keep every original byline in the graph provenance.
            previous = authorship_metadata[key]
            previous.setdefault("all_byline_entries", [
                {k: v for k, v in previous.items() if k != "all_byline_entries"}]).append(metadata)
            duplicate_authorship_rows += 1
            continue
        author_by_paper[paper_id].add(author_id)
        authorship_metadata[key] = metadata

    source_papers = {}
    paper_total = 0
    abstract_count = 0
    title_only_count = 0
    missing_both_count = 0
    skip_missing_abstract_count = 0
    for row in _paper_rows(papers_path):
        paper_total += 1
        paper_id = _clean(row.get("openalex_id"))
        if not paper_id or paper_id in source_papers:
            raise ValueError(f"Missing/duplicate openalex_id at paper row {paper_total}: {paper_id}")
        title, abstract = _clean(row.get("title")), _clean(row.get("abstract"))
        if abstract:
            abstract_count += 1
        elif title:
            title_only_count += 1
        else:
            missing_both_count += 1
        source_papers[paper_id] = row
        if paper_id not in author_by_paper:
            continue
        if not abstract:
            if missing_abstract == "error":
                raise ValueError(f"Missing abstract for paper {paper_id}; set MISSING_ABSTRACT to 'title' or 'skip'")
            if missing_abstract == "skip":
                skip_missing_abstract_count += 1
                continue
            if not title:
                # No text to encode: do not hallucinate content.
                continue
        declared_authors = row.get("author_ids")
        if isinstance(declared_authors, list):
            # Paper metadata can list OpenAlex author IDs (A...) while the
            # authorships file uses internal faculty IDs. Reconcile both.
            declared_faculty = set()
            for author_id in declared_authors:
                author_id = _clean(author_id)
                if author_id in professors:
                    declared_faculty.add(author_id)
                elif author_id in openalex_to_faculty:
                    declared_faculty.add(openalex_to_faculty[author_id])
            if declared_faculty != author_by_paper[paper_id]:
                raise ValueError(
                    f"Paper {paper_id}: author_ids and authorships.csv disagree "
                    f"(paper maps to {sorted(declared_faculty)}, "
                    f"authorships list {sorted(author_by_paper[paper_id])})"
                )
        paper = {"id": paper_id, "title": title, "abstract": abstract,
                 "text_source": "abstract" if abstract else "title",
                 "year": row.get("year"), "doi": row.get("doi"),
                 "publication_url": row.get("publication_url"),
                 "abstract_status": row.get("abstract_status"),
                 "abstract_source": row.get("abstract_source")}
        for author in sorted(author_by_paper[paper_id]):
            professors[author]["papers"].append(
                {**paper, "authorship_metadata": authorship_metadata[(paper_id, author)]})

    missing_papers = set(author_by_paper) - set(source_papers)
    if missing_papers:
        raise ValueError(f"authorships.csv references {len(missing_papers)} papers absent from papers file; "
                         f"example: {min(missing_papers)}")

    # Derive an independent projection solely for consistency auditing;
    # the actual network passed to scoring comes from faculty_edges.csv.
    projected = defaultdict(set)
    for paper_id, author_ids in author_by_paper.items():
        for a, b in itertools.combinations(sorted(author_ids), 2):
            projected[(a, b)].add(paper_id)

    edges = []
    seen = set()
    for row in _csv_rows(faculty_edges_path, ["author_a", "author_b", "weight", "last_year", "paper_ids"]):
        a, b = _clean(row["author_a"]), _clean(row["author_b"])
        if a not in professors or b not in professors or a == b:
            raise ValueError(f"Invalid faculty edge: {a} - {b}")
        pair = tuple(sorted((a, b)))
        if pair in seen:
            raise ValueError(f"Duplicate faculty edge: {pair}")
        seen.add(pair)
        paper_ids = json.loads(row["paper_ids"])
        if not isinstance(paper_ids, list) or len(set(paper_ids)) != len(paper_ids):
            raise ValueError(f"Invalid paper_ids on faculty edge {pair}")
        weight = int(row["weight"])
        if weight != len(paper_ids) or set(paper_ids) != projected[pair]:
            raise ValueError(f"faculty_edges.csv disagrees with authorships.csv for {pair}")
        year = _clean(row["last_year"])
        edges.append({"source": a, "target": b, "weight": weight,
                      "last_year": int(year) if year else None, "paper_ids": paper_ids})
    if seen != set(projected):
        raise ValueError(f"Faculty edge list disagrees with authorships projection: "
                         f"{len(set(projected)-seen)} missing, {len(seen-set(projected))} extra edges")

    used_ids = {p["id"] for person in professors.values() for p in person["papers"]}
    summary = {
        "inputs": {"nodes": Path(nodes_path).name, "authorships": Path(authorships_path).name,
                   "papers": Path(papers_path).name, "faculty_edges": Path(faculty_edges_path).name},
        "missing_abstract_policy": missing_abstract,
        "total_nodes": node_total, "eligible_faculty": len(professors),
        "authorship_rows": authorship_row_count,
        "duplicate_faculty_paper_authorship_rows_collapsed": duplicate_authorship_rows,
        "faculty_coauthorship_edges": len(edges),
        "total_paper_records": paper_total, "papers_with_abstract": abstract_count,
        "papers_without_abstract_with_title": title_only_count,
        "papers_without_title_or_abstract": missing_both_count,
        "papers_skipped_by_policy": skip_missing_abstract_count,
        "papers_excluded_for_no_usable_text": missing_both_count if missing_abstract == "title" else 0,
        "papers_used_for_semantic_scoring": len(used_ids),
        "papers_used_with_title_only": sum(not _clean(source_papers[pid].get("abstract")) for pid in used_ids),
    }
    return {"version": 1, "professors": list(professors.values()),
            "coauthorship_edges": edges, "source_summary": summary}
