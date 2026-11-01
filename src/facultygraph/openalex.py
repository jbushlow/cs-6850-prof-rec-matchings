"""Stage 2: match official faculty and collect their compact faculty-only works."""
from __future__ import annotations
import logging
from pathlib import Path
from collections import defaultdict
from .common import (normalize_orcid, normalized_name, short_id,
                     decode_inverted_index, read_csv, write_csv, write_jsonl)
from .faculty import FIELDS
from .progress import progress_bar, progress_iter
LOG = logging.getLogger(__name__)

AUTHOR_SELECT = ["id","display_name","display_name_alternatives","orcid","observed_orcids",
                 "affiliations","last_known_institutions", "works_count"]
# Do not use a publication-year filter here: it would truncate a professor's career.
WORK_SELECT = ["id","doi","title","publication_year","publication_date","type", "ids",
               "authorships", "is_authors_truncated", "abstract_inverted_index", "primary_location",
               "best_oa_location", "locations"]

COMPACT_WORK_FIELDS = ["id", "doi", "title", "publication_year", "publication_date",
                       "type", "ids"]


def _needs_full_work(work):
    """List responses contain at most 100 authorships; the flag is not universal."""
    authorships = work.get("authorships") or []
    return bool(work.get("is_authors_truncated")) or len(authorships) >= 100


def _compact_location(location):
    if not location or not location.get("landing_page_url"):
        return None
    return {"landing_page_url": location["landing_page_url"]}


def compact_work(work, faculty_ids, max_locations=4):
    """Keep only fields consumed by later stages and faculty-node authorships."""
    compact = {field: work.get(field) for field in COMPACT_WORK_FIELDS}
    compact["abstract"] = decode_inverted_index(work.get("abstract_inverted_index")) or None
    compact["authorships"] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        aid = short_id(author.get("id"), "A")
        if aid not in faculty_ids:
            continue
        compact["authorships"].append({
            "author_position": authorship.get("author_position") or "",
            "author": {"id": author.get("id"), "display_name": author.get("display_name") or ""},
            "institutions": [
                {"id": institution.get("id")}
                for institution in authorship.get("institutions") or []
                if institution.get("id")
            ],
            "raw_affiliation_strings": authorship.get("raw_affiliation_strings") or [],
        })
    compact["primary_location"] = _compact_location(work.get("primary_location"))
    compact["best_oa_location"] = _compact_location(work.get("best_oa_location"))
    locations = [_compact_location(item) for item in work.get("locations") or []]
    compact["locations"] = [item for item in locations if item][:max(0, int(max_locations))]
    return compact


def candidate_names(author):
    return {normalized_name(s) for s in [author.get("display_name", "")] +
            (author.get("display_name_alternatives") or []) if s}


def author_orcids(author):
    values = [author.get("orcid")] + (author.get("observed_orcids") or [])
    return {normalize_orcid(v) for v in values if normalize_orcid(v)}


def institution_in_author(author, iid):
    affs = author.get("affiliations") or []
    last = author.get("last_known_institutions") or []
    return (any(short_id(item.get("institution",{}).get("id"),"I") == iid for item in affs)
            or any(short_id(item.get("id"),"I") == iid for item in last))


def match_one(faculty, candidates):
    if faculty.get("match_status") == "directory_conflict":
        return "", "directory_conflict", []
    name = normalized_name(faculty["name"])
    matches = [a for a in candidates if name in candidate_names(a)]
    # ORCID shown on the directory is independent enough to disambiguate a name collision.
    official_orcid = normalize_orcid(faculty.get("orcid"))
    if official_orcid:
        verified = [a for a in matches if official_orcid in author_orcids(a)]
        if len(verified) == 1: return short_id(verified[0]["id"],"A"), "matched_orcid", [short_id(a["id"]) for a in matches]
        if len(verified) > 1: return "", "ambiguous_orcid", [short_id(a["id"]) for a in verified]
    if len(matches) == 1:
        # If the directory ORCID exists and the OpenAlex profile actively disagrees, skip.
        if official_orcid and author_orcids(matches[0]) and official_orcid not in author_orcids(matches[0]):
            return "", "orcid_conflict", [short_id(matches[0]["id"])]
        return short_id(matches[0]["id"],"A"), "matched_unique_name", [short_id(matches[0]["id"])]
    if len(matches) == 0: return "", "not_found", []
    return "", "ambiguous_name", [short_id(a["id"]) for a in matches]


def collect_openalex(cfg, oa, output_dir):
    path = Path(output_dir)
    faculty = read_csv(path/"faculty.csv")
    max_faculty = cfg.get("settings",{}).get("max_faculty")
    max_papers = cfg.get("settings",{}).get("max_papers_per_author")
    settings = cfg.get("settings", {})
    candidate_info = {}
    by_institution = defaultdict(list)
    for r in faculty: by_institution[r["institution_id"]].append(r)
    institution_cfg = {short_id(x["openalex_id"],"I"):x for x in cfg["institutions"]}
    for iid, roster in by_institution.items():
        # Include explicit child-campus / medical institution records when configured.
        aff_ids = [iid] + [short_id(v,"I") for v in institution_cfg.get(iid,{}).get("affiliation_ids",[])]
        # Affiliations is publication-derived history. Graduate students are discarded
        # by the official roster intersection (not by an unreliable heuristic).
        wanted = {normalized_name(p["name"]) for p in roster}
        indexed = defaultdict(list)
        count = 0
        seen = set()
        with progress_bar(settings=settings, desc=f"OpenAlex authors {iid}",
                          unit="profile", total=None) as bar:
            for query_iid in sorted(set(aff_ids)):
                for author in oa.pages("authors", f"affiliations.institution.id:{query_iid}", select=AUTHOR_SELECT):
                    count += 1
                    bar.update(1)
                    aid = short_id(author.get("id"),"A")
                    if aid in seen or not institution_in_author(author, query_iid): continue
                    seen.add(aid)
                    for n in candidate_names(author) & wanted: indexed[n].append(author)
                    bar.set_postfix(candidates=len(indexed), refresh=False)
        LOG.info("OpenAlex: institution %s scanned %s author profiles; %s roster names have candidates", iid, count, len(indexed))
        for person in roster:
            candidate_info[person["faculty_id"]] = match_one(person, indexed.get(normalized_name(person["name"]), []))
    # Prevent assigning the SAME OpenAlex author ID to distinct faculty records with
    # different normalized names (unless same actual faculty listed at multiple schools).
    for person in faculty:
        aid, status, candidates = candidate_info.get(person["faculty_id"], ("", "not_found", []))
        person["openalex_id"] = aid
        person["match_status"] = status
    conflicts = defaultdict(list)
    for person in faculty:
        if person["openalex_id"]: conflicts[person["openalex_id"]].append(person)
    for aid, rows in conflicts.items():
        if len({normalized_name(r["name"]) for r in rows}) > 1:
            for r in rows:
                r["openalex_id"] = ""; r["match_status"] = "conflicting_faculty_rosters"
    write_csv(path/"faculty.csv", faculty, FIELDS)
    diagnostics = []
    for person in faculty:
        diagnostics.append({**person, "candidate_ids": "|".join(candidate_info.get(person["faculty_id"], (None,None,[]))[2])})
    write_csv(path/"match_diagnostics.csv", diagnostics,
              ["faculty_id", "name", "match_status", "candidate_ids"])
    accepted = {r["openalex_id"] for r in faculty if r["openalex_id"]}
    if max_faculty is not None:
        accepted = set(sorted(accepted)[:int(max_faculty)])
        for person in faculty:
            if person["openalex_id"] and person["openalex_id"] not in accepted:
                person["openalex_id"] = ""
                person["match_status"] = "matched_excluded_by_limit"
        write_csv(path/"faculty.csv", faculty, FIELDS)
    LOG.info("Matched %s/%s faculty directory rows (%s distinct author IDs selected)",
             sum(bool(r["openalex_id"]) for r in faculty), len(faculty), len(accepted))

    # Batch author-ID OR filters when collecting complete histories. This avoids
    # repeatedly downloading papers coauthored by multiple target professors.
    work_map = {}
    max_locations = int(cfg.get("settings", {}).get("max_landing_pages", 4))
    batch_size = max(1, min(int(cfg.get("settings",{}).get("author_batch_size",20)), 100))
    if max_papers is not None: batch_size = 1  # per-author sampling semantics
    sorted_aids = sorted(accepted)
    batches = [sorted_aids[i:i+batch_size] for i in range(0,len(sorted_aids),batch_size)]
    for n, group in enumerate(progress_iter(batches, settings=settings, desc="OpenAlex work batches",
                                            unit="batch", total=len(batches)), 1):
        LOG.info("Fetching works: author batch %s/%s (%s authors)",n,len(batches),len(group))
        ids_filter = "|".join(group)
        for w in oa.pages("works", f"authorships.author.id:{ids_filter}", select=WORK_SELECT,
                          max_results=max_papers if batch_size==1 else None):
            wid = short_id(w.get("id"), "W")
            if not wid or wid in work_map: continue
            # OpenAlex list responses contain at most 100 authorships. Fetch the
            # single-work record before filtering so a target faculty author after
            # position 100 is not lost.
            if _needs_full_work(w):
                w = oa.get("works/"+wid)
            w = compact_work(w, accepted, max_locations=max_locations)
            if not w["authorships"]:
                LOG.warning("Skipping work %s: no selected faculty author remained after expansion", wid)
                continue
            work_map[wid] = w
    compressed_path = path/"raw"/"papers_openalex.jsonl.gz"
    legacy_path = path/"raw"/"papers_openalex.jsonl"
    write_jsonl(compressed_path, (work_map[k] for k in sorted(work_map)))
    # The compressed file is written atomically before its obsolete predecessor
    # is removed, so an interrupted run cannot erase the only raw-work output.
    legacy_path.unlink(missing_ok=True)
    LOG.info("Collected %s unique works from %s faculty authors", len(work_map),len(accepted))
    return faculty, work_map
