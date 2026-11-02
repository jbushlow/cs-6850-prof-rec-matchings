"""Stage 2: match official faculty and collect their compact faculty-only works."""
from __future__ import annotations
import logging
from pathlib import Path
from collections import defaultdict
from .common import (normalized_name, short_id,
                     decode_inverted_index, read_csv, stable_id, write_csv, write_jsonl)
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


def institution_in_author(author, iid):
    affs = author.get("affiliations") or []
    last = author.get("last_known_institutions") or []
    return (any(short_id(item.get("institution",{}).get("id"),"I") == iid for item in affs)
            or any(short_id(item.get("id"),"I") == iid for item in last))


def match_one(faculty, candidates):
    name = normalized_name(faculty["name"])
    matches = [a for a in candidates if name in candidate_names(a)]
    ids = [short_id(a["id"], "A") for a in matches]
    if len(matches) == 0: return "", "not_found", []
    primary = max(matches, key=lambda a: (int(a.get("works_count") or 0), short_id(a.get("id"), "A")))
    status = "matched_unique_name" if len(matches) == 1 else "matched_coalesced_profiles"
    return short_id(primary["id"], "A"), status, ids


def openalex_ids(faculty):
    """Return every OpenAlex author-profile alias attached to a faculty node."""
    values = (faculty.get("openalex_ids") or faculty.get("openalex_id") or "").split("|")
    return [aid for aid in (short_id(v, "A") for v in values) if aid]


def _merge_values(rows, field, separator="; "):
    values = []
    for row in rows:
        for value in (row.get(field) or "").split(separator):
            value = value.strip()
            if value and value not in values: values.append(value)
    return separator.join(values)


def coalesce_faculty(faculty, candidates_by_faculty):
    """Build canonical roster nodes from connected roster-name/OpenAlex-ID groups.

    Exact same-name OpenAlex profiles are aliases of one official roster node.
    Different roster-name variants are also merged when they share an OpenAlex ID.
    """
    parent = {row["faculty_id"]: row["faculty_id"] for row in faculty}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right: parent[max(left, right)] = min(left, right)

    owners = {}
    for row in faculty:
        for author in candidates_by_faculty.get(row["faculty_id"], []):
            key = (row["institution_id"], short_id(author.get("id"), "A"))
            if key in owners: union(row["faculty_id"], owners[key])
            else: owners[key] = row["faculty_id"]

    groups = defaultdict(list)
    for row in faculty: groups[find(row["faculty_id"])].append(row)
    merged = []
    diagnostics = []
    for rows in groups.values():
        # Prefer the most informative official spelling for display.
        representative = max(rows, key=lambda r: (len(normalized_name(r["name"]).split()), len(r["name"]), r["name"]))
        authors = {}
        for row in rows:
            for author in candidates_by_faculty.get(row["faculty_id"], []):
                authors[short_id(author.get("id"), "A")] = author
        aliases = sorted(authors)
        primary = max(aliases, key=lambda aid: (int(authors[aid].get("works_count") or 0), aid)) if aliases else ""
        roster_names = sorted({row["name"] for row in rows})
        profile_urls = sorted({row.get("profile_url", "") for row in rows if row.get("profile_url")})
        canonical = dict(representative)
        canonical.update({
            "faculty_id": representative["faculty_id"] if len(rows) == 1 else stable_id(
                representative["institution_id"], *sorted(row["faculty_id"] for row in rows)
            ),
            "name": representative["name"],
            "roster_names": "|".join(roster_names),
            "department": _merge_values(rows, "department"),
            "profile_url": representative.get("profile_url", ""),
            "profile_urls": "|".join(profile_urls),
            "faculty_source": "|".join(sorted({row.get("faculty_source", "") for row in rows if row.get("faculty_source")})),
            "orcid": next((row.get("orcid", "") for row in rows if row.get("orcid")), ""),
            "openalex_id": primary,
            "openalex_ids": "|".join(aliases),
            "roster_record_count": len(rows),
            "match_status": (
                "not_found" if not aliases else
                "matched_coalesced_names_and_profiles" if len(rows) > 1 and len(aliases) > 1 else
                "matched_coalesced_names" if len(rows) > 1 else
                "matched_coalesced_profiles" if len(aliases) > 1 else
                "matched_unique_name"
            ),
        })
        merged.append(canonical)
        diagnostics.append({
            "faculty_id": canonical["faculty_id"], "name": canonical["name"],
            "roster_names": canonical["roster_names"], "match_status": canonical["match_status"],
            "candidate_ids": canonical["openalex_ids"], "profile_urls": canonical["profile_urls"],
        })
    return sorted(merged, key=lambda r: (r["institution_name"], r["name"])), diagnostics


def collect_openalex(cfg, oa, output_dir):
    path = Path(output_dir)
    faculty = read_csv(path/"faculty.csv")
    max_faculty = cfg.get("settings",{}).get("max_faculty")
    max_papers = cfg.get("settings",{}).get("max_papers_per_author")
    settings = cfg.get("settings", {})
    candidates_by_faculty = {}
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
            # The official roster defines the person. Every exact-name OpenAlex
            # profile is retained as an alias, including directory conflicts.
            candidates_by_faculty[person["faculty_id"]] = indexed.get(normalized_name(person["name"]), [])
    original_roster_rows = len(faculty)
    faculty, diagnostics = coalesce_faculty(faculty, candidates_by_faculty)
    write_csv(path/"faculty.csv", faculty, FIELDS)
    write_csv(path/"match_diagnostics.csv", diagnostics,
              ["faculty_id", "name", "roster_names", "match_status", "candidate_ids", "profile_urls"])
    accepted_faculty = [row for row in faculty if openalex_ids(row)]
    if max_faculty is not None:
        selected = {row["faculty_id"] for row in sorted(accepted_faculty, key=lambda r:r["faculty_id"])[:int(max_faculty)]}
        for person in faculty:
            if openalex_ids(person) and person["faculty_id"] not in selected:
                person["openalex_id"] = ""
                person["openalex_ids"] = ""
                person["match_status"] = "matched_excluded_by_limit"
        write_csv(path/"faculty.csv", faculty, FIELDS)
        accepted_faculty = [row for row in faculty if row["faculty_id"] in selected]
    accepted_aliases = {aid for row in accepted_faculty for aid in openalex_ids(row)}
    LOG.info("Matched %s/%s canonical faculty nodes from %s roster rows (%s OpenAlex profiles)",
             len(accepted_faculty), len(faculty), original_roster_rows, len(accepted_aliases))

    work_map = {}
    max_locations = int(cfg.get("settings", {}).get("max_landing_pages", 4))
    batch_size = max(1, min(int(cfg.get("settings",{}).get("author_batch_size",20)), 100))

    def retain_work(w):
        wid = short_id(w.get("id"), "W")
        if not wid or wid in work_map: return
        if _needs_full_work(w): w = oa.get("works/"+wid)
        compact = compact_work(w, accepted_aliases, max_locations=max_locations)
        if not compact["authorships"]:
            LOG.warning("Skipping work %s: no selected faculty profile remained after expansion", wid)
            return
        work_map[wid] = compact

    if max_papers is not None:
        # Query every alias, union its results under the roster node, then apply
        # the cap once to that canonical professor. Recent works win the cap.
        groups = list(accepted_faculty)
        for n, person in enumerate(progress_iter(groups, settings=settings, desc="OpenAlex faculty works",
                                                 unit="faculty", total=len(groups)), 1):
            aliases = openalex_ids(person)
            LOG.info("Fetching works: faculty %s/%s (%s OpenAlex profiles)", n, len(groups), len(aliases))
            candidates = {}
            for aid in aliases:
                for w in oa.pages("works", f"authorships.author.id:{aid}", select=WORK_SELECT,
                                  max_results=int(max_papers)):
                    wid = short_id(w.get("id"), "W")
                    if wid: candidates[wid] = w
            ranked = sorted(candidates.values(), key=lambda w: (
                w.get("publication_date") or "", int(w.get("publication_year") or 0),
                short_id(w.get("id"), "W")
            ), reverse=True)[:int(max_papers)]
            for w in ranked: retain_work(w)
    else:
        sorted_aids = sorted(accepted_aliases)
        batches = [sorted_aids[i:i+batch_size] for i in range(0,len(sorted_aids),batch_size)]
        for n, group in enumerate(progress_iter(batches, settings=settings, desc="OpenAlex work batches",
                                                unit="batch", total=len(batches)), 1):
            LOG.info("Fetching works: author batch %s/%s (%s profiles)",n,len(batches),len(group))
            ids_filter = "|".join(group)
            for w in oa.pages("works", f"authorships.author.id:{ids_filter}", select=WORK_SELECT):
                retain_work(w)
    compressed_path = path/"raw"/"papers_openalex.jsonl.gz"
    legacy_path = path/"raw"/"papers_openalex.jsonl"
    write_jsonl(compressed_path, (work_map[k] for k in sorted(work_map)))
    # The compressed file is written atomically before its obsolete predecessor
    # is removed, so an interrupted run cannot erase the only raw-work output.
    legacy_path.unlink(missing_ok=True)
    LOG.info("Collected %s unique works from %s canonical faculty nodes (%s OpenAlex profiles)",
             len(work_map), len(accepted_faculty), len(accepted_aliases))
    return faculty, work_map
