"""Stage 4-5: export faculty-only authorships and coauthorship projections."""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from .common import read_csv, read_jsonl, short_id, write_csv
from .progress import progress_iter
LOG = logging.getLogger(__name__)


def build_graph(output_dir, settings=None):
    settings = settings or {}
    out = Path(output_dir)
    faculty = read_csv(out/"faculty.csv")
    papers = {p["openalex_id"]:p for p in read_jsonl(out/"papers.jsonl")}
    compressed = out/"raw"/"papers_openalex.jsonl.gz"
    legacy = out/"raw"/"papers_openalex.jsonl"
    raw_path = compressed if compressed.exists() else legacy
    original = {short_id(p["id"],"W"):p for p in read_jsonl(raw_path)}
    faculty_by_id = {f["faculty_id"]:f for f in faculty}
    alias_to_faculty = {}
    for f in faculty:
        aliases = (f.get("openalex_ids") or f.get("openalex_id") or "").split("|")
        for alias in aliases:
            aid = short_id(alias,"A")
            if aid: alias_to_faculty[aid] = f["faculty_id"]
    faculty_ids = set(faculty_by_id)

    authors = {}
    paper_author_rows = []
    collaborations = {}
    observed_faculty_ids = set()
    max_clique = int(settings.get("max_clique_authors", 100))
    skipped_large = []
    for wid, work in progress_iter(original.items(), settings=settings,
                                   desc="Building coauthor graph", unit="paper",
                                   total=len(original)):
        paper = papers.get(wid)
        if paper is None: continue
        # Stage 2 has already restricted authorships to selected faculty nodes.
        retained = {}
        for order,a in enumerate(work.get("authorships") or [],1):
            ref = a.get("author") or {}
            aid = short_id(ref.get("id"),"A")
            fid = alias_to_faculty.get(aid)
            if not fid: continue
            institution_ids = sorted({short_id(x.get("id"),"I") for x in a.get("institutions") or [] if x.get("id")})
            entry = retained.setdefault(fid, {"orders":[], "positions":[], "institution_ids":set(),
                                               "affiliations":set(), "openalex_ids":set()})
            entry["orders"].append(order)
            if a.get("author_position"): entry["positions"].append(a["author_position"])
            entry["institution_ids"].update(institution_ids)
            entry["affiliations"].update(x for x in a.get("raw_affiliation_strings") or [] if x)
            entry["openalex_ids"].add(aid)
        aids = sorted(retained)
        observed_faculty_ids.update(aids)
        for fid in aids:
            entry = retained[fid]
            paper_author_rows.append({"paper_id":wid,"author_id":fid,
                                      "openalex_author_ids":"|".join(sorted(entry["openalex_ids"])),
                                      "byline_index":min(entry["orders"]),
                                      "author_position":"|".join(dict.fromkeys(entry["positions"])),
                                      "institution_ids":"|".join(sorted(entry["institution_ids"])),
                                      "raw_affiliation":"|".join(sorted(entry["affiliations"]))})
        if len(aids)>max_clique:
            skipped_large.append({"paper_id":wid, "authors_count":len(aids)})
            continue
        for u,v in combinations(sorted(aids),2):
            edge = collaborations.setdefault((u,v), {"papers":set(),"last_year":0})
            edge["papers"].add(wid)
            year = paper.get("year") or 0
            if isinstance(year,int): edge["last_year"] = max(edge["last_year"],year)

    for fid, f in faculty_by_id.items():
        authors[fid] = {"faculty_id":fid,"name":f["name"],
                        "roster_names":f.get("roster_names") or f["name"],
                        "openalex_ids":f.get("openalex_ids") or f.get("openalex_id", ""),
                        "institution_ids":f["institution_id"],
                        "departments":f.get("department", ""),
                        "match_status":f.get("match_status", "")}
    nodes = sorted(authors.values(),key=lambda x:x["faculty_id"])
    faculty_edges = []
    for (u,v),x in sorted(collaborations.items()):
        row = {"author_a":u, "author_b":v, "weight":len(x["papers"]),
               "last_year":x["last_year"],"paper_ids":json.dumps(sorted(x["papers"]))}
        if u in faculty_ids and v in faculty_ids: faculty_edges.append(row)
    fields = ["author_a","author_b","weight","last_year","paper_ids"]
    write_csv(out/"graph"/"nodes.csv",nodes,
              ["faculty_id","name","roster_names","openalex_ids","institution_ids","departments","match_status"])
    write_csv(out/"graph"/"authorships.csv",paper_author_rows,
              ["paper_id","author_id","openalex_author_ids","byline_index","author_position","institution_ids","raw_affiliation"])
    write_csv(out/"graph"/"faculty_edges.csv",faculty_edges,fields)
    write_csv(out/"graph"/"skipped_large_teams.csv",skipped_large,["paper_id","authors_count"])
    # Remove artifacts created by older versions of the faculty-only exporter.
    for legacy in (out/"graph"/"coauthor_edges.csv", out/"graph"/"all_authors.graphml"):
        if legacy.exists(): legacy.unlink()

    coverage = []
    for fid in progress_iter(sorted(faculty_ids), settings=settings,
                             desc="Faculty coverage", unit="faculty",
                             total=len(faculty_ids)):
        f = faculty_by_id[fid]
        pw = [p for p in papers.values() if fid in p["author_ids"]]
        good = [p for p in pw if p["abstract_status"]=="found"]
        source_counts = defaultdict(int)
        for p in good: source_counts[p["abstract_source"]]+=1
        coverage.append({"faculty_id":fid,"openalex_ids":f.get("openalex_ids") or f.get("openalex_id", ""),
                         "name":f["name"],"institution_ids":f["institution_id"],
                         "total_papers":len(pw),"abstracts_found":len(good),
                         "abstract_fraction":round(len(good)/len(pw),5) if pw else "",
                         "openalex_abstracts":source_counts["openalex"],
                         "api_recovered_abstracts":sum(source_counts[s] for s in ("crossref","semantic_scholar","europe_pmc")),
                         "web_recovered_abstracts":source_counts["landing_page"],
                         "missing_abstracts":len(pw)-len(good)})
    write_csv(out/"faculty_coverage.csv", coverage,
              ["faculty_id","openalex_ids","name","institution_ids","total_papers","abstracts_found","abstract_fraction",
               "openalex_abstracts","api_recovered_abstracts","web_recovered_abstracts","missing_abstracts"])
    stats = {"faculty_nodes":len(faculty),
             "matched_faculty_nodes":sum(bool((x.get("openalex_ids") or x.get("openalex_id"))) for x in faculty),
             "openalex_author_profiles":len(alias_to_faculty), "papers":len(papers),
             "papers_with_abstracts":sum(p["abstract_status"]=="found" for p in papers.values()),
             "faculty_nodes_with_papers":len(observed_faculty_ids),
             "projected_faculty_edges":len(faculty_edges),
             "skipped_large_team_papers":len(skipped_large),
             "max_clique_authors":max_clique}
    (out/"summary.json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
    if settings.get("graphml"):
        import networkx as nx
        G = nx.Graph()
        for n in nodes:
            G.add_node(n["faculty_id"],name=n["name"],openalex_ids=n["openalex_ids"],
                       institution_ids=n["institution_ids"],departments=n["departments"])
        for edge in faculty_edges:
            G.add_edge(edge["author_a"],edge["author_b"],weight=edge["weight"],
                       last_year=edge["last_year"],paper_ids=edge["paper_ids"])
        nx.write_graphml(G, out/"graph"/"faculty.graphml")
    LOG.info("Graph/export summary: %s", stats)
    return stats
