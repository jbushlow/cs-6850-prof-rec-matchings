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
    faculty_by_author = defaultdict(list)
    for f in faculty:
        if f["openalex_id"]: faculty_by_author[f["openalex_id"]].append(f)
    faculty_ids = set(faculty_by_author)

    authors = {}
    paper_author_rows = []
    collaborations = {}
    max_clique = int(settings.get("max_clique_authors", 100))
    skipped_large = []
    for wid, work in progress_iter(original.items(), settings=settings,
                                   desc="Building coauthor graph", unit="paper",
                                   total=len(original)):
        paper = papers.get(wid)
        if paper is None: continue
        # Stage 2 has already restricted authorships to selected faculty nodes.
        aids = []
        for order,a in enumerate(work.get("authorships") or [],1):
            ref = a.get("author") or {}
            aid = short_id(ref.get("id"),"A")
            if not aid: continue
            authors.setdefault(aid, {"openalex_id":aid,
                                     "name":ref.get("display_name") or "",
                                     "is_faculty":int(aid in faculty_ids),
                                     "faculty_institution_ids":"|".join(sorted({f["institution_id"] for f in faculty_by_author.get(aid,[])}))})
            institution_ids = sorted({short_id(x.get("id"),"I") for x in a.get("institutions") or [] if x.get("id")})
            paper_author_rows.append({"paper_id":wid,"author_id":aid, "byline_index":order,
                                      "author_position":a.get("author_position") or "",
                                      "institution_ids":"|".join(institution_ids),
                                      "raw_affiliation": "|".join(a.get("raw_affiliation_strings") or [])})
            if aid not in aids: aids.append(aid)
        if len(aids)>max_clique:
            skipped_large.append({"paper_id":wid, "authors_count":len(aids)})
            continue
        for u,v in combinations(sorted(aids),2):
            edge = collaborations.setdefault((u,v), {"papers":set(),"last_year":0})
            edge["papers"].add(wid)
            year = paper.get("year") or 0
            if isinstance(year,int): edge["last_year"] = max(edge["last_year"],year)

    for aid, frows in faculty_by_author.items():
        authors.setdefault(aid, {"openalex_id":aid,"name":frows[0]["name"],
                                 "is_faculty":1,
                                 "faculty_institution_ids":"|".join(sorted({f["institution_id"] for f in frows}))})
    nodes = sorted(authors.values(),key=lambda x:x["openalex_id"])
    edges = []
    faculty_edges = []
    for (u,v),x in sorted(collaborations.items()):
        row = {"author_a":u, "author_b":v, "weight":len(x["papers"]),
               "last_year":x["last_year"],"paper_ids":json.dumps(sorted(x["papers"]))}
        edges.append(row)
        if u in faculty_ids and v in faculty_ids: faculty_edges.append(row)
    fields = ["author_a","author_b","weight","last_year","paper_ids"]
    write_csv(out/"graph"/"nodes.csv",nodes,
              ["openalex_id","name","is_faculty","faculty_institution_ids"])
    write_csv(out/"graph"/"authorships.csv",paper_author_rows,
              ["paper_id","author_id","byline_index","author_position","institution_ids","raw_affiliation"])
    write_csv(out/"graph"/"coauthor_edges.csv",edges,fields)
    write_csv(out/"graph"/"faculty_edges.csv",faculty_edges,fields)
    write_csv(out/"graph"/"skipped_large_teams.csv",skipped_large,["paper_id","authors_count"])

    coverage = []
    for aid in progress_iter(sorted(faculty_ids), settings=settings,
                             desc="Faculty coverage", unit="faculty",
                             total=len(faculty_ids)):
        f = faculty_by_author[aid][0]
        pw = [p for p in papers.values() if aid in p["author_ids"]]
        good = [p for p in pw if p["abstract_status"]=="found"]
        source_counts = defaultdict(int)
        for p in good: source_counts[p["abstract_source"]]+=1
        coverage.append({"openalex_id":aid,"name":f["name"],"institution_ids": "|".join(sorted({r["institution_id"] for r in faculty_by_author[aid]})),
                         "total_papers":len(pw),"abstracts_found":len(good),
                         "abstract_fraction":round(len(good)/len(pw),5) if pw else "",
                         "openalex_abstracts":source_counts["openalex"],
                         "api_recovered_abstracts":sum(source_counts[s] for s in ("crossref","semantic_scholar","europe_pmc")),
                         "web_recovered_abstracts":source_counts["landing_page"],
                         "missing_abstracts":len(pw)-len(good)})
    write_csv(out/"faculty_coverage.csv", coverage,
              ["openalex_id","name","institution_ids","total_papers","abstracts_found","abstract_fraction",
               "openalex_abstracts","api_recovered_abstracts","web_recovered_abstracts","missing_abstracts"])
    stats = {"faculty_roster_rows":len(faculty),"matched_faculty_rows":sum(bool(x["openalex_id"]) for x in faculty),
             "matched_unique_faculty_authors":len(faculty_ids), "papers":len(papers),
             "papers_with_abstracts":sum(p["abstract_status"]=="found" for p in papers.values()),
             "authors_in_retrieved_papers":len(authors),"projected_coauthor_edges":len(edges),
             "projected_faculty_edges":len(faculty_edges),
             "skipped_large_team_papers":len(skipped_large),
             "max_clique_authors":max_clique}
    (out/"summary.json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
    if settings.get("graphml"):
        import networkx as nx
        for name, dataset, faculty_only in (("all_authors",edges,False),("faculty",faculty_edges,True)):
            G = nx.Graph()
            for n in nodes:
                if not faculty_only or n["is_faculty"]:
                    G.add_node(n["openalex_id"],name=n["name"],is_faculty=int(n["is_faculty"]),
                               faculty_institution_ids=n["faculty_institution_ids"])
            for edge in dataset:
                G.add_edge(edge["author_a"],edge["author_b"],weight=edge["weight"],
                           last_year=edge["last_year"],paper_ids=edge["paper_ids"])
            nx.write_graphml(G, out/"graph"/(name+".graphml"))
    LOG.info("Graph/export summary: %s", stats)
    return stats
