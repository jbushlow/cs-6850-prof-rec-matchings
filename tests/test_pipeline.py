import csv
import json
from pathlib import Path
import pytest
from facultygraph.common import normalize_doi, read_csv, read_jsonl, write_csv, write_jsonl
from facultygraph.faculty import (collect_faculty, parse_directory_html,
                                  load_indexed_html_source, canonical_profile_url)
from facultygraph.openalex import (WORK_SELECT, _needs_full_work, compact_work,
                                   match_one, collect_openalex)
from facultygraph.abstracts import (decode_inverted_index, valid_abstract, extract_html_abstract,
                                    AbstractCollector)
from facultygraph.graph import build_graph

LONG_ABS = "This study introduces an approach to computational network analysis that measures collaboration patterns and research expertise across multiple scientific communities and several different institutions using bibliographic metadata and text analysis of publication abstracts."

class FakeHttp:
    def __init__(self, pages): self.pages = pages; self.calls = []; self.options = []
    def robots_allowed(self, url, **kwargs): return True
    def get(self, url, params=None, headers=None, text=False, allow_not_found=False, cache=True,
            max_attempts=5, timeout=35):
        self.calls.append((url,params))
        self.options.append((max_attempts, timeout))
        if url not in self.pages:
            if allow_not_found: return None
            raise AssertionError(url)
        return self.pages[url]

class FakeOpenAlex:
    def __init__(self, authors, works): self.authors=authors; self.works=works; self.calls=[]
    def pages(self, endpoint, filters, select=None, max_results=None):
        self.calls.append((endpoint,filters))
        if endpoint == "authors": data = self.authors
        else:
            ids = filters.split(":")[-1].split("|")
            seen = {}
            for aid in ids:
                for w in self.works[aid]: seen[w["id"]] = w
            data = list(seen.values())
        yield from data[:max_results] if max_results else data
    def get(self, endpoint, params=None): raise AssertionError(f"Unexpected: {endpoint}")

def test_directory_scraping_and_csv(tmp_path):
    src = tmp_path/"faculty.csv"
    src.write_text("name,department,profile_url\nAlice Smith,Computer Science,https://a.edu/alice\n",encoding="utf-8")
    config = {"institutions":[{"name":"A","openalex_id":"I20", "faculty_sources":[{"type":"csv","path":str(src)}]}]}
    rows = collect_faculty(config,None,tmp_path,tmp_path/"output")
    assert len(rows)==1 and rows[0]["name"]=="Alice Smith"
    html = '<article class="faculty"><a class="person" href="/jane">Jane Doe</a><span class="dept">ECE</span></article>'
    result, _ = parse_directory_html(html,"https://a.edu/faculty",config["institutions"][0],
                                    {"card_selector":"article.faculty","name_selector":"a.person",
                                     "profile_link_selector":"a.person","department_selector":".dept"})
    assert result[0]["profile_url"]=="https://a.edu/jane"
    assert result[0]["department"]=="ECE"

    compressed = tmp_path/"roundtrip.jsonl.gz"
    write_jsonl(compressed, [{"value": "compressed"}])
    assert list(read_jsonl(compressed)) == [{"value": "compressed"}]


def test_indexed_directory_discovers_filters_and_deduplicates_pages():
    index_url = "https://a.edu/departments"
    anthropology = "https://a.edu/department/anthropology"
    program = "https://a.edu/department/interdisciplinary-program"
    pages = {
        index_url: '''<h2><a class="detail" href="/department/anthropology">Anthropology</a></h2>
                      <a class="detail" href="/department/anthropology">duplicate</a>
                      <h2><a class="detail" href="/department/interdisciplinary-program">Program</a></h2>''',
        anthropology: '''<main><p class="type">Academic Department</p><h1>Anthropology</h1>
                         <section class="faculty"><ul><li><a href="/people/jane">Jane Doe</a></li></ul></section></main>''',
        program: '''<main><p class="type">Academic Program</p><h1>Program</h1>
                    <section class="faculty"><ul><li><a href="/people/alex">Alex Roe</a></li></ul></section></main>''',
    }
    http = FakeHttp(pages)
    source = {
        "index_url": index_url,
        "detail_link_selector": "a.detail",
        "required_text_selector": ".type",
        "required_text": "Academic Department",
        "card_selector": ".faculty li",
        "name_selector": "a",
        "profile_link_selector": "a",
        "department_page_selector": "h1",
    }
    inst = {"name": "A", "openalex_id": "I20"}
    rows = load_indexed_html_source(inst, source, http)
    assert [(r["name"], r["department"], r["profile_url"]) for r in rows] == [
        ("Jane Doe", "Anthropology", "https://a.edu/people/jane")
    ]
    assert [call[0] for call in http.calls] == [index_url, anthropology, program]


def test_cross_host_profile_aliases_merge_but_same_host_profiles_conflict(tmp_path):
    first = tmp_path/"first.csv"
    second = tmp_path/"second.csv"
    first.write_text(
        "name,department,profile_url\nJane Doe,Engineering,https://engineering.a.edu/jane\n",
        encoding="utf-8",
    )
    second.write_text(
        "name,department,profile_url\nJane Doe,Physics,https://as.a.edu/people/jane\n",
        encoding="utf-8",
    )
    inst = {
        "name": "A",
        "openalex_id": "I20",
        "allow_cross_host_profile_aliases": True,
        "faculty_sources": [
            {"type": "csv", "path": str(first)},
            {"type": "csv", "path": str(second)},
        ],
    }
    rows = collect_faculty({"institutions": [inst]}, None, tmp_path, tmp_path/"out")
    assert len(rows) == 1
    assert rows[0]["department"] == "Engineering; Physics"
    assert rows[0]["match_status"] == "unmatched"

    second.write_text(
        "name,department,profile_url\nJane Doe,Physics,https://engineering.a.edu/jane-2\n",
        encoding="utf-8",
    )
    rows = collect_faculty({"institutions": [inst]}, None, tmp_path, tmp_path/"out")
    assert rows[0]["match_status"] == "directory_conflict"


def test_profile_url_canonicalization_handles_legacy_index_aliases(tmp_path):
    assert canonical_profile_url(
        "HTTPS://AS.CORNELL.EDU/index%2Ephp/people/jane-doe/"
    ) == "https://as.cornell.edu/people/jane-doe"

    first = tmp_path/"first.csv"
    second = tmp_path/"second.csv"
    first.write_text(
        "name,department,profile_url\nJane Doe,History,https://as.cornell.edu/index%2Ephp/people/jane-doe\n",
        encoding="utf-8",
    )
    second.write_text(
        "name,department,profile_url\nJane Doe,Classics,https://as.cornell.edu/people/jane-doe/\n",
        encoding="utf-8",
    )
    inst = {
        "name": "A",
        "openalex_id": "I20",
        "faculty_sources": [
            {"type": "csv", "path": str(first)},
            {"type": "csv", "path": str(second)},
        ],
    }
    rows = collect_faculty({"institutions": [inst]}, None, tmp_path, tmp_path/"out")
    assert len(rows) == 1
    assert rows[0]["match_status"] == "unmatched"
    assert rows[0]["department"] == "History; Classics"

def _author(aid,name,inst="I20",orcid=""):
    return {"id":"https://openalex.org/"+aid,"display_name":name,"display_name_alternatives":[],
            "orcid":orcid,"observed_orcids":[],"affiliations":[{"institution":{"id":"https://openalex.org/"+inst},"years":[2023]}],
            "last_known_institutions":[]}

def test_match_rejects_ambiguous_and_accepts_directory_orcid():
    one=_author("A123","Walker White")
    two=_author("A456","Walker White",orcid="https://orcid.org/0000-0002-1825-0097")
    assert match_one({"name":"Walker White","orcid":""},[one,two])[1]=="ambiguous_name"
    assert match_one({"name":"Walker White","orcid":"0000-0002-1825-0097"},[one,two])[:2] == ("A456","matched_orcid")
    assert match_one({"name":"Someone Else","orcid":""},[one])[1]=="not_found"

def _work(wid, authors, abstract=None):
    return {"id":"https://openalex.org/"+wid,"title":"Research collaboration paper",
            "doi":None,"publication_year":2025,"publication_date":"2025-02-01","type":"article",
            "ids":{},"is_authors_truncated":False,"authorships":[{"author":{"id":"https://openalex.org/"+a,
             "display_name":a},"author_position":"middle","institutions":[]} for a in authors],
            "abstract_inverted_index":abstract,"locations":[],"locations_count":0}


def test_work_select_uses_api_truncation_field():
    assert "is_authors_truncated" in WORK_SELECT
    assert "authors_count" not in WORK_SELECT
    assert _needs_full_work({"authorships": [{}] * 100})
    assert _needs_full_work({"authorships": [], "is_authors_truncated": True})
    assert not _needs_full_work({"authorships": [{}] * 99})


def test_compact_work_keeps_only_faculty_authors_and_required_fields():
    work = _work("W10", ["A1", "A2", "A3"])
    work["authorships"][0]["institutions"] = [
        {"id": "https://openalex.org/I20", "display_name": "A", "country_code": "US"}
    ]
    work["authorships"][0]["raw_affiliation_strings"] = ["Department A"]
    work["locations"] = [
        {"landing_page_url": "https://one.example", "pdf_url": "https://one.example/p.pdf"},
        {"landing_page_url": "https://two.example"},
    ]
    work["unneeded"] = "discard me"
    compact = compact_work(work, {"A1", "A2"}, max_locations=1)
    assert [a["author"]["id"].rsplit("/", 1)[-1] for a in compact["authorships"]] == ["A1", "A2"]
    assert compact["authorships"][0]["institutions"] == [{"id": "https://openalex.org/I20"}]
    assert compact["locations"] == [{"landing_page_url": "https://one.example"}]
    assert "abstract_inverted_index" not in compact
    assert "abstract" in compact
    assert "unneeded" not in compact

def test_end_to_end_with_fake_openalex_and_abstracts(tmp_path):
    output = tmp_path/"output"
    output.mkdir()
    rows = [dict(faculty_id="f1",name="Alice Smith",institution_name="A",institution_id="I20",department="ECE",
                 profile_url="",orcid="",faculty_source="test",openalex_id="",match_status="unmatched"),
            dict(faculty_id="f2",name="Bob Jones",institution_name="A",institution_id="I20",department="CS",
                 profile_url="",orcid="",faculty_source="test",openalex_id="",match_status="unmatched")]
    write_csv(output/"faculty.csv",rows,list(rows[0]))
    write_jsonl(output/"raw"/"papers_openalex.jsonl", [{"legacy": True}])
    index={"Network":[0],"science":[1]}  # invalid (short) abstract triggers fallback
    w1=_work("W1",["A1","A2","A3"],index)
    w2=_work("W2",["A1","A3"],None)
    w3=_work("W3",["A2","A4"],None)
    api=FakeOpenAlex([_author("A1","Alice Smith"),_author("A2","Bob Jones"),_author("A3","Student One")],
                    {"A1":[w1,w2],"A2":[w1,w3]})
    faculty, work_map = collect_openalex({"institutions":[]},api,output)
    assert {f["openalex_id"] for f in faculty}=={"A1","A2"}
    assert len(work_map)==3
    assert (output/"raw"/"papers_openalex.jsonl.gz").exists()
    assert not (output/"raw"/"papers_openalex.jsonl").exists()
    assert list(read_csv(output/"match_diagnostics.csv")[0]) == [
        "faculty_id", "name", "match_status", "candidate_ids"
    ]
    assert all(
        {a["author"]["id"].rsplit("/", 1)[-1] for a in work["authorships"]} <= {"A1", "A2"}
        for work in work_map.values()
    )
    assert api.calls[0]==("authors","affiliations.institution.id:I20")
    # Avoid fallback test websites by putting valid source abstract for W1 and using offline stubs for others.
    class TestCollector(AbstractCollector):
        def _crossref(self,w):
            return (LONG_ABS,"crossref","https://crossref.example") if w["id"].endswith("W2") else None
        def _s2(self,w): return None
        def _europepmc(self,w): return None
        def _html(self,w): return None
    collector=TestCollector(FakeHttp({}),output,{"abstract_min_words":15})
    docs=collector.run()
    assert len(docs)==3
    assert sum(x["abstract_status"]=="found" for x in docs)==1
    stats=build_graph(output,{"max_clique_authors":100})
    assert stats["papers"]==3
    assert stats["projected_faculty_edges"]==1
    assert stats["projected_coauthor_edges"]==1
    edges=read_csv(output/"graph"/"coauthor_edges.csv")
    assert edges[0]["author_a"]=="A1" and edges[0]["author_b"]=="A2"
    coverage=read_csv(output/"faculty_coverage.csv")
    assert next(r for r in coverage if r["openalex_id"]=="A1")["abstract_fraction"]=="0.5"
    # Result cache means second pass emits no new extraction calls
    previous=len(list(read_jsonl(output/"state"/"abstract_results.jsonl")))
    collector.run()
    assert len(list(read_jsonl(output/"state"/"abstract_results.jsonl")))==previous

def test_abstract_extraction_and_page_verification():
    abstract = "Some"+" important"*40
    inv={"A":[0,2], "B":[1]}
    assert decode_inverted_index(inv)=="A B A"
    assert valid_abstract(abstract,25)
    html=f'''<html><head><meta name="citation_title" content="Hardware-Aware Scientific Collaboration">
             <meta name="citation_doi" content="10.1/abc">
             <meta name="citation_abstract" content="{abstract}"></head></html>'''
    assert extract_html_abstract(html,"Hardware-Aware Scientific Collaboration","https://doi.org/10.1/abc",25)
    assert not extract_html_abstract(html,"Other title","10.1/xyz",25)
    assert not extract_html_abstract(html,"Other title","",25)

def test_provider_fallback_order_and_web_parse(tmp_path,monkeypatch):
    # For this work, Crossref fails to provide an abstract; S2 succeeds. EuropePMC
    # and publisher HTML should never be queried after a valid S2 hit.
    work = _work("W101",["A101"])
    work["title"] = "Research collaboration paper"
    work["doi"] = "https://doi.org/10.1000/example"
    c_url = "https://api.crossref.org/works/10.1000/example"
    s_url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1000/example"
    api = FakeHttp({
        c_url: {"message":{"title":[work["title"]]}},
        s_url: {"title":work["title"],"abstract":LONG_ABS,"year":2025},
    })
    collector = AbstractCollector(api,tmp_path,{"abstract_min_words":20})
    hit = collector.collect_one(work)
    assert hit["status"] == "found" and hit["source"] == "semantic_scholar"
    assert len(api.calls) == 2
    assert api.calls[0][0] == c_url and api.calls[1][0] == s_url
    # On another object, direct OpenAlex abstract prevents *any* HTTP request.
    text = LONG_ABS.split()
    index = {}
    for i, token in enumerate(text): index.setdefault(token,[]).append(i)
    work2 = _work("W102",["A101"], index)
    assert collector.collect_one(work2)["source"] == "openalex"
    assert len(api.calls) == 2


def test_europe_pmc_uses_doi_without_openalex_biomedical_ids(tmp_path):
    work = _work("W103", ["A101"])
    work["title"] = "Policy Development for Pandemic Response Using System Dynamics"
    work["doi"] = "https://doi.org/10.1007/s41660-020-00130-x"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    api = FakeHttp({url: {"resultList": {"result": [{
        "title": work["title"], "doi": "10.1007/s41660-020-00130-x",
        "abstractText": LONG_ABS,
    }]}}})
    collector = AbstractCollector(api, tmp_path, {
        "abstract_min_words": 20,
        "abstract_max_attempts": 3,
        "abstract_request_timeout": 20,
    })

    hit = collector._europepmc(work)

    assert hit[1] == "europe_pmc"
    assert api.calls[0][1]["query"] == "DOI:10.1007/s41660-020-00130-x"
    assert api.options[0] == (3, 20.0)


def test_abstract_fallbacks_can_be_disabled_and_cached_fallback_is_not_reused(tmp_path):
    work = _work("W104", ["A101"])
    work["doi"] = "https://doi.org/10.1000/no-fallback"
    write_jsonl(tmp_path/"state"/"abstract_results.jsonl", [{
        "paper_id": "W104", "status": "found", "abstract": LONG_ABS,
        "source": "crossref", "source_url": "https://crossref.example/W104",
    }])
    api = FakeHttp({})
    collector = AbstractCollector(api, tmp_path, {
        "abstract_min_words": 20,
        "abstract_fallbacks": False,
    })

    result = collector.collect_one(work)

    assert result["status"] == "missing"
    assert result["source"] is None
    assert api.calls == []
    assert list(read_jsonl(tmp_path/"state"/"abstract_results.jsonl"))[-1] == result


def test_webpage_abstract_section():
    abstract = LONG_ABS
    page = f'''<html><head><title>Target experimental paper | journal</title></head>
       <body><h2>Abstract</h2><p>{abstract}</p><h2>References</h2></body></html>'''
    assert extract_html_abstract(page,"Target experimental paper",min_words=20) == abstract
