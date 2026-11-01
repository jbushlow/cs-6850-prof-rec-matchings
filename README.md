# CS 6850 Professor Recommendation Matchings

Cornell CS 6850 (Fall 2026) project by Jiasheng Shi, Julian Bushlow, and Sidharth Rao.

## Faculty collaboration data pipeline

A small, scriptable ETL package for **faculty rosters → OpenAlex curated-core works → abstract recovery → author–paper and coauthor graphs**. Built for network-research experiments, not as a standalone recommendation system. The graph processing, embeddings, community detection, structural-hole scoring, and recommendation algorithms belong in separate downstream packages.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[test]'
playwright install chromium          # one-time browser install for JS directories
cp .env.example .env                # optional; env files are NOT auto-loaded
export OPENALEX_API_KEY='your_free_openalex_key'
export RESEARCH_CONTACT_EMAIL='you@university.edu'

# A small offline dry-run of the directory/CSV stage, using synthetic records:
facultygraph faculty --config configs/local-demo.yaml

# Cornell's configured public directories (including browser-rendered Engineering):
facultygraph faculty --config configs/cornell.yaml

# For a REAL university, edit configs/example.yaml with valid institution IDs
# and real directory URLs + selectors, or use CSV roster export(s).
facultygraph faculty   --config configs/example.yaml
facultygraph openalex  --config configs/example.yaml
facultygraph abstracts --config configs/example.yaml
facultygraph graph     --config configs/example.yaml
# Or run all stages sequentially:
facultygraph all --config configs/example.yaml

pytest -q
```

Interactive runs show progress bars for directory pages, OpenAlex batches,
abstract recovery, and graph construction. Pass `--no-progress` to suppress
them (for example, in a scheduled job or when redirecting output to a file).

**The provided `configs/example.yaml` contains an intentionally fake university and placeholder selectors. It is NOT ready for a real end-to-end run.** For institution-specific deployment, first inspect that institution's public, faculty-only directory HTML and configure real CSS selectors. Use `rendered_html` for a JavaScript-generated directory and `indexed_html` when one public index links to homogeneous department/detail pages. Nothing here bypasses logins, CAPTCHAs, robots restrictions, or publisher access controls.

### Prerequisites

* Python 3.10+
* Chromium installed through `playwright install chromium` when a config uses a `rendered_html` source.
* A **free OpenAlex API key** for practical pagination; set `OPENALEX_API_KEY`. The OpenAlex API currently requires a key when following cursor pages beyond the first one.
* Optional `SEMANTIC_SCHOLAR_API_KEY`, depending on provider limits.
* Contact email (`RESEARCH_CONTACT_EMAIL`) recommended for responsible API use.
* Appropriate permission under each institution directory's and publisher's access policies.

## Stage 1: Collect official faculty rosters

Your config lists universities and **explicitly specifies faculty-only sources**. Anyone listed under the relevant faculty directory is eligible, regardless of tenure track or appointment type. We do **not** infer faculty status from OpenAlex (which cannot reliably distinguish students and professors).

```yaml
institutions:
  - name: Cornell University
    openalex_id: I201448701   # example ID; verify institution before use
    # affiliation_ids: [I...] # optional child campuses/medical institutions
    faculty_sources:
      - type: html
        url: https://your-university.example/faculty
        card_selector: '.faculty-card'
        name_selector: '.name'
        profile_link_selector: 'a'
        department_selector: '.department'
        next_page_selector: 'a.next'
        max_pages: 20
      - type: rendered_html       # same selectors, but after JavaScript renders
        url: https://your-university.example/people?role=faculty
        card_selector: '.person-card'
        wait_selector: '.person-card'
        name_selector: '.name'
        profile_link_selector: '.name a'
        department_selector: '.department'
        next_page_selector: 'a[aria-label="Next Page"]'
        max_pages: 30
      - type: indexed_html        # discover and parse homogeneous detail pages
        index_url: https://your-university.example/departments
        detail_link_selector: 'a.department-link'
        required_text_selector: '.page-type' # optional detail-page filter
        required_text: 'Academic Department'
        card_selector: '.faculty-list li'
        name_selector: 'a'
        profile_link_selector: 'a'
        department_page_selector: 'h1'
        allow_empty_detail_pages: true
      - type: csv
        path: ../exports/faculty.csv # relative to config YAML file
        name_column: name
        department_column: department
        profile_url_column: profile_url
        orcid_column: orcid
```

Optional `faculty_selector` constrains cards to those containing a faculty-specific child marker. Use a faculty-only page whenever possible. If a directory has multiple pages and a Next link, `next_page_selector` supports pagination; the collector does not auto-discover arbitrary pages. `rendered_html` uses the same extraction fields after loading the page in headless Chromium. `indexed_html` follows only links matching `detail_link_selector`, optionally filters detail pages by configured text, and uses `department_page_selector` to label records. Multiple sources combine, deduplicating normalized names **within an institution**. Directory conflicts in ORCID are retained as `directory_conflict` and are never auto-matched.

Output: `output/faculty.csv`. Fields include `faculty_id`, `name`, `institution_id`, `department`, `profile_url`, `orcid`, `openalex_id`, and `match_status`. The initial `openalex_id` is blank.

## Stage 2: Resolve faculty authors and collect their work histories

* Query `/authors?filter=affiliations.institution.id:<I...>` with cursor pagination for each configured institution. This finds publication-derived associations, **not employment classifications**. Sub-institutions can be listed in `affiliation_ids` (e.g. medical-school branches).
* Intersect OpenAlex author names/alternatives with the **official** faculty roster (exact normalized name). Only a **unique match** is accepted, unless an ORCID explicitly extracted from the faculty roster disambiguates multiple candidates. Reject multiple candidates and conflicts; **no human review** and no guesses based on search rank. Missing a match is preferable to accepting a false brokerage node.
* Batch matched faculty IDs in OR filters (config `author_batch_size`, default 20) to fetch full publication histories from the **curated core** with `corpus=core`. This reduces duplicate requests from coauthorship overlap. Request full single-work records only when list results truncate author or location lists.
* Keep only authorships whose OpenAlex IDs belong to the selected faculty roster. Students, postdocs, outside professors and other non-node coauthors are removed before raw work records are saved.
* Compact retained authorships to the fields used downstream: faculty ID/name, byline position, institution IDs and raw affiliation strings. Location objects are reduced to the small set of landing-page URLs used for abstract recovery.
* The stage does **not** require abstracts. All publications with usable authorships remain available for structural analysis.

Outputs: `output/faculty.csv` (updated IDs/status), a compact `output/match_diagnostics.csv` (`faculty_id`, name, status and candidate IDs), and compressed `output/raw/papers_openalex.jsonl.gz` records. OpenAlex abstract inverted indexes are decoded to plain text before storage.

**Scaling note:** A full institution-wide authors scan can be expensive in API credits; collecting thousands of institutions should move to the OpenAlex snapshot or a narrower roster-based lookup. Configure only the institutions you actually want, monitor usage, and reuse the HTTP cache. Free-key budgets and rate limits change; refer to current OpenAlex documentation before a large job.

**Selection caveat:** This source population is *faculty with a matching institution-associated OpenAlex profile*. Faculty without a recent indexed affiliation, or with an alternative spelling not in their OpenAlex names, are automatically excluded and counted in the diagnostics. False positives/negatives remain possible even for unique-name matches.

**Optional quick sampling:** set `max_faculty: 10` and `max_papers_per_author: 100` in `settings`. The former limits accepted author IDs for a development run; the latter limits works returned per author (and disables author batching to preserve that meaning). Remove both for an actual dataset.

## Stage 3: Abstract recovery, validation, and persistent state

For each unique OpenAlex work:

1. Read the decoded OpenAlex abstract text and validate minimum word count/obvious garbage. Legacy raw files containing `abstract_inverted_index` remain supported.
2. If missing or invalid and DOI exists, attempt **Crossref** DOI metadata.
3. If still missing, attempt **Semantic Scholar** DOI lookup; verify title.
4. If still missing, attempt **Europe PMC** by PMID, PMCID, or DOI.
5. If still missing, inspect configured maximum number of OpenAlex `best_oa_location`, `primary_location`, other `locations`, and DOI landing pages for a publicly displayed abstract in scholarly metadata or an explicit abstract section. Check robots.txt and do not fetch login-restricted content or PDFs.
6. Record `found` or `missing`, and retain the source and URL. If all fail, the **paper still contributes authorship information** but cannot contribute an abstract embedding.

Queries stop immediately after a validated abstract. The durable `output/state/abstract_results.jsonl` cache also remembers **missing** results, so rerunning the stage doesn't re-query them unless you pass `--retry-failed-abstracts`. HTTP responses are cached across stages in `.cache/http` (omit `--refresh-http` on normal runs). Note: on `--retry-failed-abstracts`, also use `--refresh-http` if the old HTTP metadata may have changed. Cached misses and transients are treated alike at present; you can refine retry policy as needed.

Set `settings.abstract_fallbacks: false` to use only decoded OpenAlex abstracts and skip Crossref, Semantic Scholar, Europe PMC, and landing-page requests. It defaults to `true` for backward compatibility. OpenAlex-only mode ignores previously cached fallback-derived abstracts; if fallbacks are enabled again later, pass `--retry-failed-abstracts` to revisit cached missing results.

**Abstract validation is heuristic.** Minimum length and text checks do not guarantee semantic correctness, and some publishers label summaries as abstracts. The webpage fallback deliberately avoids generic SEO descriptions. Each recovered abstract carries source provenance. Provider access controls, robots rules and occasional missing DOI metadata reduce coverage. The script does not synthesize missing abstracts.

Output: `output/papers.jsonl` (unique paper ID, DOI, title, year, all author IDs, abstract or `null`, status/source/source URL). Raw OpenAlex records remain separately available.

## Stages 4–5: Export data and collaboration networks

Outputs:

| File | Content |
|---|---|
| `output/graph/authorships.csv` | Faculty-only authorship edges linking each paper to retained faculty nodes, byline order and publication affiliation IDs |
| `output/graph/nodes.csv` | Selected faculty authors observed on collected papers |
| `output/graph/coauthor_edges.csv` | Pairwise faculty projection; weight = number of shared papers |
| `output/graph/faculty_edges.csv` | Subgraph of pairwise collaborations where **both endpoints** match a listed faculty member |
| `output/graph/skipped_large_teams.csv` | Papers not projected into pairwise edges because of large author-team size |
| `output/faculty_coverage.csv` | Per-faculty total papers, validated abstracts, coverage fraction, source breakdown |
| `output/summary.json` | Dataset sizes and omissions |

Coauthor edges have `author_a`, `author_b`, `weight`, `last_year`, `paper_ids` (JSON array stored in a CSV field). All join back through short OpenAlex IDs (`A...`, `W...`). Only selected faculty authorships are stored, so students and outside coauthors never become nodes. Papers with one selected faculty author remain part of that professor's publication history but create no pairwise edge.

Set `settings.graphml: true` to additionally export NetworkX GraphML graphs (can consume substantial memory for larger datasets). The paper–author relationship is the primary representation: a multi-authored paper is a hyperedge, and a pairwise projection is a modeling decision.

**Important graph completeness limitation:** These are the papers and pairwise relationships of the *selected faculty*. Nonfaculty intermediaries and outside coauthors are deliberately discarded. Do not interpret missing paths as definitive structural holes in the broader research community; this export supports the faculty-only recommendation graph.

## Reproducibility and failure handling

- Stable `A...`, `W...`, `I...` OpenAlex identifiers; deduplication across faculty and paper lists.
- Cursor pagination for institution author search and grouped faculty-work queries.
- Conservative identity resolution with explicit `not_found`, `ambiguous_name`, and `orcid_conflict` statuses.
- Rate-paced HTTP GET requests, retry/backoff on 429/5xx, 35-second timeouts, disk cache, and safe re-runs.
- Never commit credentials: use environment variables and keep `.env` out of git.
- `--refresh-http` disables HTTP cache for **that run**, and `--retry-failed-abstracts` re-attempts stored missing records.
- `pytest -q` runs deterministic offline fixtures for directory parsing, matching, paper deduplication, fallback logic, and graph exports.

### External documentation

- [OpenAlex API](https://help.openalex.org/api/), [filtering](https://help.openalex.org/api/filtering/), [cursor pagination](https://help.openalex.org/api/paging/), [authorship truncation](https://help.openalex.org/data/authorships/), [work attributes](https://help.openalex.org/data/works/attributes/)
- [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
- [Semantic Scholar API](https://api.semanticscholar.org/api-docs/)
- [Europe PMC API](https://www.ebi.ac.uk/europepmc/webservices/rest/search)
