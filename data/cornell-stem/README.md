# Cornell faculty network dataset

Preliminary research snapshot generated on November 1, 2026 from the configured
public Cornell faculty directories and OpenAlex metadata. Despite the historical
folder name, this snapshot includes every configured source: Bowers CIS, CALS,
Cornell Engineering, and Cornell Arts & Sciences.

## Snapshot summary

- 2,314 canonical Cornell faculty nodes
- 1,793 faculty nodes matched to OpenAlex
- 2,566 OpenAlex author profiles coalesced under those faculty nodes
- 521 unmatched roster faculty retained as isolates
- 59,887 unique works
- 41,805 works with a validated OpenAlex abstract
- 18,082 works without an abstract
- 68,358 unique faculty-to-paper authorships
- 3,815 faculty-to-faculty collaboration edges

The exact pipeline settings are preserved in `config.yaml`. Publication
collection was capped at 40 works per canonical faculty node. Abstract fallbacks
were disabled, so every stored abstract comes directly from OpenAlex. Papers
without abstracts remain in the authorship and collaboration data.

## Identity model

The official Cornell directory roster defines canonical people. Graph node IDs
are stable `faculty_id` values, not OpenAlex author IDs. All institution-affiliated
OpenAlex profiles matching an official roster name are treated as aliases of that
faculty node. Their works are unioned and deduplicated. Official roster-name
variants that share an OpenAlex profile (for example, names with and without a
middle initial) are also merged.

Each canonical node retains its contributing `openalex_ids`, `roster_names`,
departments, and profile URLs for auditability. Multiple aliases of one faculty
member appearing on the same paper count as one authorship and cannot create a
self-edge.

## Files

- `config.yaml`: exact pipeline configuration used for this snapshot.
- `summary.json`: machine-readable dataset counts.
- `faculty.csv`: canonical roster nodes, roster aliases, OpenAlex aliases, and match status.
- `match_diagnostics.csv`: roster names, match status, candidate IDs, and profile URLs.
- `faculty_coverage.csv`: publication and abstract coverage by canonical faculty node.
- `papers.jsonl.gz`: normalized work records, canonical faculty IDs, contributing
  OpenAlex author IDs, and decoded OpenAlex abstracts.
- `graph/nodes.csv`: canonical faculty graph nodes, including unmatched isolates.
- `graph/authorships.csv.gz`: paper-to-canonical-faculty incidence records, with
  contributing OpenAlex aliases and publication-time affiliations.
- `graph/faculty_edges.csv`: undirected faculty projection. `weight` is the number
  of distinct shared papers; `last_year` is the most recent shared-paper year;
  `paper_ids` is a JSON array stored in the CSV field.
- `graph/skipped_large_teams.csv`: works omitted from pairwise projection because
  of the configured clique-size limit. This snapshot contains none.

To stream the compressed records without creating permanent uncompressed copies:

```bash
gzip -cd data/cornell-stem/papers.jsonl.gz
gzip -cd data/cornell-stem/graph/authorships.csv.gz
```

Python's `gzip`, `json`, and `csv` modules or equivalent dataframe readers can
read these files directly.

## Known limitations

This is a permissive, high-recall identity snapshot. Coalescing split OpenAlex
profiles prevents one professor from appearing as multiple graph nodes, but two
different Cornell-affiliated people with the same name may occasionally be
merged. Their publications and connections would then be combined.

The 521 roster nodes without a matching OpenAlex profile have no papers or graph
edges. Only listed Cornell faculty become nodes; students, outside collaborators,
and other non-roster coauthors are discarded. Network paths therefore describe
the retained Cornell faculty network, not the complete scholarly collaboration
network.

The dataset uses a per-faculty work cap rather than complete career histories.
Analyses should document this sampling rule and should choose an appropriate
time window and weighting scheme before drawing substantive conclusions.
