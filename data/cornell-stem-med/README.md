# Cornell STEM and medical research faculty network dataset

Preliminary research snapshot generated on November 2, 2026 from the configured
public Cornell faculty directories and OpenAlex metadata. It includes Bowers
CIS, CALS, Cornell Engineering, Cornell Arts & Sciences, and the Weill Cornell
Graduate School research-faculty directory. It does not include the broader
Weill Cornell clinical-provider roster.

## Snapshot summary

- 2,836 canonical Cornell faculty nodes
- 2,236 faculty nodes matched to OpenAlex
- 3,428 OpenAlex author profiles coalesced under those faculty nodes
- 600 unmatched roster faculty retained as isolates
- 74,667 unique works
- 51,700 works with a validated OpenAlex abstract
- 22,967 works without an abstract
- 88,808 unique faculty-to-paper authorships
- 6,953 faculty-to-faculty collaboration edges

The exact pipeline settings are preserved in `config.yaml`. Publication
collection was capped at 40 works per canonical faculty node. Abstract fallbacks
were disabled, so every stored abstract comes directly from OpenAlex. Papers
without abstracts remain in the authorship and collaboration data.

## Identity model

The official Cornell directory roster defines canonical people. Graph node IDs
are stable `faculty_id` values, not OpenAlex author IDs. Institution-affiliated
OpenAlex profiles matching an official roster name are treated as aliases of
that faculty node. Their works are unioned and deduplicated.

Official roster variants can merge when they share an OpenAlex profile and have
the same first complete given name and surname. Middle names and initials are
ignored for this comparison, and a leading initial does not replace a later
complete given name. For example, `Jon Kleinberg` and `Jon M. Kleinberg` can
merge, while `Y. Samuel Wang`, `Yi Wang`, and `Yu Wang` remain separate. A noisy
OpenAlex profile cannot bridge incompatible roster identities; its primary
display name selects the compatible person when possible.

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
gzip -cd data/cornell-stem-med/papers.jsonl.gz
gzip -cd data/cornell-stem-med/graph/authorships.csv.gz
```

Python's `gzip`, `json`, and `csv` modules or equivalent dataframe readers can
read these files directly.

## Known limitations

This is a permissive, high-recall identity snapshot. Coalescing split OpenAlex
profiles prevents one professor from appearing as multiple graph nodes, but two
different Cornell-affiliated people with exactly the same name may occasionally
be merged. Their publications and connections would then be combined.

The Weill Cornell source is a graduate and research faculty directory, not a
complete medical-college or clinical-provider roster. Some clinically active
researchers may therefore be absent, while the graduate directory may include
affiliated researchers whose primary institution is outside Cornell. Only
faculty matching a configured Cornell OpenAlex affiliation receive papers.

The 600 roster nodes without a matching OpenAlex profile have no papers or graph
edges. Only listed Cornell faculty become nodes; students, outside collaborators,
and other non-roster coauthors are discarded. Network paths therefore describe
the retained Cornell faculty network, not the complete scholarly collaboration
network.

The dataset uses a per-faculty work cap rather than complete career histories.
Analyses should document this sampling rule and choose an appropriate time window
and weighting scheme before drawing substantive conclusions.
