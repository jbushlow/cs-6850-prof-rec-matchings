# Cornell Bowers faculty network dataset (v0)

Preliminary research snapshot generated on November 1, 2026 from the public
Cornell Bowers CIS faculty directory and OpenAlex metadata.

## Snapshot summary

- 136 Bowers directory entries
- 87 faculty matched to distinct OpenAlex author IDs
- 11,765 unique works
- 8,624 works with a validated OpenAlex abstract
- 3,141 works without an abstract
- 12,387 retained faculty-to-paper authorships
- 135 faculty-to-faculty coauthorship edges

The exact pipeline settings are preserved in `config.yaml`. Abstract fallbacks
were disabled for this snapshot, so every stored abstract comes directly from
OpenAlex. Papers without abstracts remain in the authorship and coauthorship
data.

## Files

- `faculty.csv`: Bowers roster and conservative OpenAlex match status.
- `match_diagnostics.csv`: match status and candidate OpenAlex IDs.
- `faculty_coverage.csv`: publication and abstract coverage by matched faculty.
- `papers.jsonl.gz`: normalized work records and decoded OpenAlex abstracts.
- `raw/papers_openalex.jsonl.gz`: compact OpenAlex records used by the graph exporter.
- `graph/nodes.csv`: matched faculty nodes.
- `graph/authorships.csv`: faculty-to-paper incidence records.
- `graph/faculty_edges.csv`: faculty-only projection; `weight` is shared-paper count.
- `graph/skipped_large_teams.csv`: works omitted from projection due to clique limits.
- `summary.json`: machine-readable dataset counts.

To stream the compressed paper records without creating an uncompressed copy:

```bash
gzip -cd data/bowers_v0/papers.jsonl.gz
```

## Known limitations

This is a development snapshot, not a final identity-resolved dataset. Of the
136 directory entries, 37 had ambiguous OpenAlex name matches and 12 were not
found. Even unique-name matches can be wrong or can point to merged OpenAlex
profiles; manual identity review is still required. One known suspicious case
is Melissa Laird Smith, whose matched profile contains implausibly old biomedical
publications.

Only matched Bowers faculty are retained as author nodes. Students, outside
collaborators, and other non-roster authors are deliberately excluded. Network
paths therefore describe the retained Bowers faculty network, not the complete
scholarly collaboration network.

The publication history spans 1951–2026 and is not time-filtered. Analyses should
choose and document an appropriate time window and weighting scheme before
drawing substantive conclusions.
