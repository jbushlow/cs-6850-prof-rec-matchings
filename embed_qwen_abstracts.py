#!/usr/bin/env python3
"""Cache Qwen3 retrieval embeddings that mirror sherlocking.py score_papers().

Every unique paper gets one *document* vector for "title. abstract" (or
just title when abstract is absent). Every abstract sentence gets a separate
*passage* vector; a title-only paper gets its title as one passage.

The independent shard sets can be resumed without recomputing completed shards.
The index files preserve paper IDs, authors, passage sources, hashes, and
embedding locations. This script does not modify sherlocking.py or make it
consume the cache; using the cache in recommendations needs a reader change.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

# ============================ SETTINGS ============================
BASE_DIR = Path("/home/jiasheng/CS6850/prof_rec_colab")
MODEL = "Qwen/Qwen3-Embedding-0.6B"  # or 4B / 8B
BATCH_SIZE = 8
SHARD_SIZE = 2048
MAX_SEQ_LENGTH = 2048
DTYPE = "float16"                 # float16 or float32
DEVICE = "auto"                   # auto, cuda or cpu
LIMIT = None                      # None = all distinct papers
OVERWRITE = True                 # Only affects the NEW cache directory
# =================================================================


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def split_sentences(abstract: str) -> list[str]:
    """Exactly the same segmentation as sherlocking.py."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|;\s*", abstract.strip()) if s.strip()]


def load_papers(base: Path, limit: int | None):
    from faculty_data import load_faculty_files

    paths = [base / "data/nodes.csv", base / "data/authorships.csv",
             base / "data/papers.jsonl.gz", base / "data/faculty_edges.csv"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing input files:\n  " + "\n  ".join(missing))

    # The crucial difference from the abstract-only script:
    data = load_faculty_files(*paths, missing_abstract="title")
    by_id = {}
    refs = 0
    for professor in data["professors"]:
        professor_id = str(professor["id"])
        for paper in professor["papers"]:
            pid = str(paper["id"])
            title = str(paper.get("title") or "").strip()
            abstract = str(paper.get("abstract") or "").strip()
            if not title and not abstract:
                continue
            refs += 1
            if pid in by_id:
                prev = by_id[pid]
                if (prev["title"], prev["abstract"]) != (title, abstract):
                    raise ValueError(f"Conflicting title/abstract for shared paper ID {pid!r}")
                prev["author_ids"].add(professor_id)
            else:
                by_id[pid] = {
                    "paper_id": pid, "title": title, "abstract": abstract,
                    "year": paper.get("year"), "author_ids": {professor_id},
                }

    papers = sorted(by_id.values(), key=lambda row: row["paper_id"])
    if limit is not None:
        papers = papers[:limit]
    if not papers:
        raise SystemExit("No papers with a title or abstract found")
    for paper in papers:
        paper["author_ids"] = sorted(paper["author_ids"])
    summary = {
        "source_files": [str(p) for p in paths],
        "faculty_count": len(data["professors"]),
        "paper_faculty_authorship_references_before_limit": refs,
        "paper_count": len(papers),
        "papers_with_abstract": sum(bool(p["abstract"]) for p in papers),
        "title_only_papers": sum(bool(p["title"]) and not p["abstract"] for p in papers),
        "abstract_only_papers": sum(bool(p["abstract"]) and not p["title"] for p in papers),
    }
    return papers, summary


def build_entries(papers):
    documents, passages = [], []
    for paper in papers:
        pid, title, abstract = (paper[k] for k in ("paper_id", "title", "abstract"))
        # EXACTLY matches sherlocking.py's document construction.
        document_text = f"{title}. {abstract}".strip(" .")
        source = "title_and_abstract" if title and abstract else ("abstract" if abstract else "title")
        documents.append({
            "entry_id": pid, "paper_id": pid, "text": document_text,
            "text_sha256": sha256_text(document_text), "text_source": source,
        })
        # Sherlocking embeds only abstract sentences when an abstract exists.
        passage_texts = split_sentences(abstract) if abstract else [title]
        for idx, text in enumerate(passage_texts):
            passages.append({
                "entry_id": f"{pid}:passage:{idx:05d}", "paper_id": pid,
                "passage_number": idx, "text": text,
                "text_sha256": sha256_text(text),
                "text_source": "abstract" if abstract else "title",
            })
    return documents, passages


def fingerprint(documents, passages):
    digest = hashlib.sha256()
    for kind, rows in (("documents", documents), ("passages", passages)):
        for row in rows:
            entry = [kind, row["entry_id"], row["text_sha256"]]
            digest.update(json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def location(kind: str, index: int):
    part, row = divmod(index, SHARD_SIZE)
    return {"shard": f"{kind}/shards/part_{part:05d}.pt", "row": row}


def check_shards(root: Path, kind: str, entries: list[dict], torch):
    count = (len(entries) + SHARD_SIZE - 1) // SHARD_SIZE
    missing = []
    for part in range(count):
        chunk = entries[part * SHARD_SIZE:(part + 1) * SHARD_SIZE]
        path = root / kind / "shards" / f"part_{part:05d}.pt"
        if not path.exists():
            missing.append(part)
            continue
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            vectors = saved.get("embeddings")
            valid = (
                saved.get("entry_ids") == [r["entry_id"] for r in chunk]
                and saved.get("text_sha256") == [r["text_sha256"] for r in chunk]
                and isinstance(vectors, torch.Tensor)
                and vectors.ndim == 2
                and vectors.shape[0] == len(chunk)
                and vectors.shape[1] > 0
                and vectors.dtype == (torch.float16 if DTYPE == "float16" else torch.float32)
            )
        except Exception as exc:
            raise SystemExit(f"Cannot read cache shard {path}: {exc}. Use OVERWRITE=True to rebuild.") from exc
        if not valid:
            raise SystemExit(f"Cache shard {path} mismatches inputs. Use OVERWRITE=True to rebuild.")
        print(f"[{kind} {part+1}/{count}] Reusing {len(chunk)} embeddings", flush=True)
    return missing, count



def check_token_lengths(chunk, tokenizer, kind, stats, prompt=""):
    texts = [prompt + row["text"] for row in chunk]

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

    lengths = [len(ids) for ids in encoded["input_ids"]]
    hit = sum(n > MAX_SEQ_LENGTH for n in lengths)
    near = sum(n >= MAX_SEQ_LENGTH * 0.9 for n in lengths)

    stats[kind]["total"] += len(lengths)
    stats[kind]["hit"] += hit
    stats[kind]["near"] += near
    stats[kind]["max"] = max(stats[kind]["max"], max(lengths, default=0))

    s = stats[kind]
    print(
        f"\n[TOKEN CHECK: {kind}] "
        f"Current shard: {hit}/{len(lengths)} truncated | "
        f"Overall: {s['hit']}/{s['total']} "
        f"({100 * s['hit'] / s['total']:.2f}%) | "
        f"Near limit: {s['near']} | "
        f"Longest: {s['max']} tokens",
        flush=True,
    )



def save_missing(root, kind, entries, parts, count, model, torch, np, stats):
    for part in parts:
        chunk = entries[part * SHARD_SIZE:(part + 1) * SHARD_SIZE]
        print(f"[{kind} {part+1}/{count}] Embedding {len(chunk)} texts...", flush=True)
        check_token_lengths(
            chunk,
            model.tokenizer,
            kind,
            stats,
            prompt=next(
                (model.prompts[name]
                 for name in ("document", "passage", "corpus")
                 if name in model.prompts),
                "",
            ),
        ) 
        
        with torch.inference_mode():
            result = model.encode_document(
                [r["text"] for r in chunk], batch_size=BATCH_SIZE,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
            )
        arr = np.asarray(result, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(chunk) or not np.isfinite(arr).all():
            raise RuntimeError(f"Invalid model output for {kind} shard {part}")
        emb = torch.from_numpy(arr.copy()).to(torch.float16 if DTYPE == "float16" else torch.float32)
        payload = {
            "entry_ids": [r["entry_id"] for r in chunk],
            "text_sha256": [r["text_sha256"] for r in chunk],
            "embeddings": emb.cpu().contiguous(),
        }
        path = root / kind / "shards" / f"part_{part:05d}.pt"
        temp = path.with_suffix(".pt.tmp")
        torch.save(payload, temp)
        os.replace(temp, path)
        print(f"    Saved {path}", flush=True)


def write_indexes(root, papers, documents, passages):
    paper_lookup = {p["paper_id"]: {
        "title": p["title"], "year": p["year"],
        "author_ids": p["author_ids"], "has_abstract": bool(p["abstract"]),
        "document": None, "passages": [],
    } for p in papers}
    document_rows = []
    for i, doc in enumerate(documents):
        loc = location("documents", i)
        paper = paper_lookup[doc["paper_id"]]
        paper["document"] = loc
        document_rows.append({**doc, **loc, "title": paper["title"],
                              "year": paper["year"], "author_ids": paper["author_ids"]})

    passage_rows = []
    for i, passage in enumerate(passages):
        loc = location("passages", i)
        paper_lookup[passage["paper_id"]]["passages"].append({
            "entry_id": passage["entry_id"], "text_source": passage["text_source"],
            **loc,
        })
        passage_rows.append({**passage, **loc})

    write_jsonl_atomic(root / "documents" / "index.jsonl", document_rows)
    write_jsonl_atomic(root / "passages" / "index.jsonl", passage_rows)
    write_json_atomic(root / "paper_lookup.json", paper_lookup)
    # Direct ID -> embedding location maps, useful for random access.
    write_json_atomic(root / "documents" / "lookup.json", {
        d["entry_id"]: location("documents", i) for i, d in enumerate(documents)
    })
    write_json_atomic(root / "passages" / "lookup.json", {
        p["entry_id"]: location("passages", i) for i, p in enumerate(passages)
    })


def run():
    import numpy as np
    import torch

    if not re.fullmatch(r"Qwen/Qwen3-Embedding-(0\.6B|4B|8B)", MODEL):
        raise SystemExit("MODEL must be Qwen/Qwen3-Embedding-0.6B, 4B, or 8B")
    if BATCH_SIZE <= 0 or SHARD_SIZE <= 0 or MAX_SEQ_LENGTH <= 0:
        raise SystemExit("Batch, shard and max sequence lengths must be positive")
    if LIMIT is not None and LIMIT <= 0:
        raise SystemExit("LIMIT must be None or positive")
    if DTYPE not in ("float16", "float32") or DEVICE not in ("auto", "cuda", "cpu"):
        raise SystemExit("Invalid DTYPE or DEVICE")

    base = BASE_DIR.expanduser().resolve()
    cache_name = "paper_retrieval_v1" if LIMIT is None else f"paper_retrieval_v1_sample_{LIMIT}"
    root = base / "embedding_cache" / MODEL.rsplit("/", 1)[-1] / cache_name
    if OVERWRITE and root.exists():
        shutil.rmtree(root)
    for kind in ("documents", "passages"):
        (root / kind / "shards").mkdir(parents=True, exist_ok=True)

    print("Loading papers with missing_abstract='title'...", flush=True)
    papers, summary = load_papers(base, LIMIT)
    documents, passages = build_entries(papers)
    source_fingerprint = fingerprint(documents, passages)
    config = {
        "format_version": 1, "model": MODEL, "role": "document",
        "scoring_compatibility": "sherlocking.py score_papers: full document + best abstract sentence/title passage",
        "document_text": "title + '. ' + abstract, stripped of spaces/dots; title-only fallback",
        "passage_text": "split abstract into sentences; title-only fallback",
        "normalized": True, "dtype_on_disk": DTYPE,
        "max_seq_length": MAX_SEQ_LENGTH, "shard_size": SHARD_SIZE,
        "source_fingerprint_sha256": source_fingerprint,
        "document_count": len(documents), "passage_count": len(passages),
        **summary,
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            # Authorship metadata may legitimately change without altering embeddings.
            if key in ("source_files", "faculty_count", "paper_faculty_authorship_references_before_limit"):
                continue
            if prev.get(key) != value:
                raise SystemExit(f"Existing cache {key} differs. Choose OVERWRITE=True to rebuild "
                                 f"or select a new cache directory. Existing cache was not changed.")
        created = prev["created_utc"]
    else:
        existing = list(root.glob("*/shards/*.pt"))
        if existing:
            raise SystemExit("Shards exist without manifest. Use OVERWRITE=True after inspecting them.")
        created = datetime.now(timezone.utc).isoformat()

    device = DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {**config, "created_utc": created,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "device_last_used": device, "status": "in_progress"}
    write_json_atomic(manifest_path, manifest)
    doc_missing, nd = check_shards(root, "documents", documents, torch)
    passage_missing, npa = check_shards(root, "passages", passages, torch)
    if doc_missing or passage_missing:
        from sentence_transformers import SentenceTransformer
        weight_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
                       if device == "cuda" else torch.float32
        print(f"Loading {MODEL} on {device}...", flush=True)
        model = SentenceTransformer(MODEL, device=device,
                                    model_kwargs={"torch_dtype": weight_dtype})
        model.max_seq_length = MAX_SEQ_LENGTH
        model.eval()
        if not hasattr(model, "encode_document"):
            raise SystemExit("Update sentence-transformers: encode_document() is required")
        
        stats = {
            "documents": {"total": 0, "hit": 0, "near": 0, "max": 0},
            "passages": {"total": 0, "hit": 0, "near": 0, "max": 0},
        }
 
        save_missing(root, "documents", documents, doc_missing, nd, model, torch, np, stats)
        save_missing(root, "passages", passages, passage_missing, npa, model, torch, np, stats)

    write_indexes(root, papers, documents, passages)
    example = torch.load(root / "documents" / "shards" / "part_00000.pt",
                         map_location="cpu", weights_only=True)
    manifest.update({"status": "complete", "document_shard_count": nd,
                     "passage_shard_count": npa,
                     "embedding_dimension": int(example["embeddings"].shape[1]),
                     "updated_utc": datetime.now(timezone.utc).isoformat()})
    write_json_atomic(manifest_path, manifest)
    print(f"DONE: {len(papers)} papers, {len(documents)} document embeddings, "
          f"{len(passages)} passage embeddings; cache: {root}")
    return root


if __name__ == "__main__":
    run()
