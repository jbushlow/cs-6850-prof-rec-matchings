#!/usr/bin/env python3
"""Sherlocking: paper evidence + requirement-relevant structural-hole brokerage.

Scientific: F_q(T) = sum_j w_j max_{p in {initiator} union T} S[p,j].
Brokerage: new team edges initiator--broker create open wedges to observed
           coauthors with evidence for complementary project requirements.
           Per wedge the harmonic intermediary payoff is 2*g_q(scientist)/r.
Total:      J_q(T) = F_q(T) + lambda_network * B_q,u(T).

Qwen3-4B generatively decomposes each input project abstract into research
requirements. Qwen3-Embedding then does retrieval; observed coauthorships plus
precomputed NCUT paper clusters determine network brokerage.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
import re
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np

from faculty_data import (apply_missing_abstract_policy, load_faculty_files,
                          MISSING_ABSTRACT_CHOICES)

# ============================================================================
# SETTINGS -- EDIT THESE VARIABLES, THEN RUN:  python sherlocking.py
# Paths are relative to this script, so it runs from any working directory.
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

# What should this script do? "recommend" is the normal pipeline.
# Other options: "inspect-data", "decompose", "evolve".
ACTION = "recommend"

# Real faculty and publication data (the default inputs).
NODES_CSV = BASE_DIR / "data/nodes.csv"
AUTHORSHIPS_CSV = BASE_DIR / "data/authorships.csv"
PAPERS_JSONL_GZ = BASE_DIR / "data/papers.jsonl.gz"
FACULTY_EDGES_CSV = BASE_DIR / "data/faculty_edges.csv"
QUERY_JSON = BASE_DIR / "examples/proposal_real_faculty2.json"
OUTPUT_DIR = BASE_DIR / "output_real_faculty"

# How to handle a paper with no abstract: "title", "skip", or "error".
# "title" means score the paper using its title alone, when available.
MISSING_ABSTRACT = "title"

# Embedding model and team-selection parameters.
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_GENERATIVE_MODEL = "Qwen/Qwen3-4B"  # Separate instruct/generative model
MAX_ADDITIONAL = 3
LAMBDA_NETWORK = 0.25
# Component 2 uses the previously computed NCUT paper clusters directly.
# The number of clusters is read from the saved NCUT manifest; there are no
# per-requirement cluster-count, similarity-cutoff, or saturation parameters.
# Use the output of embed_qwen_abstracts(2).py when available. When missing,
# Qwen embeds texts normally as before. An existing mismatched cache raises.
USE_EMBEDDING_CACHE = True
EMBEDDING_CACHE_DIR = (BASE_DIR / "embedding_cache" /
                       QWEN_MODEL.rsplit("/", 1)[-1] / "paper_retrieval_v1")
# Required: existing offline NCUT output, strictly checked against paper IDs,
# text hashes and model. Run cluster_ncut_abstracts.py once beforehand.
PRECOMPUTED_TOPIC_CLUSTERS_DIR = EMBEDDING_CACHE_DIR / "topic_clusters_ncut_v1"
THRESHOLD_LOW = None        # None for automatic Qwen3 cosine calibration
THRESHOLD_HIGH = None       # Must set BOTH bounds to override calibration
EVIDENCE_LABEL_THRESHOLD = None  # None means the default 0.65
RESTARTS = 30
TOP_R = 4
MAX_ROUNDS = 12
SCIENCE_TOLERANCE = 0.03
N_TEAMS = 5
EXACT_CHECK_MAX_CANDIDATES = 0
SEED = 42

# Optional/legacy workflows: normally leave these alone.
# If True, recommendations use LEGACY_DATA_JSON instead of the four CSV/GZ files.
USE_LEGACY_JSON = False
LEGACY_DATA_JSON = BASE_DIR / "examples/professors_v1.json"
# Recommendation ALWAYS regenerates requirements from the input abstract with
# QWEN_GENERATIVE_MODEL. No heuristic, manual-merge, or silent fallback.
EVOLVE_UPDATE_JSON = BASE_DIR / "examples/observed_update.json"
UTILITY_OUTPUT_JSON = BASE_DIR / "output_utility.json"
# ============================================================================

EPS = 1e-10

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def split_sentences(abstract):
    """Transparent sentence segmentation, not a scientific claim extractor."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|;\s*", abstract.strip()) if s.strip()]


def validate_dataset(data):
    professors = data.get("professors")
    if not isinstance(professors, list) or not professors:
        raise ValueError("professors must be a nonempty list")
    ids = set()
    paper_texts = {}
    for p in professors:
        pid = p.get("id")
        if not isinstance(pid, str) or not pid.strip() or pid in ids:
            raise ValueError("professor ids must be unique nonempty strings")
        ids.add(pid)
        if not isinstance(p.get("name"), str) or not p["name"].strip():
            raise ValueError(f"{pid}: missing name")
        papers = p.get("papers")
        if not isinstance(papers, list):
            raise ValueError(f"{pid}: papers must be a list")
        local_ids = set()
        for paper in papers:
            paper_id = paper.get("id")
            if not isinstance(paper_id, str) or not paper_id.strip() or paper_id in local_ids:
                raise ValueError(f"{pid}: paper ids must be unique nonempty strings")
            local_ids.add(paper_id)
            title = str(paper.get("title") or "").strip()
            abstract = str(paper.get("abstract") or "").strip()
            if not title and not abstract:
                raise ValueError(f"{pid}/{paper_id}: paper needs an abstract or a title")
            signature = (title, abstract)
            if paper_id in paper_texts and signature != paper_texts[paper_id]:
                raise ValueError(f"Conflicting metadata for shared paper {paper_id}")
            paper_texts[paper_id] = signature
    for edge in data.get("coauthorship_edges", []):
        if edge.get("source") not in ids or edge.get("target") not in ids or edge["source"] == edge["target"]:
            raise ValueError(f"Invalid observed coauthorship edge: {edge}")
    return ids


def resolve_requirements(query, ids):
    if query.get("initiator_id") not in ids:
        raise ValueError("initiator_id must be one of the professor ids")
    requirements = query.get("requirements")
    if requirements is None or requirements == []:
        raise ValueError("No requirements: generative Qwen decomposition must run first")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("requirements must be a nonempty list")
    clean = []
    for i, r in enumerate(requirements):
        if not isinstance(r, dict) or not isinstance(r.get("description"), str) or not r["description"].strip():
            raise ValueError(f"Requirement {i}: description must be nonempty text")
        weight = float(r.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("requirement weights must be finite and nonnegative")
        clean.append({"id": f"req_{i+1}", "description": r["description"].strip(), "weight": weight})
    weight_sum = sum(r["weight"] for r in clean)
    if weight_sum <= 0:
        raise ValueError("At least one requirement needs positive weight")
    for r in clean:
        r["weight"] /= weight_sum
    return clean


def build_heterogeneous_graph(data):
    """Professor + paper nodes and authorship edges; observed coauthorship only."""
    graph = nx.Graph()
    for p in data["professors"]:
        graph.add_node(f"prof:{p['id']}", node_type="professor", id=p["id"],
                       name=p["name"], department=p.get("department", ""))
        for d in p["papers"]:
            key = f"paper:{d['id']}"
            if key not in graph:
                graph.add_node(key, node_type="paper", id=d["id"],
                               title=d.get("title", ""), abstract=d.get("abstract", ""),
                               text_source=d.get("text_source", "abstract" if d.get("abstract") else "title"),
                               year=d.get("year"), doi=d.get("doi"),
                               publication_url=d.get("publication_url"))
            graph.add_edge(f"prof:{p['id']}", key, edge_type="authorship",
                           **d.get("authorship_metadata", {}))
    for e in data.get("coauthorship_edges", []):
        graph.add_edge(f"prof:{e['source']}", f"prof:{e['target']}",
                       edge_type="observed_coauthorship",
                       **{k: e[k] for k in ("weight", "last_year", "paper_ids") if k in e})
    return graph


def make_collaboration_graph(data):
    """Only professors + observed coauthorship; no paper nodes/edges."""
    g = nx.Graph()
    g.add_nodes_from(p["id"] for p in data["professors"])
    g.add_edges_from((e["source"], e["target"],
                      {k: e[k] for k in ("weight", "last_year", "paper_ids") if k in e})
                     for e in data.get("coauthorship_edges", []))
    return g


def _dense_similarity(a, b):
    sim = a @ b.T
    return np.asarray(sim.toarray() if hasattr(sim, "toarray") else sim, dtype=np.float64)


class TextEncoder:
    """Qwen3 text embeddings: instructed project/requirements as queries,
    paper titles/abstracts and evidence sentences as retrieval documents.

    No TF-IDF or implicit fallback; downloading Qwen model weights may be needed.
    Generative Qwen3 requirement decomposition runs separately before retrieval.
    """
    def __init__(self, backend="qwen3", model="Qwen/Qwen3-Embedding-0.6B", batch_size=16):
        if backend != "qwen3":
            raise ValueError("Only the Qwen3 text embedding backend is supported")
        if not model.startswith("Qwen/Qwen3-Embedding-"):
            raise ValueError("QWEN_MODEL must be a Qwen3 text embedding checkpoint, e.g. Qwen/Qwen3-Embedding-0.6B")
        self.backend, self.model, self.batch_size = backend, model, batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install the Qwen3 embedding requirements: pip install -r requirements.txt") from exc
            try:
                self._model = SentenceTransformer(self.model)
            except Exception as exc:
                raise RuntimeError(f"Could not load {self.model}; Qwen3 weights must be accessible locally or downloaded") from exc
        return self._model

    def _embed(self, texts, role):
        model = self._load()
        method = model.encode_query if role == "query" else model.encode_document
        vectors = np.asarray(method(texts, normalize_embeddings=True,
                                    batch_size=self.batch_size, show_progress_bar=False), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Qwen3 embedding output shape mismatch")
        return vectors

    def encode_queries(self, texts):
        return self._embed(texts, "query")

    def encode_documents(self, texts):
        return self._embed(texts, "document")


def evidence_type(project_cosine, requirement_sentence_cosine, score, *,
                  strong_project_cutoff, weak_project_cutoff,
                  strong_sentence_cutoff, score_threshold=0.65):
    """Heuristic descriptive label, never proof of scientific transfer.

    A strong requirement-specific paper match is necessary for either label.
    Project and sentence cutoffs come from Qwen3 cosine distributions.
    """
    if score < score_threshold:
        return "partial_or_weak_match"
    if project_cosine >= strong_project_cutoff:
        return "similarity_candidate"
    if project_cosine <= weak_project_cutoff and requirement_sentence_cosine >= strong_sentence_cutoff:
        return "complementary_candidate_needs_review"
    return "partial_or_weak_match"


def read_retrieval_embedding_cache(cache_root, model_name, paper_ids, full_texts,
                                   passage_info):
    """Read the exact document/passage cache written by embed_qwen_abstracts.

    Return None if it has not been generated; fail loudly if present but stale.
    Title-only papers and abstract sentences must agree on entry IDs AND text.
    """
    root = Path(cache_root)
    manifest_path = root / "manifest.json"
    if not root.exists():
        return None
    if not manifest_path.is_file():
        raise ValueError(f"Embedding cache exists without manifest: {root}")
    manifest = load_json(manifest_path)
    if (manifest.get("status") != "complete" or manifest.get("model") != model_name
            or manifest.get("format_version") != 1 or not manifest.get("normalized")
            or manifest.get("role") != "document"):
        raise ValueError(f"Incomplete or incompatible retrieval embedding cache: {root}")

    from collections import Counter
    passage_counts = Counter()
    passage_items = []
    for pid, text, _ in passage_info:
        i = passage_counts[pid]
        passage_counts[pid] += 1
        passage_items.append((f"{pid}:passage:{i:05d}", text))
    expected = {
        "documents": list(zip(paper_ids, full_texts)),
        "passages": passage_items,
    }
    import torch
    outputs = []
    for kind in ("documents", "passages"):
        index_path = root / kind / "index.jsonl"
        with index_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        items = expected[kind]
        if len(records) != len(items):
            raise ValueError(f"Cached {kind} count differs from current papers: {root}")
        current_shard_path = None
        saved = None
        arrays = []
        for rec, (eid, text) in zip(records, items):
            if (rec.get("entry_id") != eid or
                    rec.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest()):
                raise ValueError(f"Cached {kind} text/ID differs for {eid}; rebuild the cache")
            shard_path = rec["shard"]
            if shard_path != current_shard_path:
                saved = torch.load(root / shard_path, map_location="cpu", weights_only=True)
                current_shard_path = shard_path
            row = int(rec["row"])
            if (saved["entry_ids"][row] != eid or saved["text_sha256"][row] != rec["text_sha256"]):
                raise ValueError(f"Corrupt embedding cache shard {shard_path}")
            arrays.append(saved["embeddings"][row].to(torch.float32).numpy().copy())
        arr = np.stack(arrays)
        if not np.isfinite(arr).all() or arr.ndim != 2 or (np.linalg.norm(arr, axis=1) < EPS).any():
            raise ValueError(f"Invalid embeddings in {kind} cache")
        outputs.append(arr)
    if outputs[0].shape[1] != outputs[1].shape[1]:
        raise ValueError("Document/passage cache vector dimensions do not agree")
    return outputs[0], outputs[1]


def score_papers(data, requirements, encoder, low=None, high=None,
                  evidence_label_threshold=None, project_abstract=None,
                  cache_root=None, return_retrieval_vectors=False):
    """Qwen3 retrieval: z=.5 cosine(query, full paper) + .5 best passage cosine.

    By default, scores are RELATIVE to the query's paper index: calibrated
    with the 10th/90th percentiles of all requirement-paper z values.
    Unlike the original TF-IDF values, fixed .02/.45 bounds are NOT reused.
    low/high can explicitly override automatic percentile calibration.
    """
    if (low is None) != (high is None):
        raise ValueError("Set both THRESHOLD_LOW and THRESHOLD_HIGH, or neither")
    if low is not None and not (math.isfinite(low) and math.isfinite(high) and -1 <= low < high <= 1):
        raise ValueError("Require -1 <= low < high <= 1 for cosine calibration")
    if evidence_label_threshold is None:
        evidence_label_threshold = 0.65  # minimum *calibrated* evidence strength
    if not 0 <= evidence_label_threshold <= 1:
        raise ValueError("evidence label threshold must be within [0,1]")
    papers = {}
    authors = defaultdict(set)
    for p in data["professors"]:
        for d in p["papers"]:
            papers[d["id"]] = {**d, "title": str(d.get("title") or "").strip(),
                                "abstract": str(d.get("abstract") or "").strip()}
            authors[d["id"]].add(p["id"])
    paper_ids = sorted(papers)
    if not paper_ids:
        raise ValueError("No scorable papers. Try setting MISSING_ABSTRACT to 'title' or check the input files.")
    full_texts = [f"{papers[pid].get('title') or ''}. {papers[pid].get('abstract') or ''}".strip(" .")
                  for pid in paper_ids]
    # For title-only papers, use the title as BOTH the full document and the
    # single evidence passage, while labelling it as title (not abstract).
    passage_info = [(paper_id, sentence, "abstract")
                    for paper_id in paper_ids if papers[paper_id].get("abstract")
                    for sentence in split_sentences(papers[paper_id]["abstract"])]
    passage_info += [(paper_id, papers[paper_id]["title"], "title")
                     for paper_id in paper_ids if not papers[paper_id].get("abstract")]
    # Preserve deterministic paper order in the best-passage index.
    passage_info.sort(key=lambda row: row[0])
    req_texts = [r["description"] for r in requirements]
    project_text = str(project_abstract or " ".join(req_texts)).strip()
    query_vecs = encoder.encode_queries(req_texts + [project_text])
    cached = (read_retrieval_embedding_cache(cache_root, encoder.model,
                                             paper_ids, full_texts, passage_info)
              if cache_root is not None else None)
    if cached is None:
        document_vecs = encoder.encode_documents(full_texts + [s for _, s, _ in passage_info])
        full_vecs, sent_vecs = document_vecs[:len(paper_ids)], document_vecs[len(paper_ids):]
    else:
        full_vecs, sent_vecs = cached
    if (query_vecs.ndim != 2 or query_vecs.shape[0] != len(req_texts)+1
            or full_vecs.shape[0] != len(paper_ids) or sent_vecs.shape[0] != len(passage_info)
            or query_vecs.shape[1] != full_vecs.shape[1] or full_vecs.shape[1] != sent_vecs.shape[1]):
        raise ValueError("Encoder/cache output shape mismatch")
    nr, npapers = len(req_texts), len(paper_ids)
    req_vecs, project_vec = query_vecs[:nr], query_vecs[nr:nr+1]
    full_sim = np.clip(_dense_similarity(req_vecs, full_vecs), -1, 1)
    project_doc_sim = np.clip(_dense_similarity(project_vec, full_vecs)[0], -1, 1)
    sent_sim = np.clip(_dense_similarity(req_vecs, sent_vecs), -1, 1)
    offsets = defaultdict(list)
    for idx, (paper_id, _, _) in enumerate(passage_info):
        offsets[paper_id].append(idx)
    # One maximum sentence score per (requirement, paper), for scoring and labels.
    best_indices = np.zeros((nr, npapers), dtype=int)
    best_sentence_sim = np.zeros((nr, npapers), dtype=np.float64)
    for k, paper_id in enumerate(paper_ids):
        idxs = offsets[paper_id]
        for j in range(nr):
            best_loc = int(idxs[int(np.argmax(sent_sim[j, idxs]))])
            best_indices[j, k] = best_loc
            best_sentence_sim[j, k] = sent_sim[j, best_loc]
    raw_sim = 0.5 * (full_sim + best_sentence_sim)
    if low is None:
        calibration_low, calibration_high = (float(v) for v in np.percentile(raw_sim, [10, 90]))
        method = "within_query_p10_p90"
    else:
        calibration_low, calibration_high = float(low), float(high)
        method = "fixed_user_supplied"
    if calibration_high-calibration_low < 1e-8:
        # Identical raw scores carry zero within-query discriminatory evidence.
        calibrated = np.full_like(raw_sim, 0.5)
        method += "_degenerate_equal_scores"
    else:
        calibrated = np.clip((raw_sim-calibration_low)/(calibration_high-calibration_low), 0, 1)
    project_high = float(np.percentile(project_doc_sim, 75))
    project_low = float(np.percentile(project_doc_sim, 50))
    strong_sentence = np.percentile(best_sentence_sim, 75, axis=1)
    calibration = {"method": method, "low": calibration_low, "high": calibration_high,
                   "percentiles": [10, 90] if low is None else None,
                   "label_score_threshold": evidence_label_threshold,
                   "label_project_high_p75": project_high,
                   "label_project_low_p50": project_low,
                   "label_sentence_high_p75_per_requirement": [float(v) for v in strong_sentence],
                   "warning": "Query-relative scores and quantile labels are retrieval heuristics, not calibrated probabilities of expertise."}
    scores = {p["id"]: np.zeros(nr, dtype=np.float64) for p in data["professors"]}
    selected_evidence = {p["id"]: [None] * nr for p in data["professors"]}
    paper_evidence = []
    for k, paper_id in enumerate(paper_ids):
        for j in range(nr):
            best_passage, passage_source = passage_info[int(best_indices[j, k])][1:]
            doc_cos = float(full_sim[j, k])
            sentence_cos = float(best_sentence_sim[j, k])
            raw = float(raw_sim[j, k])
            score = float(calibrated[j, k])
            record = {"requirement_id": requirements[j]["id"],
                      "requirement": requirements[j]["description"],
                      "paper_id": paper_id, "title": papers[paper_id].get("title", ""),
                      "year": papers[paper_id].get("year"), "author_ids": sorted(authors[paper_id]),
                      "text_source": "abstract" if papers[paper_id].get("abstract") else "title",
                      "score": round(score, 7), "raw_combined_cosine": round(raw, 7),
                      "document_cosine": round(doc_cos, 7), "sentence_cosine": round(sentence_cos, 7),
                      "project_document_cosine": round(float(project_doc_sim[k]), 7),
                      "supporting_passage": best_passage,
                      "supporting_passage_source": passage_source,
                      "supporting_abstract_sentence": best_passage if passage_source == "abstract" else None,
                      "relationship_candidate": evidence_type(
                          float(project_doc_sim[k]), sentence_cos, score,
                          strong_project_cutoff=project_high,
                          weak_project_cutoff=project_low,
                          strong_sentence_cutoff=float(strong_sentence[j]),
                          score_threshold=evidence_label_threshold)}
            paper_evidence.append(record)
            for professor in authors[paper_id]:
                old = float(scores[professor][j])
                previous = selected_evidence[professor][j]
                if (previous is None or score > old + EPS or
                    (abs(score-old) <= EPS and
                     (raw > previous["raw_combined_cosine"]+EPS or
                      (abs(raw-previous["raw_combined_cosine"]) <= EPS and paper_id < previous["paper_id"])))):
                    scores[professor][j] = score
                    selected_evidence[professor][j] = record
    if return_retrieval_vectors:
        retrieval_vectors = {"paper_ids": paper_ids, "documents": full_vecs,
                             "requirements": req_vecs, "project": project_vec[0],
                             "model": encoder.model}
        return scores, selected_evidence, paper_evidence, calibration, retrieval_vectors
    return scores, selected_evidence, paper_evidence, calibration


def _l2_normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), EPS)


def load_precomputed_topic_clusters(folder, data, paper_ids, dimension, model):
    """Read an offline NCUT assignment, verifying exact document IDs and hashes.

    The offline file is tied to a corpus, not merely a shape: reordering papers,
    replacing abstracts, or changing the Qwen checkpoint must fail visibly.
    """
    root = Path(folder)
    mf = root / "manifest.json"
    af = root / "paper_clusters.jsonl"
    cf = root / "centroids.npy"
    if not all(p.is_file() for p in (mf, af, cf)):
        raise FileNotFoundError(f"NCUT preclustering missing/incomplete at {root}; "
                                "run python cluster_ncut_abstracts.py first")
    manifest = load_json(mf)
    if (manifest.get("status") != "complete" or manifest.get("format_version") != 1
            or manifest.get("method") != "ncut_pytorch_nystrom_kway"
            or manifest.get("model") != model or manifest.get("document_count") != len(paper_ids)
            or manifest.get("embedding_dimension") != dimension):
        raise ValueError("Precomputed NCUT manifest differs from the model or paper corpus")
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("Duplicate unique paper IDs")
    by_id = {}
    for prof in data["professors"]:
        for paper in prof["papers"]:
            by_id[paper["id"]] = paper
    from cluster_ncut_abstracts import fingerprint_documents
    expected = []
    for pid in paper_ids:
        paper = by_id[pid]
        full_text = f"{paper.get('title') or ''}. {paper.get('abstract') or ''}".strip(" .")
        expected.append((pid, hashlib.sha256(full_text.encode("utf-8")).hexdigest()))
    if fingerprint_documents(expected) != manifest.get("document_fingerprint_sha256"):
        raise ValueError("Precomputed NCUT paper IDs or title/abstract hashes are stale")
    labels = []
    with af.open(encoding="utf-8") as stream:
        for idx, line in enumerate(stream):
            if idx >= len(expected):
                raise ValueError("Too many precomputed NCUT assignments")
            rec = json.loads(line)
            if (rec.get("paper_id"), rec.get("text_sha256")) != expected[idx]:
                raise ValueError(f"Precomputed NCUT row mismatch at position {idx}")
            labels.append(rec["cluster_id"])
    if len(labels) != len(expected):
        raise ValueError("Too few precomputed NCUT assignments")
    labels = np.asarray(labels)
    k = manifest["actual_clusters"]
    if (not np.issubdtype(labels.dtype, np.integer) or k < 2
            or not np.array_equal(np.unique(labels), np.arange(k))):
        raise ValueError("Precomputed NCUT labels are invalid or contain empty cluster IDs")
    centers = np.load(cf, allow_pickle=False)
    if (centers.shape != (k, dimension) or not np.isfinite(centers).all()
            or not np.allclose(np.linalg.norm(centers, axis=1), 1, atol=1e-3)):
        raise ValueError("Invalid precomputed NCUT Qwen-space centroids")
    counts = np.bincount(labels, minlength=k)
    if np.any(counts <= 0):
        raise ValueError("Precomputed NCUT contains empty clusters")
    return labels.astype(np.int32), centers.astype(np.float64), manifest


def build_topic_brokerage(data, graph, requirements, scores, initiator, retrieval,
                          paper_evidence, *, precomputed_dir):
    """Find real boundary brokers and requirement-relevant external coauthors.

    A broker publishes primarily in the initiator's project-aligned NCUT cluster
    and has an OBSERVED coauthor in a different primary NCUT cluster. Adding a
    previously absent initiator--broker edge can open initiator--broker--scientist
    wedges. Only scientists whose outside-cluster papers improve coverage of a
    different project requirement can be witnesses. No reach/exponential or
    arbitrary topic-similarity cutoffs are used.
    """
    paper_ids = retrieval["paper_ids"]
    documents = _l2_normalize(retrieval["documents"])
    dim = documents.shape[1]
    if not precomputed_dir:
        raise ValueError("Component 2 requires precomputed NCUT clusters")
    labels, centers, _manifest = load_precomputed_topic_clusters(
        precomputed_dir, data, paper_ids, dim, retrieval["model"])
    k = centers.shape[0]  # Inferred from the actual NCUT output.
    cluster_of_paper = dict(zip(paper_ids, map(int, labels)))

    # Each paper has one NCUT cluster; a faculty member's primary cluster is
    # their most-published cluster (deterministic lowest-ID tie-break).
    profiles = {p: np.zeros(k, dtype=np.float64) for p in graph.nodes}
    for prof in data["professors"]:
        profile = profiles[prof["id"]]
        for paper in prof["papers"]:
            cluster = cluster_of_paper.get(paper["id"])
            if cluster is not None:
                profile[cluster] += 1.0
        if profile.sum() > 0:
            profile /= profile.sum()
    primary = {p: (int(np.argmax(profile)) if profile.sum() > EPS else None)
               for p, profile in profiles.items()}

    project_vec = _l2_normalize(retrieval["project"])
    project_cos = centers @ project_vec
    initiator_profile = profiles[initiator]
    source_cluster = None
    if initiator_profile.sum() > EPS:
        source_scores = initiator_profile * np.maximum(project_cos, 0.0)
        if source_scores.sum() <= EPS:
            source_scores = initiator_profile
        source_cluster = int(np.argmax(source_scores))

    # Identify the scientific requirement the initiator most strongly covers.
    # An external scientist must improve a DIFFERENT requirement, based on
    # papers actually assigned outside the initiator's source topic.
    weights = np.array([r["weight"] for r in requirements], dtype=np.float64)
    own_scores = np.asarray(scores[initiator], dtype=np.float64)
    strongest = weights * own_scores
    home_requirement = int(np.argmax(strongest)) if strongest.max() > EPS else None
    req_idx = {r["id"]: j for j, r in enumerate(requirements)}
    outside_scores = {p: np.zeros(len(requirements), dtype=np.float64)
                      for p in graph.nodes}
    outside_papers = {p: {} for p in graph.nodes}
    if source_cluster is not None:
        for row in paper_evidence:
            if cluster_of_paper.get(row["paper_id"]) == source_cluster:
                continue
            j = req_idx[row["requirement_id"]]
            paper_score = float(row["score"])
            for p in row["author_ids"]:
                if paper_score > outside_scores[p][j]:
                    outside_scores[p][j] = paper_score
                    outside_papers[p][j] = row

    names = {p["id"]: p["name"] for p in data["professors"]}
    own_neighbors = set(graph.neighbors(initiator))
    witness_by_scientist = {}

    def relevant_scientist(s):
        if s in witness_by_scientist:
            return witness_by_scientist[s]
        best = None
        if home_requirement is not None:
            for j, req in enumerate(requirements):
                if j == home_requirement:
                    continue
                improvement = max(0.0, float(outside_scores[s][j] - own_scores[j]))
                relevance = float(own_scores[home_requirement] * weights[j] * improvement)
                if relevance <= EPS:
                    continue
                if best is None or relevance > best["relevance"]:
                    best = {
                        "scientist_id": s,
                        "scientist_name": names[s],
                        "scientist_primary_cluster": primary[s],
                        "requirement_id": req["id"],
                        "requirement_description": req["description"],
                        "initiator_requirement_score": float(own_scores[j]),
                        "scientist_outside_paper_score": float(outside_scores[s][j]),
                        "relevance": relevance,
                        "supporting_paper": outside_papers[s][j],
                    }
        witness_by_scientist[s] = best
        return best

    witnesses_by_broker = {}
    if source_cluster is not None:
        for b in sorted(graph.nodes):
            if b == initiator or b in own_neighbors or primary[b] != source_cluster:
                continue
            witnesses = []
            for s in sorted(graph.neighbors(b)):
                if (s == initiator or s in own_neighbors or
                        primary[s] is None or primary[s] == source_cluster):
                    continue
                relevant = relevant_scientist(s)
                if relevant is not None:
                    witnesses.append(relevant)
            if witnesses:
                witnesses_by_broker[b] = tuple(witnesses)

    # One global, team-independent bound: at most K direct-link benefits and
    # at most 2*g(s) across all selected brokers for each external scientist.
    all_witnesses = {w["scientist_id"]: w for witnesses in witnesses_by_broker.values()
                     for w in witnesses}
    titles = {d["id"]: d.get("title", "") for p in data["professors"] for d in p["papers"]}
    topics = []
    for c in range(k):
        idxs = np.flatnonzero(labels == c)
        examples = sorted(idxs, key=lambda i: (-float(documents[i] @ centers[c]), paper_ids[i]))[:3]
        topics.append({
            "id": c, "paper_count": len(idxs),
            "representative_papers": [
                {"paper_id": paper_ids[i], "title": titles.get(paper_ids[i], "")}
                for i in examples],
            "source_cluster": c == source_cluster,
            "project_cosine": round(float(project_cos[c]), 7),
            "eligible_external_scientist_count": sum(
                w["scientist_primary_cluster"] == c for w in all_witnesses.values()),
        })
    return {
        "paper_cluster": cluster_of_paper,
        "profiles": profiles,
        "primary_clusters": primary,
        "source_cluster": source_cluster,
        "home_requirement": (requirements[home_requirement]["id"]
                             if home_requirement is not None else None),
        "witnesses_by_broker": witnesses_by_broker,
        "relevant_scientists": all_witnesses,
        "topics": topics,
        "settings": {
            "model": "requirement_relevant_new_open_wedges_harmonic_v1",
            "cluster_source": "precomputed_ncut",
            "actual_clusters": k,
            "source_cluster": source_cluster,
            "initiator_primary_requirement": (requirements[home_requirement]["id"]
                                               if home_requirement is not None else None),
            "eligible_boundary_brokers": len(witnesses_by_broker),
            "distinct_relevant_external_scientists": len(all_witnesses),
            "direct_link_unit_alpha0": 1.0,
            "intermediary_payoff": "2 * scientific_relevance / number_of_two_hop_intermediaries",
        },
    }


class Objective:
    """Joint scientific and new requirement-relevant structural-hole objective.

    The team contains the initiator plus selected additional professors.
    Each proposed initiator--broker edge is evaluated on the FULL team graph:
    new edges can create competing intermediaries or close open wedges.
    Observed coauthorship edges are never modified.
    """

    def __init__(self, initiator, graph, scores, weights, *, max_additional=3,
                 lambda_network=0.25, topic_brokerage=None):
        if initiator not in graph:
            raise ValueError("Unknown initiator")
        if max_additional < 0 or not math.isfinite(lambda_network) or lambda_network < 0:
            raise ValueError("max_additional and lambda_network must be nonnegative")
        if topic_brokerage is None:
            raise ValueError("NCUT structural-hole brokerage data must be computed first")
        self.u, self.g, self.scores = initiator, graph, scores
        self.w = np.asarray(weights, dtype=np.float64)
        if len(self.w) == 0 or abs(float(self.w.sum()) - 1) > 1e-6:
            raise ValueError("weights must be nonempty and normalized")
        if any(p not in scores for p in graph.nodes):
            raise ValueError("Missing professor score vector")
        self.max_additional = int(max_additional)
        self.lambda_network = float(lambda_network)
        self.topic_brokerage = topic_brokerage
        self.witnesses = topic_brokerage["witnesses_by_broker"]
        self.original_neighbors = {p: frozenset(graph.neighbors(p)) for p in graph.nodes}
        self.initiator_neighbors = self.original_neighbors[initiator]
        distinct_witnesses = topic_brokerage["relevant_scientists"]
        self.network_normalizer = (
            float(self.max_additional) +
            2.0 * sum(w["relevance"] for w in distinct_witnesses.values())
        ) if self.witnesses and self.max_additional > 0 else 0.0
        self.topic_brokerage["settings"]["normalization_upper_bound"] = self.network_normalizer

    def valid(self, additional):
        members = set(additional)
        return (len(members) <= self.max_additional and self.u not in members
                and members.issubset(self.g.nodes))

    @lru_cache(maxsize=100000)
    def science(self, additional: frozenset):
        members = sorted(additional | {self.u})
        mat = np.vstack([self.scores[p] for p in members])
        return float(self.w @ mat.max(axis=0))

    def _open_wedges(self, additional):
        """Yield (broker, relevant-scientist evidence, r_us) on proposed graph.

        With only new initiator--member edges, N_T(u)=N_G(u) union T. For s != u,
        intersecting that set with N_G(s) counts the same two-hop paths as
        materializing G_T, without copying the entire collaboration graph.
        """
        proposed_neighbors = self.initiator_neighbors | frozenset(additional)
        for b in sorted(additional):
            if b in self.initiator_neighbors:
                continue  # Not a newly created direct link.
            for scientist in self.witnesses.get(b, ()):
                s = scientist["scientist_id"]
                if s in proposed_neighbors:
                    continue  # An existing/proposed u--s edge closes this wedge.
                r_us = len(proposed_neighbors & self.original_neighbors[s])
                if r_us:
                    yield b, scientist, r_us

    @lru_cache(maxsize=100000)
    def network(self, additional: frozenset):
        if not additional or self.network_normalizer <= EPS:
            return 0.0
        # Each eligible new u--b edge gets +1 only if it creates at least one
        # relevant open wedge. Each wedge contributes 2*g_q(s)/r_us.
        active_brokers = set()
        intermediary = 0.0
        for b, scientist, r_us in self._open_wedges(additional):
            active_brokers.add(b)
            intermediary += 2.0 * scientist["relevance"] / r_us
        raw = len(active_brokers) + intermediary
        return float(np.clip(raw / self.network_normalizer, 0.0, 1.0))

    @lru_cache(maxsize=100000)
    def value(self, additional: frozenset):
        return self.science(additional) + self.lambda_network * self.network(additional)

    def components(self, additional):
        additional = frozenset(additional)
        if not self.valid(additional):
            raise ValueError("Invalid proposed team")
        return {"scientific": self.science(additional),
                "brokerage": self.network(additional),
                "combined": self.value(additional)}

    def marginal(self, candidate, additional):
        team = frozenset(additional)
        if candidate in team or candidate == self.u or len(team) >= self.max_additional:
            return float("-inf")
        return self.value(team | {candidate}) - self.value(team)

    def brokerage_explanation(self, candidate, team):
        """Actual proposed brokerage paths, with observed coauthors and papers."""
        paths = []
        for b, s, r_us in self._open_wedges(frozenset(team)):
            if b != candidate:
                continue
            observed = self.g[b][s["scientist_id"]]
            raw_payoff = 2.0 * s["relevance"] / r_us
            paths.append({
                "broker_id": b,
                "new_collaboration_edge": {"source": self.u, "target": b},
                "observed_coauthor_edge": {
                    "source": b, "target": s["scientist_id"],
                    "paper_ids": observed.get("paper_ids", []),
                    "weight": observed.get("weight"),
                    "last_year": observed.get("last_year"),
                },
                "external_scientist_id": s["scientist_id"],
                "external_scientist_name": s["scientist_name"],
                "external_scientist_primary_cluster": s["scientist_primary_cluster"],
                "requirement_id": s["requirement_id"],
                "requirement_description": s["requirement_description"],
                "initiator_requirement_score": round(s["initiator_requirement_score"], 7),
                "scientist_outside_paper_score": round(s["scientist_outside_paper_score"], 7),
                "relevance": round(s["relevance"], 7),
                "two_hop_intermediaries": r_us,
                "raw_intermediary_payoff": round(raw_payoff, 7),
                "normalized_intermediary_payoff": round(
                    raw_payoff / self.network_normalizer, 7),
                "scientific_supporting_paper": s["supporting_paper"],
            })
        paths.sort(key=lambda x: (-x["raw_intermediary_payoff"], x["external_scientist_id"]))
        return paths


def _event(obj, run, round_index, phase, action, before, after, **details):
    before, after = frozenset(before), frozenset(after)
    evt = {"round": round_index, "phase": phase, "action": action,
           "additional_before": sorted(before), "additional_after": sorted(after),
           "scientific_after": round(obj.science(after), 7),
           "brokerage_after": round(obj.network(after), 7),
           "combined_after": round(obj.value(after), 7),
           "combined_gain": round(obj.value(after)-obj.value(before), 7)}
    evt.update(details)
    run["events"].append(evt)
    run["visited"].add(after)


def sequential_search(candidates, obj, *, seed, top_r=4, max_rounds=12,
                      min_gain=1e-9, randomized=True, forced_first=None):
    """Randomized restricted-candidate greedy + immediately accepted local edits.

    Later choices depend on previously selected people. Randomized order affects
    discovery paths; final objective is order invariant.
    """
    rng = random.Random(seed)
    run = {"seed": seed, "search_type": "randomized" if randomized else "greedy",
           "forced_first": forced_first, "events": [], "rounds": [],
           "visited": {frozenset()}}
    team = frozenset()
    if forced_first is not None:
        if forced_first not in candidates or obj.max_additional == 0:
            raise ValueError("Invalid forced first selection")
        new = team | {forced_first}
        _event(obj, run, 0, "construction", "forced_first", team, new, candidate=forced_first)
        team = new
    while len(team) < obj.max_additional:
        ranked = sorted(((obj.marginal(p, team), p) for p in candidates if p not in team),
                        key=lambda x: (-x[0], x[1]))
        eligible = [(gain, p) for gain, p in ranked if gain > min_gain]
        if not eligible:
            break
        gain, chosen = rng.choice(eligible[:top_r]) if randomized else eligible[0]
        new = team | {chosen}
        _event(obj, run, 0, "construction", "add", team, new,
               candidate=chosen, marginal_gain=round(gain, 7))
        team = new
    run["constructed_additional"] = sorted(team)
    for round_num in range(1, max_rounds + 1):
        changed = 0
        attempts = 0
        # Re-evaluate after every accepted edit; do not score from an old team.
        for candidate in rng.sample(candidates, len(candidates)) if randomized else candidates:
            if candidate in team:
                continue
            proposals = []
            if len(team) < obj.max_additional:
                proposals.append(("add", team | {candidate}, None))
            removals = rng.sample(sorted(team), len(team)) if randomized else sorted(team)
            proposals.extend(("swap", (team - {removed}) | {candidate}, removed)
                             for removed in removals)
            for action, trial, removed in proposals:
                attempts += 1
                gain = obj.value(trial)-obj.value(team)
                if gain > min_gain:
                    before = team
                    team = trial
                    _event(obj, run, round_num, "revision", action, before, team,
                           candidate=candidate, removed=removed)
                    changed += 1
                    break
        for removed in (rng.sample(sorted(team), len(team)) if randomized else sorted(team)):
            trial = team-{removed}
            attempts += 1
            if obj.value(trial)-obj.value(team) > min_gain:
                before = team
                team = trial
                _event(obj, run, round_num, "revision", "remove", before, team, removed=removed)
                changed += 1
        run["rounds"].append({"round": round_num, "attempted_edits": attempts,
                              "accepted_edits": changed, "team": sorted(team),
                              "score": round(obj.value(team), 7)})
        if changed == 0:
            break
    run["final_additional"] = sorted(team)
    run["final_score"] = round(obj.value(team), 7)
    return run


def nondominated(teams, obj):
    """Pareto nondominance within discovered pool; ties are both nondominated."""
    teams = list(teams)
    coords = {t: (obj.science(t), obj.network(t)) for t in teams}
    frontier = set()
    for t in teams:
        s, b = coords[t]
        if not any((coords[o][0] >= s-EPS and coords[o][1] >= b-EPS and
                    (coords[o][0] > s+EPS or coords[o][1] > b+EPS))
                   for o in teams if o != t):
            frontier.add(t)
    return frontier


def team_record(team, obj, profs, requirements, selected_evidence):
    team = frozenset(team)
    full = [obj.u] + sorted(team)
    comps = obj.components(team)
    need_details = []
    for j, req in enumerate(requirements):
        owner = min(full, key=lambda p: (-obj.scores[p][j], p))
        best = selected_evidence[owner][j]
        need_details.append({"requirement_id": req["id"], "description": req["description"],
                             "weight": req["weight"], "team_quality": round(float(obj.scores[owner][j]), 7),
                             "best_professor_id": owner,
                             "best_professor_name": profs[owner]["name"],
                             "best_professor_departments": profs[owner].get("departments", ""),
                             "best_paper": best})
    contributions = []
    for p in sorted(team):
        without = team-{p}
        evidence = []
        for j, req in enumerate(requirements):
            if obj.scores[p][j] > 0:
                evidence.append({"requirement_id": req["id"],
                                 "professor_requirement_score": round(float(obj.scores[p][j]), 7),
                                 "best_paper": selected_evidence[p][j]})
        contributions.append({"professor_id": p, "name": profs[p]["name"],
                              "departments": profs[p].get("departments", ""),
                              "scientific_leave_one_out": round(obj.science(team)-obj.science(without), 7),
                              "brokerage_leave_one_out": round(obj.network(team)-obj.network(without), 7),
                              "combined_leave_one_out": round(obj.value(team)-obj.value(without), 7),
                              "supporting_papers_by_requirement": evidence,
                              "brokerage_path_evidence": obj.brokerage_explanation(p, team)})
    return {"professor_ids": full, "additional_professor_ids": sorted(team),
            "professor_names": [profs[p]["name"] for p in full],
            "professor_departments": [profs[p].get("departments", "") for p in full],
            "hypothetical_collaboration_links": [
                {"source": obj.u, "source_name": profs[obj.u]["name"],
                 "source_departments": profs[obj.u].get("departments", ""),
                 "target": p, "target_name": profs[p]["name"],
                 "target_departments": profs[p].get("departments", ""), "directed": False,
                 "already_observed_coauthors": obj.g.has_edge(obj.u, p),
                 "would_be_new_edge": not obj.g.has_edge(obj.u, p)}
                for p in sorted(team)],
            "professors": [{"id": p, "name": profs[p]["name"],
                            "departments": profs[p].get("departments", "")} for p in full],
            "additional_count": len(team), "scientific_score": round(comps["scientific"], 7),
            "brokerage_score": round(comps["brokerage"], 7),
            "combined_score": round(comps["combined"], 7),
            "per_requirement": need_details, "contributions": contributions}


def make_query_evidence_graph(heterogeneous, requirements, paper_evidence, min_score=0.05):
    """Ephemeral project view: paper -> query-requirement scientific edges.

    Unlike stable authorship/coauthorship, these relationships are specific to
    ONE input abstract. They are NOT inserted into the observed dataset.
    """
    if not 0 <= min_score <= 1:
        raise ValueError("Query evidence graph minimum score must be in [0,1]")
    nodes = list(heterogeneous["nodes"])
    edges = list(heterogeneous["edges"])
    for req in requirements:
        nodes.append({"id": f"query:{req['id']}", "node_type": "query_requirement",
                      "description": req["description"], "weight": req["weight"]})
    for row in paper_evidence:
        if row["score"] + EPS < min_score or row["score"] <= 0:
            continue
        edges.append({"source": f"paper:{row['paper_id']}",
                      "target": f"query:{row['requirement_id']}",
                      "edge_type": "query_scientific_evidence",
                      "directed": True, "score": row["score"],
                      "relationship_candidate": row["relationship_candidate"],
                      "supporting_abstract_sentence": row["supporting_abstract_sentence"],
                      "supporting_passage": row["supporting_passage"],
                      "supporting_passage_source": row["supporting_passage_source"],
                      "document_cosine": row["document_cosine"],
                      "project_document_cosine": row["project_document_cosine"],
                      "sentence_cosine": row["sentence_cosine"]})
    return {"nodes": nodes, "edges": edges,
            "note": "Project-specific paper-to-requirement evidence edges. Not observed collaborations or proven knowledge transfer."}


def recommend(data, query, *, encoder=None, max_additional=3, lambda_network=0.25,
              threshold_low=None, threshold_high=None,
              evidence_label_threshold=None, restarts=30, top_r=4,
              max_rounds=12, min_gain=1e-9, science_tolerance=0.03,
              n_teams=5, seed=42, exact_check_max_candidates=0,
              embedding_cache_dir=None, topic_cluster_dir=None):
    ids = validate_dataset(data)
    reqs = resolve_requirements(query, ids)
    if encoder is None:
        encoder = TextEncoder()
    if restarts < 0 or top_r < 1 or max_rounds < 0 or n_teams < 1:
        raise ValueError("Invalid search parameters")
    if min_gain < 0 or science_tolerance < 0 or exact_check_max_candidates < 0:
        raise ValueError("Gains, tolerance and exact cap must be nonnegative")
    scores, evidence, paper_rows, calibration, retrieval = score_papers(
        data, reqs, encoder, threshold_low, threshold_high,
        evidence_label_threshold, project_abstract=query.get("abstract"),
        cache_root=embedding_cache_dir, return_retrieval_vectors=True)
    graph = make_collaboration_graph(data)
    u = query["initiator_id"]
    profs = {p["id"]: p for p in data["professors"]}
    weights = [r["weight"] for r in reqs]
    topics = build_topic_brokerage(data, graph, reqs, scores, u, retrieval,
                                   paper_rows, precomputed_dir=topic_cluster_dir)
    obj = Objective(u, graph, scores, weights, max_additional=max_additional,
                    lambda_network=lambda_network, topic_brokerage=topics)
    candidates = sorted(set(profs)-{u})
    runs = [sequential_search(candidates, obj, seed=seed, top_r=top_r,
                              max_rounds=max_rounds, min_gain=min_gain, randomized=False)]
    for k in range(restarts):
        # Explicitly explore distinct first choices in the first batch of restarts.
        forced = candidates[k] if k < len(candidates) and max_additional > 0 else None
        runs.append(sequential_search(candidates, obj, seed=seed+k+1, top_r=top_r,
                                      max_rounds=max_rounds, min_gain=min_gain,
                                      randomized=True, forced_first=forced))
    discovered = {team for run in runs for team in run["visited"]}
    final_teams = {frozenset(run["final_additional"]) for run in runs}
    exact = None
    if exact_check_max_candidates and len(candidates) <= exact_check_max_candidates:
        all_teams = {frozenset(s) for k in range(min(max_additional, len(candidates))+1)
                     for s in itertools.combinations(candidates, k)}
        discovered |= all_teams
        exact = {"enumerated_team_count": len(all_teams),
                 "best_objective": round(max(obj.value(s) for s in all_teams), 7)}
    frontier = nondominated(discovered, obj)
    max_science = max(obj.science(t) for t in discovered)
    eligible_default = [t for t in discovered if obj.science(t) >= max_science-science_tolerance-EPS]
    order = lambda t: (-obj.value(t), -obj.science(t), -obj.network(t), len(t), tuple(sorted(t)))
    champion = min(eligible_default, key=order)
    # Output alternatives ranked by objective. We do not force every reported team
    # to be Pareto optimal, since users explicitly want multiple combinations.
    alternatives = sorted(discovered-{champion}, key=order)
    picks = [champion] + alternatives[:n_teams-1]
    teams = []
    for idx, team in enumerate(picks, 1):
        record = team_record(team, obj, profs, reqs, evidence)
        record.update({"rank": idx, "is_default": idx == 1,
                       "pareto_nondominated_in_discovered_pool": team in frontier,
                       "seen_as_final_run_team": team in final_teams,
                       "within_science_tolerance": obj.science(team) >= max_science-science_tolerance-EPS})
        teams.append(record)
    hgraph = build_heterogeneous_graph(data)
    graph_out = {"nodes": [{"id": node_id, **attrs} for node_id, attrs in hgraph.nodes(data=True)],
                 "edges": [{"source": a, "target": b, **attrs} for a, b, attrs in hgraph.edges(data=True)]}
    # Sets are used internally to deduplicate path states; do not serialize them.
    for run in runs:
        run["visited_unique_team_count"] = len(run["visited"])
        del run["visited"]
    query_graph = make_query_evidence_graph(graph_out, reqs, paper_rows)
    result = {"project_title": query.get("title", ""), "initiator_id": u,
              "initiator_name": profs[u]["name"],
              "initiator_departments": profs[u].get("departments", ""),
              "model": "paper_evidence_qwen3_structural_holes_v1", "encoder_backend": encoder.backend,
              "encoder_model": encoder.model,
              "max_additional_professors": max_additional,
              "max_team_size_including_initiator": max_additional+1,
              "lambda_network": lambda_network,
              "science_tolerance": science_tolerance,
              "raw_similarity_calibration": calibration,
              "relationship_label_threshold": calibration["label_score_threshold"],
              "input_data_summary": data.get("source_summary"),
              "requirement_count": len(reqs),
              "initiator_only_scientific_score": round(obj.science(frozenset()), 7),
              "initiator_only_brokerage_score": 0.,
              "brokerage_method": topics["settings"],
              "brokerage_definition": (
                  "A selected boundary professor becomes a new intermediary on the open wedge "
                  "initiator--broker--observed external coauthor. The coauthor must have "
                  "outside-source-cluster publication evidence improving a different project "
                  "requirement. Each newly created broker edge receives direct-link benefit "
                  "1 if it opens a relevant wedge; each wedge receives 2*relevance/r_us. "
                  "The team payoff is divided by a fixed data-dependent upper bound. "
                  "This is a project-conditioned extension of Kleinberg-style harmonic "
                  "structural-hole payoff, not the original network game."),
              "source_and_destination_topics": topics["topics"],
              "discovered_unique_team_count": len(discovered),
              "pareto_frontier_count": len(frontier), "recommended_teams": teams,
              "selected_default_team": teams[0],
              "exact_check": exact,
              "warnings": [
                  "Paper similarity is a retrieval proxy, not verified expertise or individual author contribution.",
                  "Similarity/complementary labels are tentative and should be manually evaluated.",
                  "NCUT clusters are unsupervised heuristics; review the source and external scientists' clusters for topic coherence.",
                  "Only observed broker-scientist edges support proposed intermediary paths; no future introductions are guaranteed.",
                  "Brokerage extends Kleinberg-style harmonic open-wedge payoff with requirement relevance; it is not the paper's exact network game.",
                  "All proposed u-to-candidate edges are hypothetical; the observed graph is unchanged.",
                  "The default choice is best combined score within science_tolerance of highest discovered scientific score.",
                  "Requirements were decomposed by generative Qwen3 from the proposal abstract; Qwen3-Embedding scores paper evidence.",
                  "Scores are relative to the candidate papers for this project unless fixed calibration cutoffs were supplied.",
                  "Title-only papers use the title as the evidence passage; title-based evidence is less detailed than abstract-based evidence."
              ]}
    professor_directory = [
        {"professor_id": p, "name": profs[p]["name"],
         "departments": profs[p].get("departments", "")}
        for p in sorted(profs)
    ]
    return {"result": result, "requirements": reqs, "graph": graph_out,
            "professor_directory": professor_directory,
            "query_graph": query_graph,
            "topic_papers": [{"paper_id": pid, "cluster_id": int(topics["paper_cluster"][pid])}
                             for pid in retrieval["paper_ids"]],
            "paper_evidence": paper_rows, "professor_requirement_scores":
                [{"professor_id": p, "requirement_id": reqs[j]["id"],
                  "professor_name": profs[p]["name"],
                  "professor_departments": profs[p].get("departments", ""),
                  "score": round(float(scores[p][j]), 7), "best_paper": evidence[p][j]}
                 for p in sorted(profs) for j in range(len(reqs))],
            "trace": {"greedy": runs[0], "randomized_runs": runs[1:]}}


def export_results(folder, artifacts):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key, filename in [("result", "recommendations.json"),
                          ("requirements", "requirements.json"), ("graph", "graph.json"),
                          ("professor_directory", "professor_directory.json"),
                          ("query_graph", "query_evidence_graph.json"),
                          ("topic_papers", "paper_topic_assignments.json"),
                          ("paper_evidence", "paper_evidence.json"),
                          ("professor_requirement_scores", "professor_scores.json"),
                          ("trace", "trace.json")]:
        save_json(folder/filename, artifacts[key])
    directory = {row["professor_id"]: row for row in artifacts["professor_directory"]}
    with (folder/"professor_directory.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["professor_id", "name", "departments"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(artifacts["professor_directory"])
    with (folder/"paper_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["requirement_id", "paper_id", "title", "author_ids", "text_source", "score",
                "raw_combined_cosine", "document_cosine", "sentence_cosine", "project_document_cosine",
                "relationship_candidate", "supporting_abstract_sentence",
                "supporting_passage", "supporting_passage_source"]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in artifacts["paper_evidence"]:
            writer.writerow({**row, "author_ids": ";".join(row["author_ids"])})
    with (folder/"recommended_teams.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["rank", "professor_ids", "professor_names", "professor_departments",
                "additional_count", "scientific_score", "brokerage_score", "combined_score",
                "pareto_nondominated_in_discovered_pool"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for team in artifacts["result"]["recommended_teams"]:
            writer.writerow({k: ";".join(team[k]) if k in
                             ("professor_ids", "professor_names", "professor_departments")
                             else team[k] for k in cols})
    with (folder/"recommended_edges.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["source", "source_name", "source_departments", "target", "target_name",
                "target_departments", "already_observed_coauthors", "scientific_leave_one_out",
                "brokerage_leave_one_out", "combined_leave_one_out"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        chosen = artifacts["result"]["selected_default_team"]
        for p in chosen["contributions"]:
            source = artifacts["result"]["initiator_id"]
            p_id = p["professor_id"]
            link = next(link for link in chosen["hypothetical_collaboration_links"] if link["target"] == p_id)
            writer.writerow({"source": source,
                             "source_name": directory[source]["name"],
                             "source_departments": directory[source]["departments"],
                             "target": p_id,
                             "target_name": directory[p_id]["name"],
                             "target_departments": directory[p_id]["departments"],
                             "already_observed_coauthors": link["already_observed_coauthors"],
                             **{key: p[key] for key in cols[7:]}})
    with (folder/"rounds.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "round", "attempted_edits",
                                               "accepted_edits", "team", "team_names",
                                               "team_departments", "score"])
        writer.writeheader()
        for run in [artifacts["trace"]["greedy"]] + artifacts["trace"]["randomized_runs"]:
            for row in run["rounds"]:
                team_ids = row["team"]
                writer.writerow({"seed": run["seed"],
                                 **{k: ";".join(v) if k == "team" else v
                                    for k, v in row.items()},
                                 "team_names": ";".join(directory[p]["name"] for p in team_ids),
                                 "team_departments": ";".join(directory[p]["departments"]
                                                               for p in team_ids)})


def evolve(data, updates):
    """Create a new observed-data snapshot. Recommendations NEVER become facts."""
    validate_dataset(data)
    new = deepcopy(data)
    by_id = {p["id"]: p for p in new["professors"]}
    for p in updates.get("new_professors", []):
        if p["id"] in by_id:
            raise ValueError(f"Existing professor id: {p['id']}")
        new["professors"].append(p)
        by_id[p["id"]] = p
    for item in updates.get("new_papers", []):
        pid = item["professor_id"]
        p = by_id[pid]
        if all(x["id"] != item["paper"]["id"] for x in p["papers"]):
            p["papers"].append(item["paper"])
    seen = {frozenset((e["source"], e["target"])) for e in new.get("coauthorship_edges", [])}
    for e in updates.get("new_collaborations", []):
        key = frozenset((e["source"], e["target"]))
        if key not in seen:
            new.setdefault("coauthorship_edges", []).append(e)
            seen.add(key)
    new["version"] = int(data.get("version", 1))+1
    new.setdefault("update_log", []).append({"version": new["version"],
                                               "note": updates.get("note", "Verified observed-data update")})
    validate_dataset(new)
    return new


def _load_configured_data():
    """Choose between the bundled four-file dataset and the legacy JSON format."""
    if USE_LEGACY_JSON:
        return apply_missing_abstract_policy(load_json(LEGACY_DATA_JSON), MISSING_ABSTRACT)
    return load_faculty_files(NODES_CSV, AUTHORSHIPS_CSV, PAPERS_JSONL_GZ,
                              FACULTY_EDGES_CSV, missing_abstract=MISSING_ABSTRACT)


def main():
    """Run using the editable SETTINGS section above (no CLI arguments)."""
    if MISSING_ABSTRACT not in MISSING_ABSTRACT_CHOICES:
        raise ValueError(f"MISSING_ABSTRACT must be one of {MISSING_ABSTRACT_CHOICES}")
    if ACTION not in ("recommend", "inspect-data", "decompose", "evolve"):
        raise ValueError(f"Unknown ACTION: {ACTION!r}")

    if ACTION == "inspect-data":
        data = _load_configured_data()
        if USE_LEGACY_JSON:
            print(f"Validated legacy faculty JSON: {LEGACY_DATA_JSON}")
            print(f"Professors: {len(data['professors'])}")
        else:
            print(json.dumps(data["source_summary"], indent=2))
        print("Input validation successful. No Qwen3 model was loaded.")
        return

    if ACTION == "evolve":
        updated = evolve(load_json(LEGACY_DATA_JSON), load_json(EVOLVE_UPDATE_JSON))
        save_json(UTILITY_OUTPUT_JSON, updated)
        print(f"Observed snapshot updated to version {updated['version']}: {UTILITY_OUTPUT_JSON}")
        return

    # Qwen receives only query["abstract"], never papers or graph data.
    from qwen_requirements import decompose_query
    raw_query = load_json(QUERY_JSON)
    query = decompose_query(raw_query, model_name=QWEN_GENERATIVE_MODEL)
    if ACTION == "decompose":
        save_json(UTILITY_OUTPUT_JSON, query)
        print(f"Qwen3 generated {len(query['requirements'])} requirements: {UTILITY_OUTPUT_JSON}")
        return

    # For ACTION="recommend", use the Qwen response DIRECTLY (no manual merge).
    # Save the exact augmented query for traceability/review.
    generated_query_path = OUTPUT_DIR / "qwen_decomposed_query.json"
    save_json(generated_query_path, query)
    print(f"Qwen3 generated {len(query['requirements'])} requirements: {generated_query_path}")

    data = _load_configured_data()
    if not USE_LEGACY_JSON:
        print("Loaded input files:", json.dumps(data["source_summary"], indent=2))

    artifacts = recommend(data, query,
                          encoder=TextEncoder(model=QWEN_MODEL),
                          max_additional=MAX_ADDITIONAL,
                          lambda_network=LAMBDA_NETWORK,
                          threshold_low=THRESHOLD_LOW,
                          threshold_high=THRESHOLD_HIGH,
                          evidence_label_threshold=EVIDENCE_LABEL_THRESHOLD,
                          restarts=RESTARTS, top_r=TOP_R, max_rounds=MAX_ROUNDS,
                          science_tolerance=SCIENCE_TOLERANCE, n_teams=N_TEAMS,
                          exact_check_max_candidates=EXACT_CHECK_MAX_CANDIDATES,
                          embedding_cache_dir=EMBEDDING_CACHE_DIR if USE_EMBEDDING_CACHE else None,
                          topic_cluster_dir=PRECOMPUTED_TOPIC_CLUSTERS_DIR,
                          seed=SEED)
    export_results(OUTPUT_DIR, artifacts)
    result = artifacts["result"]
    print(f"Found {result['discovered_unique_team_count']} unique teams; returned {len(result['recommended_teams'])}")
    for team in result["recommended_teams"]:
        members = "; ".join(
            f"{p['name']} ({p['departments'] or 'department unavailable'}, {p['id']})"
            for p in team["professors"])
        print(f"  #{team['rank']} {members}: "
              f"science={team['scientific_score']:.3f} "
              f"brokerage={team['brokerage_score']:.3f} "
              f"combined={team['combined_score']:.3f}")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
