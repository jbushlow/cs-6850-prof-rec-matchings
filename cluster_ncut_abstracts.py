
#!/usr/bin/env python3
"""Offline NCUT clustering of the document embeddings from embed_qwen_abstracts(2).py.

No Qwen inference, CSV loading, or command-line flags. Run once after the
embedding script, then set USE_PRECOMPUTED_TOPIC_CLUSTERS=True in sherlocking.py.

Input (exact embed_qwen_abstracts output):
    embedding_cache/<model>/paper_retrieval_v1/
        manifest.json
        documents/index.jsonl
        documents/shards/part_XXXXX.pt

Output (not written into the embedding cache's source indexes or shards):
    embedding_cache/<model>/paper_retrieval_v1/topic_clusters_ncut_v1/
        manifest.json                  # provenance and hyperparameters
        paper_clusters.jsonl           # paper ID / text hash / cluster ID
        centroids.npy                  # normalized mean *Qwen* vectors per cluster
        spectral_embeddings.npy        # NCUT spectral coordinates per paper
        cluster_summary.json           # sizes + representative paper titles

NCUT is computed on the Qwen document embeddings (title + abstract or title
fallback), NOT passage embeddings. Cluster centroids deliberately remain in
Qwen embedding space so sherlocking.py can compare requirement query embeddings
with these centroids; spectral eigenvectors cannot be compared to Qwen queries.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sys

# Your project directory
BASE_DIR = Path("/home/jiasheng/CS6850/prof_rec_colab")

# Existing NCUT repository
NCUT_REPO = Path("/home/jiasheng/ncut_pytorch")

# Make the existing repository importable
sys.path.insert(0, str(NCUT_REPO))
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # MUST match embed_qwen_abstracts(2).py
CACHE_NAME = "paper_retrieval_v1"           # or paper_retrieval_v1_sample_<LIMIT>
EMBEDDING_CACHE_DIR = BASE_DIR / "embedding_cache" / QWEN_MODEL.rsplit("/", 1)[-1] / CACHE_NAME
OUTPUT_DIR = EMBEDDING_CACHE_DIR / "topic_clusters_ncut_v1"

N_CLUSTERS = 50                 # coarse topics, number is a research hyperparameter
N_EIG = 50                  # number of spectral eigenvectors, >= N_CLUSTERS
NCUT_AFFINITY_SIGMA = 0.5      # cosine affinity strength (ncut-pytorch)
NYSTROM_SAMPLES = 59878         # representative points, 2048 limits quadratic workspace
NYSTROM_PROPAGATION_SAMPLES = 14969
NYSTROM_NEIGHBORS = 32
DEVICE = "auto"                # "auto", "cuda", or "cpu"; GPU recommended for 100k
SEED = 67
OVERWRITE = True              # true replaces ONLY OUTPUT_DIR, not embedding cache
# ==========================================================================


def fingerprint_documents(rows) -> str:
    """Hash ordered (paper_id, text_sha256) pairs, common to writer and reader."""
    digest = hashlib.sha256()
    for paper_id, text_hash in rows:
        digest.update(json.dumps([paper_id, text_hash], ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_document_cache(root: Path):
    """Read only unique-paper document shards, in index order, checking hashes."""
    import torch

    manifest_file = root / "manifest.json"
    index_file = root / "documents" / "index.jsonl"
    if not manifest_file.is_file() or not index_file.is_file():
        raise FileNotFoundError("Run embed_qwen_abstracts(2).py first: missing "
                                f"{manifest_file} or {index_file}")
    source_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (source_manifest.get("status") != "complete"
            or source_manifest.get("model") != QWEN_MODEL
            or source_manifest.get("role") != "document"
            or not source_manifest.get("normalized")
            or source_manifest.get("format_version") != 1):
        raise ValueError("Source embedding cache is incomplete or incompatible with settings")
    with index_file.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len(rows) != source_manifest.get("document_count"):
        raise ValueError("Document index count disagrees with embedding manifest")
    if len({r["paper_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate paper IDs in document index")
    if any(r["entry_id"] != r["paper_id"] for r in rows):
        raise ValueError("Document entry_id must equal paper_id")

    # The writer sorts paper IDs before writing document shards; preserve the
    # index order here for exact ID -> embedding alignment.
    vectors = None
    loaded_shard = None
    loaded_shard_path = None
    for i, row in enumerate(rows):
        shard_relative = Path(row["shard"])
        if (shard_relative.is_absolute() or ".." in shard_relative.parts
                or shard_relative.parts[:2] != ("documents", "shards")):
            raise ValueError(f"Invalid document shard path {shard_relative}")
        if shard_relative != loaded_shard_path:
            loaded_shard = torch.load(root / shard_relative, map_location="cpu", weights_only=True)
            loaded_shard_path = shard_relative
        j = int(row["row"])
        if (j < 0 or j >= len(loaded_shard["entry_ids"])
                or loaded_shard["entry_ids"][j] != row["entry_id"]
                or loaded_shard["text_sha256"][j] != row["text_sha256"]):
            raise ValueError(f"Document embedding index mismatch: {row['paper_id']}")
        v = loaded_shard["embeddings"][j]
        if vectors is None:
            if v.ndim != 1 or v.numel() == 0:
                raise ValueError("Invalid first document embedding")
            vectors = np.empty((len(rows), v.numel()), dtype=np.float32)
        if v.ndim != 1 or v.numel() != vectors.shape[1]:
            raise ValueError(f"Inconsistent dimension for {row['paper_id']}")
        vectors[i] = v.to(torch.float32).numpy()
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite document embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-10):
        raise ValueError("Zero-length document embeddings")
    vectors /= norms[:, None]
    return rows, vectors, source_manifest


def run():
    import torch

    if N_CLUSTERS < 2 or N_EIG < N_CLUSTERS or NYSTROM_SAMPLES < 2:
        raise ValueError("Require N_CLUSTERS >=2, N_EIG >= N_CLUSTERS, NYSTROM_SAMPLES >=2")
    if DEVICE not in ("auto", "cuda", "cpu"):
        raise ValueError("DEVICE must be auto, cuda or cpu")
    if NYSTROM_PROPAGATION_SAMPLES < 2 or NYSTROM_NEIGHBORS < 1:
        raise ValueError("Invalid Nyström propagation settings")
    if OUTPUT_DIR.exists() and not OVERWRITE:
        raise FileExistsError(f"{OUTPUT_DIR} already exists; set OVERWRITE=True to regenerate")

    root = EMBEDDING_CACHE_DIR.expanduser().resolve()
    print(f"Reading Qwen paper DOCUMENT embeddings: {root}", flush=True)
    index_rows, vectors, source_manifest = load_document_cache(root)
    n, dim = vectors.shape
    if n < 2:
        raise ValueError("Need at least two unique papers to cluster")
    k = min(N_CLUSTERS, n)
    neig = min(max(N_EIG, k), n)
    device = "cuda" if DEVICE == "auto" and torch.cuda.is_available() else ("cpu" if DEVICE == "auto" else DEVICE)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but no CUDA device is available")
    print(f"Clustering {n:,} papers, dimension {dim}, into {k} NCUT clusters on {device}", flush=True)

    try:
        from ncut_pytorch import Ncut, kway_ncut
        from ncut_pytorch.utils.math import cosine_affinity
    except ImportError as exc:
        raise RuntimeError("Missing ncut_pytorch. Install with: pip install -U ncut-pytorch") from exc

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    with torch.inference_mode():
        # Keep the original X on CPU; the library uses its own device/chunking.
        features = torch.from_numpy(vectors)
        ncut = Ncut(n_eig=neig, device=device, affinity_fn=cosine_affinity,
                    sigma=NCUT_AFFINITY_SIGMA,
                    n_sample=min(NYSTROM_SAMPLES, n),
                    n_sample2=min(NYSTROM_PROPAGATION_SAMPLES, n),
                    n_neighbors=NYSTROM_NEIGHBORS)
        spectral = ncut.fit_transform(features)
        if spectral.ndim != 2 or spectral.shape != (n, neig):
            raise ValueError(f"Unexpected NCUT eigenvector shape: {spectral.shape}")
        # Official ncut_pytorch procedure: discretize spectral coordinates,
        # then take argmax; never K-means on raw Qwen embeddings.
        discrete = kway_ncut(spectral[:, :k], n_clusters=k, device=device)
        if discrete.shape != (n, k):
            raise ValueError(f"Unexpected NCUT discrete shape: {discrete.shape}")
        labels = discrete.argmax(dim=1).cpu().numpy().astype(np.int32)
        # Retain the NCUT coordinates for downstream visualization before the
        # torch tensor is released. Rows remain aligned with documents/index.jsonl.
        spectral_embeddings = spectral.detach().cpu().numpy().astype(np.float32, copy=True)
    del features, spectral, discrete, ncut
    if not np.isfinite(labels).all() or not np.isfinite(spectral_embeddings).all():
        raise ValueError("Invalid NCUT labels or spectral embeddings")
    # K-way may leave empty groups; compress IDs and reflect actual nonempty k.
    occupied, labels = np.unique(labels, return_inverse=True)
    actual_k = len(occupied)
    if actual_k < 2:
        raise RuntimeError("NCUT collapsed to one cluster. Change NCUT parameters and rerun.")
    print(f"Found {actual_k} nonempty clusters", flush=True)

    # NCUT eigenvectors are not in Qwen's feature space. Save the average
    # original Qwen embedding per topic so that new projects can compare
    # their Qwen query embeddings with Qwen document centroids.
    centers = np.zeros((actual_k, dim), dtype=np.float32)
    np.add.at(centers, labels, vectors)
    counts = np.bincount(labels, minlength=actual_k)
    centers /= counts[:, None]
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-10)

    summary = []
    for c in range(actual_k):
        members = np.flatnonzero(labels == c)
        similarities = vectors[members] @ centers[c]
        top = members[np.argsort(-similarities, kind="stable")[:5]]
        summary.append({
            "cluster_id": c, "paper_count": int(counts[c]),
            "representative_papers": [
                {"paper_id": index_rows[i]["paper_id"],
                 "title": index_rows[i].get("title", ""),
                 "cosine_to_mean": round(float(vectors[i] @ centers[c]), 6)}
                for i in top],
        })

    output = OUTPUT_DIR.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f".staging_{os.getpid()}")
    if stage.exists():
        raise FileExistsError(f"Staging directory already exists: {stage}")
    stage.mkdir()
    try:
        with (stage / "paper_clusters.jsonl").open("w", encoding="utf-8") as stream:
            for row, c in zip(index_rows, labels):
                stream.write(json.dumps({"paper_id": row["paper_id"],
                                         "text_sha256": row["text_sha256"],
                                         "cluster_id": int(c)}, ensure_ascii=False) + "\n")
        np.save(stage / "centroids.npy", centers, allow_pickle=False)
        np.save(stage / "spectral_embeddings.npy", spectral_embeddings, allow_pickle=False)
        (stage / "cluster_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = {
            "format_version": 1,
            "method": "ncut_pytorch_nystrom_kway",
            "model": QWEN_MODEL,
            "document_count": n,
            "embedding_dimension": dim,
            "requested_clusters": N_CLUSTERS,
            "actual_clusters": actual_k,
            "n_eig": neig,
            "spectral_embedding_dimension": spectral_embeddings.shape[1],
            "affinity": "cosine",
            "affinity_sigma": NCUT_AFFINITY_SIGMA,
            "nystrom_samples": NYSTROM_SAMPLES,
            "nystrom_propagation_samples": NYSTROM_PROPAGATION_SAMPLES,
            "nystrom_neighbors": NYSTROM_NEIGHBORS,
            "seed": SEED,
            "device_used": device,
            "source_embedding_cache": str(root),
            "source_embedding_fingerprint_sha256": source_manifest.get("source_fingerprint_sha256"),
            "document_fingerprint_sha256": fingerprint_documents(
                (r["paper_id"], r["text_sha256"]) for r in index_rows),
            "centroid_space": "normalized_mean_original_qwen_document_embeddings",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"DONE. Wrote {n:,} paper assignments to {output}", flush=True)
    return output


if __name__ == "__main__":
    run()
