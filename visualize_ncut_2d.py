#!/usr/bin/env python3
"""Interactive 2D map of actual NCUT paper assignments.

Usage:
  pip install umap-learn plotly
  python visualize_ncut_2d.py --cache-dir embedding_cache/Qwen3-Embedding-0.6B/paper_retrieval_v1

Default projects Qwen document embeddings (requires no NCUT rerun).
Add --space spectral to project spectral_embeddings.npy saved during clustering.
"""
import argparse
import json
from pathlib import Path
import numpy as np

"""
CUDA_VISIBLE_DEVICES=1 python visualize_ncut_2d.py \
  --cache-dir embedding_cache/Qwen3-Embedding-0.6B/paper_retrieval_v1 \
  --space spectral
"""

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--space', choices=['qwen', 'spectral'], default='qwen')
    ap.add_argument('--neighbors', type=int, default=30)
    ap.add_argument('--min-dist', type=float, default=0.10)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    root = args.cache_dir.resolve()
    clusters = root / 'topic_clusters_ncut_v1'
    rows = [json.loads(s) for s in (root / 'documents/index.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    c_rows = [json.loads(s) for s in (clusters / 'paper_clusters.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    c_by_id = {str(r['paper_id']): r for r in c_rows}
    if len(c_by_id) != len(c_rows) or len(rows) != len(c_rows):
        raise ValueError('Paper IDs/counts do not match document index and NCUT assignments')
    labels = []
    for r in rows:
        c = c_by_id.get(str(r['paper_id']))
        if not c or c['text_sha256'] != r['text_sha256']:
            raise ValueError('Cluster assignment does not match paper ID/text hash: ' + str(r['paper_id']))
        labels.append(int(c['cluster_id']))
    print(f'Loaded {len(rows):,} papers and {len(set(labels))} occupied NCUT clusters.', flush=True)

    if args.space == 'spectral':
        spec_path = clusters / 'spectral_embeddings.npy'
        if not spec_path.is_file():
            raise FileNotFoundError(f'{spec_path} not found. Save spectral.cpu().numpy() in the NCUT script *before* deleting spectral. See instructions.')
        X = np.load(spec_path, allow_pickle=False, mmap_mode='r')
        if X.shape[0] != len(rows):
            raise ValueError('Spectral embedding row count does not match documents index')
        X = np.asarray(X, dtype=np.float32)
    else:
        import torch
        X = None
        shard_name = None
        shard = None
        for idx, r in enumerate(rows):
            if r['shard'] != shard_name:
                shard_name = r['shard']
                shard = torch.load(root / shard_name, map_location='cpu', weights_only=True)
            j = int(r['row'])
            if shard['entry_ids'][j] != r['paper_id'] or shard['text_sha256'][j] != r['text_sha256']:
                raise ValueError('Qwen shard index mismatch at ' + str(r['paper_id']))
            v = shard['embeddings'][j].float().numpy()
            if X is None:
                X = np.empty((len(rows), len(v)), dtype=np.float32)
            X[idx] = v
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite embedding detected')
    print(f'UMAP: {X.shape[0]:,} × {X.shape[1]:,} ({args.space})...', flush=True)
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=args.neighbors,
                       min_dist=args.min_dist, metric='cosine',
                       random_state=42, low_memory=True).fit_transform(X)
    np.save(clusters / f'umap_{args.space}_2d.npy', coords)
    print('Building interactive WebGL scatterplot...', flush=True)
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    colors = qualitative.Dark24 + qualitative.Light24 + qualitative.Alphabet
    fig = go.Figure()
    labels = np.asarray(labels)
    titles = [r.get('title') or '[Untitled paper]' for r in rows]
    ids = [str(r['paper_id']) for r in rows]
    for cl in sorted(set(labels)):
        selected = np.flatnonzero(labels == cl)
        fig.add_trace(go.Scattergl(
            x=coords[selected, 0], y=coords[selected, 1], mode='markers',
            name=f'Cluster {cl} ({len(selected):,})',
            marker=dict(size=3.0, opacity=0.65, color=colors[cl % len(colors)]),
            customdata=[[ids[i], titles[i][:150]] for i in selected],
            hovertemplate='Cluster '+str(cl)+'<br>%{customdata[1]}<br>Paper ID: %{customdata[0]}<extra></extra>',
        ))
    fig.update_layout(title=f'NCUT paper clusters — {len(rows):,} papers, {len(set(labels))} clusters — UMAP of {args.space} features',
        template='plotly_white', height=900,
        xaxis=dict(title='UMAP 1', showgrid=False, zeroline=False),
        yaxis=dict(title='UMAP 2', showgrid=False, zeroline=False, scaleanchor='x'),
        legend=dict(title='Click to hide/show clusters', itemsizing='constant'),
        margin=dict(l=60,r=20,t=75,b=60))
    out = args.output or (clusters / f'ncut_2d_{args.space}.html')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs='cdn', full_html=True)
    print(f'Saved: {out} ({out.stat().st_size/1024/1024:.1f} MiB)', flush=True)
    print('Tip: Legend-click to hide a cluster; double-click a legend item to isolate it.', flush=True)

if __name__ == '__main__':
    main()
