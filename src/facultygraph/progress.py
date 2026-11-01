"""Small terminal-aware progress helpers shared by pipeline stages."""
from __future__ import annotations

import sys
from tqdm.auto import tqdm


def progress_enabled(settings=None):
    """Show progress by default only on an interactive terminal."""
    configured = (settings or {}).get("progress")
    if configured is not None:
        return bool(configured)
    return sys.stderr.isatty()


def progress_bar(*, settings=None, **kwargs):
    return tqdm(
        disable=not progress_enabled(settings),
        dynamic_ncols=True,
        **kwargs,
    )


def progress_iter(iterable, *, settings=None, **kwargs):
    return tqdm(
        iterable,
        disable=not progress_enabled(settings),
        dynamic_ncols=True,
        **kwargs,
    )
