"""Command-line pipeline runner. `facultygraph --help`."""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import yaml
from .http import Http, OpenAlex
from .faculty import collect_faculty
from .openalex import collect_openalex
from .abstracts import AbstractCollector
from .graph import build_graph


def main(argv=None):
    p = argparse.ArgumentParser(prog="facultygraph",description="Public faculty rosters → OpenAlex papers → abstracts → coauthorship graphs")
    p.add_argument("command",choices=["faculty","openalex","abstracts","graph","all"],help="stage or all stages in sequence")
    p.add_argument("--config",default="configs/example.yaml")
    p.add_argument("--refresh-http",action="store_true",help="Ignore persistent HTTP cache on this run")
    p.add_argument("--retry-failed-abstracts",action="store_true",help="Retry cached missing abstracts")
    p.add_argument("--no-progress",action="store_true",help="Disable interactive progress bars")
    p.add_argument("--verbose",action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,format="%(levelname)s %(message)s")
    path = Path(args.config).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = cfg.setdefault("settings",{})
    if args.no_progress:
        settings["progress"] = False
    out = Path(settings.get("output_dir","output")).resolve()
    cache = Path(settings.get("cache_dir",".cache/http")).resolve()
    email = settings.get("contact_email") or os.getenv("RESEARCH_CONTACT_EMAIL","")
    http = Http(cache,email,refresh=args.refresh_http)
    oa = OpenAlex(http)
    if args.command in ("faculty","all"):
        collect_faculty(cfg,http,path.parent,out)
    if args.command in ("openalex","all"):
        collect_openalex(cfg,oa,out)
    if args.command in ("abstracts","all"):
        AbstractCollector(http,out,settings).run(retry_failed=args.retry_failed_abstracts)
    if args.command in ("graph","all"):
        build_graph(out,settings)
    print(f"Finished {args.command}. See {out}")

if __name__ == "__main__": main()
