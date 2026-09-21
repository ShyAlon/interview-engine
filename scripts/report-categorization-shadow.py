#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from interview_engine.categorization_shadow import render_report
p=argparse.ArgumentParser(description="Regenerate a Jev shadow Markdown report from append-only JSONL")
p.add_argument("jsonl",nargs="+",help="one or more shadow JSONL files")
p.add_argument("-o","--output",required=True)
a=p.parse_args();Path(a.output).write_text(render_report(a.jsonl)+"\n")
print(a.output)
