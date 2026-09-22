#!/usr/bin/env python3
"""Find session checkpoints/shadow logs under artifact roots and enrich each pair."""
import argparse
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('enrich_one',Path(__file__).with_name('enrich-categorization-shadow.py'));mod=module_from_spec(spec);spec.loader.exec_module(mod)
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--output-dir',required=True);a=p.parse_args()
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);written=[]
for root in map(Path,a.roots):
 for shadow in root.rglob('categorization-shadow.jsonl'):
  checkpoint=shadow.parent/'checkpoints.sqlite'
  if not checkpoint.is_file():
   print(f'skip {shadow}: no sibling checkpoints.sqlite');continue
  session=shadow.parent.parent.name
  target=out/f'{session}.categorization-shadow.enriched.jsonl'
  turns,submission=mod._turns_from_checkpoint(checkpoint);mod.enrich(shadow,target,turns,submission);written.append(target)
print(f'wrote {len(written)} enriched files to {out}')
