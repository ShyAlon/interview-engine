#!/usr/bin/env python3
"""Validate completed coverage labels without changing runtime configuration."""
import argparse,csv
from interview_engine.categorization import COVERAGE_AREAS
p=argparse.ArgumentParser();p.add_argument('csv');a=p.parse_args();errors=[];count=0
with open(a.csv,newline='') as f:
 for i,row in enumerate(csv.DictReader(f),2):
  count+=1
  if row['human_relevance'] not in {'addressed','partial','dodged'}:errors.append(f'row {i}: invalid human_relevance')
  facets={x.strip() for x in row['human_covered_facets'].split(',') if x.strip()};unknown=facets-set(COVERAGE_AREAS)
  if unknown:errors.append(f'row {i}: unknown facets {sorted(unknown)}')
  if not row['reviewer_id'].strip():errors.append(f'row {i}: reviewer_id is required')
if errors:raise SystemExit('\n'.join(errors))
print(f'validated {count} labeled coverage decisions')
