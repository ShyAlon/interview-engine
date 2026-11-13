"""The interview producer shares the source-snapshot contract with consumers."""
from pathlib import Path
import hashlib,zipfile
from vmodel_contracts import CONTRACT_VERSION,ArchitectureDecision

def test_citation_contract_is_exactly_pinned_and_legacy_is_readable():
 root=Path(__file__).parents[1];wheel=root/'vendor/vmodel_contracts-0.29.0-py3-none-any.whl'
 assert CONTRACT_VERSION=='0.29.0'
 assert hashlib.sha256(wheel.read_bytes()).hexdigest() in (root/'uv.lock').read_text()
 with zipfile.ZipFile(wheel) as z:assert 'vmodel_contracts/citations.py' in z.namelist()
 old=ArchitectureDecision.model_validate({'contract_version':'0.28.0','decision_id':'ADR-1','title':'Old','decision':'Old','rationale':'Old','driver':'simplicity','options_considered':[]})
 assert old.founder_citations==[] and old.contract_version=='0.28.0'
