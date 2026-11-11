from types import SimpleNamespace
import hashlib
import pytest
from interview_engine.pg_documents import InterviewDocuments


def documents(raw):
    claim=SimpleNamespace(session_id='s',owner_id='owner',generation=1,epoch=2,instance='child')
    rows=[{'ordinal':0,'total':0,'payload':raw.hex(),'sha256':hashlib.sha256(raw).hexdigest()}]
    store=SimpleNamespace(claims=('owner','auth'),_answer_schema_binding=('s','owner',('owner','auth'),1,2,'child',rows))
    docs=InterviewDocuments(store,claim,'task')
    def forbidden():raise AssertionError('ordinary guarded bind transaction required')
    docs.transaction=forbidden
    return docs,rows


def test_claimed_schema_binding_reuses_only_exact_verified_bytes_and_identity():
    docs,rows=documents(b'{"schema":"one"}')
    docs.bind('evidence/schema-binding.json',b'{"schema":"one"}')
    with pytest.raises(ValueError,match='schema binding changed'):docs.bind('evidence/schema-binding.json',b'{"schema":"two"}')
    rows[0]['sha256']='0'*64
    with pytest.raises(ValueError,match='hash differs'):docs.bind('evidence/schema-binding.json',b'{"schema":"one"}')
    docs,rows=documents(b'one');docs.claim.epoch=3
    with pytest.raises(AssertionError,match='guarded bind'):docs.bind('evidence/schema-binding.json',b'one')


def test_chunked_claimed_schema_binding_preserves_sequence_check():
    docs,rows=documents(b'one')
    rows[0]['total']=1
    with pytest.raises(ValueError,match='chunk sequence'):docs.bind('evidence/schema-binding.json',b'one')
