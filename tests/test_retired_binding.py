import json
import pytest
from interview_engine.evidence import EvidenceLake
from interview_engine.binding_compat import BindingMigrationRequired,BindingIdentityMismatch


def test_legacy_binding_still_refuses_mutation_without_migration_route(tmp_path):
    lake=EvidenceLake(tmp_path,'task');binding=lake.bind_session('s')
    path=lake.paths.run/'session_id=s/schema-binding.json'
    body=json.loads(path.read_text());body['contract_version']='0.19.0';path.write_text(json.dumps(body));before=path.read_bytes()
    with pytest.raises(BindingMigrationRequired) as exc:lake.bind_session('s')
    assert 'migration is retired' in exc.value.render()
    assert path.read_bytes()==before
    body['schema_version']='2.0.0';path.write_text(json.dumps(body));before=path.read_bytes()
    with pytest.raises(BindingIdentityMismatch):lake.bind_session('s')
    assert path.read_bytes()==before
