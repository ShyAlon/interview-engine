"""Readable binding-upgrade errors, never an implicit migration."""
from pathlib import Path
from .errors import UserFacingError

class BindingMigrationRequired(UserFacingError):
    def __init__(self,path,saved,current):
        self.path,self.saved,self.current=Path(path),saved,current
        super().__init__('Session needs binding migration.',f'Saved contract {saved.contract_version}; installed contract {current.contract_version}. Coverage identity is unchanged.',
            'Stop the services, then run python scripts/migrate-session-bindings.py --artifacts-root <artifacts> --session '+saved.session_id+' to preview. Use --apply --services-stopped only after reviewing the preview and backup plan.')

def identity(binding):return binding.session_id,binding.schema_id,binding.schema_version

class BindingIdentityMismatch(UserFacingError,ValueError):
    def __init__(self):
        super().__init__('Session schema binding is immutable: coverage identity differs.',
            'The saved session, schema ID or coverage schema version differs from this runtime.',
            'Do not run the stamp-only migration. Keep the original files and review the schema mismatch.')
