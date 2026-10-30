# Bind global engine configuration to the host, not process-global files

The hosted PG engine uses a tenant registry router. Its question-quality HTTP
routes and graph helpers still read the module-global ConfigStore, which was
memory by default or a file with an explicit environment override. That made
settings differ after host replacement or between otherwise identical hosts.

PG hosts now inject one fixed global configuration row through the engine's
context-bound interface. Hosted and standalone engines share their existing
pool. HTTP worker threads inherit the app binding; producer worker entry binds
explicitly and restores the previous context afterward. No config is imported
from old files. A missing row means code constants; corrupt or unavailable PG
is an error, never an automatic file/memory/default fallback.

The new role can SELECT only the fixed quality configuration table. It cannot
write configuration, read customers or credits, or grant itself other roles.
The existing cloud UI/API fence remains unchanged. Only trusted database-admin
maintenance can set this row for now. Runtime global editing is a separate
product/authority decision: the Python services.operate name alone is not a
reviewed database capability, and this PR neither adds nor grants that power.
This is durable read consistency, not a restored cloud settings editor.

Rollout requires the matching engine binding PR, the fixed-row migration and
SET membership in requi_engine_config_read. Explicit legacy quality-file
configuration is refused for a PG host. Startup probes table access and validates
a saved body before admitting work. It changes no data.

Fail-first tests cover config/app isolation, safe unavailable/read-only errors,
real PG visibility across hosts/restarts/threads, corrupt config rejection,
service-token enforcement, direct SQL write denial and public-role read denial.
Standalone tests verify one lazy pool and no files. A real producer binding test
reads PG instead of an ambient store and restores the previous store on exit.

Startup's exact successful check count changes from eleven to twelve, and the
crash matrix adds the new required config-store check. The table inventory and
role-access map include the new fixed table. Existing assertions remain intact.
No old implementation is removed. No live DB mutation, task reset, deploy or
spend is part of this work. Hosted CI remains unverified.
