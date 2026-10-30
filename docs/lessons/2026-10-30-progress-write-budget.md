# Do not store a running clock as durable progress

Owner asked to minimize writes end-to-end, not merely batch delivery copies. Application persisted graph stages plus model heartbeat text and finished-call timing into the fenced source journal. Each observational append costs one transaction and three data queries. Model diagnostics and usage already retain the call evidence.

Application now keeps stage/item boundaries, coverage explanations and failures but excludes heartbeat events. Its elapsed-time and successful-call-summary subscribers do not journal. Console diagnostics, UsageLog/accounting and lease renewal stay unchanged. The lower-level journal still accepts legacy heartbeat pages, so cursor/protocol/fencing tests are retained, not weakened. Two former application tests requiring call-summary journal rows now assert their deliberate absence; new filter tests fail before the change.

Paired delivery change atomically consumes pages and filters legacy transient copies. Combined owned-PG test in delivery exercises both repos' code under restricted roles, comparing the same five attempted signals. Source rows5to2, source data queries15to6, source transactions5to2; no provider calls. Never infer full-answer speedup from this page benchmark.
