# Operator-diagnostic omission needs its own safe evidence

The start failure was logged, but an oversized diagnostic block or a build failure
silently omitted the header. Log a fixed omission reason and safe category without
the diagnostic body, raw exception or traceback. Preserve the service-token boundary,
minimal customer 503 and retry contract. Failing-first tests cover both omissions;
existing PostgreSQL header/security tests remain unchanged.
