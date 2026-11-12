# Batch fixtures must match the composed execution path

Scenario and option cuts were tested separately. Their combined tree must use
both complete scenario and complete option replies in its fake-model fixtures.
Keep exact call-count, field-parity, repair-target and identity assertions.
Nine scenarios plus two options now require eleven fake calls, not fifteen or
twenty-nine. This is a deterministic fake-input count, not a production time or
cost prediction. Run the exact composed main gate before handing off a merge.
