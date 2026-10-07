import pytest
from langchain_core.messages import AIMessage, HumanMessage
from interview_engine.distill_qa import (reference_name_reason,
    reject_non_founder_reference_citations, verify_reference_names)

MESSAGES = [HumanMessage(content='Build a Virtual DM'),
            AIMessage(content='Should it integrate with D&D Beyond?'),
            HumanMessage(content='No, everything should be baked in.')]
ITEM = {'name': 'D&D Beyond', 'kind': 'reference_product', 'source_turns': [1]}


def test_interviewer_citation_is_producer_defect_with_exact_quote():
    with pytest.raises(ValueError) as error:
        reject_non_founder_reference_citations([ITEM], MESSAGES)
    reason = str(error.value)
    assert 'interviewer, not you' in reason
    assert 'Turn 1 (Interviewer): "Should it integrate with D&D Beyond?"' in reason
    assert verify_reference_names([ITEM], {0: MESSAGES[0].content, 2: MESSAGES[2].content})[0] == []


@pytest.mark.parametrize('text,passes', [('D&D Beyond', True), ('d&d beyond', True),
    ('D&D   Beyond', True), ('DnD Beyond', False), ('D and D Beyond', False),
    ('D & D Beyond', False), ('D&D-Beyond', False)])
def test_matching_protection_unchanged(text, passes):
    item = {**ITEM, 'source_turns': [0]}
    messages = [HumanMessage(content=text)]
    reject_non_founder_reference_citations([item], messages)
    assert bool(verify_reference_names([item], {0: text})[0]) is passes
    if not passes:
        assert f'Turn 0 (You): "{text}"' in reference_name_reason(item, messages)


def test_other_founder_turn_is_not_silently_accepted():
    item = {**ITEM, 'source_turns': [2]}
    messages = [HumanMessage(content='D&D Beyond'), MESSAGES[1], MESSAGES[2]]
    assert verify_reference_names([item], {0: messages[0].content, 2: messages[2].content})[0] == []
    assert 'No, everything should be baked in.' in reference_name_reason(item, messages)


@pytest.mark.parametrize('field', ['stakeholder_needs', 'assumption_items', 'tech_preferences',
                                   'reference_names', 'scope_exclusions', 'done_criteria'])
def test_every_cited_list_field_rejects_interviewer(field):
    from interview_engine.distill_qa import reject_non_founder_citations
    with pytest.raises(ValueError, match='non-founder evidence'):
        reject_non_founder_citations({field: [{'statement': 'example', 'source_turns': [1]}]}, MESSAGES)
    reject_non_founder_citations({field: [{'statement': 'example', 'source_turns': [2]}]}, MESSAGES)


@pytest.mark.parametrize('turns', [[1], [True], [99], [-1], ['2']])
def test_framing_map_invalid_citations_rejected_before_coercion(turns):
    from interview_engine.distill_qa import reject_non_founder_citations
    with pytest.raises(ValueError, match='non-founder evidence'):
        reject_non_founder_citations({'source_turns': {'problem_statement': turns}}, MESSAGES)
