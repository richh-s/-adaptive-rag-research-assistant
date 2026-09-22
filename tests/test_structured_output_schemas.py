"""Tool-calling models occasionally return a list argument as a JSON string. Seen from Claude
on a real eval run: decomposition failed outright and grading silently fell back to trusting
retrieval. These pin the shapes that must parse, and the ones that must still be rejected."""

import pytest
from pydantic import ValidationError

from rag_assistant.schemas.models import DocGradeBatch, SubQueries

GRADES = [{"relevant": True, "score": 0.9}, {"relevant": False, "score": 0.1}]


def test_grades_parse_from_a_real_list():
    assert len(DocGradeBatch(grades=GRADES).grades) == 2


def test_grades_parse_when_the_whole_object_is_re_encoded():
    """The exact shape observed: the field holds the entire object as a string."""
    batch = DocGradeBatch(
        grades='{"grades":[{"relevant":true,"score":0.9},{"relevant":false,"score":0.1}]}'
    )

    assert [g.relevant for g in batch.grades] == [True, False]


def test_grades_parse_when_just_the_list_is_encoded():
    batch = DocGradeBatch(grades='[{"relevant": true, "score": 0.9}]')

    assert batch.grades[0].score == 0.9


def test_sub_queries_parse_when_encoded():
    assert SubQueries(sub_queries='["a?", "b?"]').sub_queries == ["a?", "b?"]
    assert SubQueries(sub_queries='{"sub_queries": ["a?"]}').sub_queries == ["a?"]


@pytest.mark.parametrize("bad", ["not json", '{"other": [1]}', '"a string"', "42"])
def test_undecodable_strings_are_still_rejected(bad):
    """The decoding must not turn garbage into an empty-but-valid result."""
    with pytest.raises(ValidationError):
        SubQueries(sub_queries=bad)
